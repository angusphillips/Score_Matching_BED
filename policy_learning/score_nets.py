from collections.abc import Callable
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from check_shapes import check_shapes
from flax.linen.initializers import normal
from jaxtyping import Array

from policy_learning.utils.func import _apply_score_net_parameterisation
from policy_learning.utils.logging import instantiate

gelu_init = nn.initializers.variance_scaling(
    scale=1.0, mode="fan_avg", distribution="truncated_normal"
)

relu_init = nn.initializers.variance_scaling(
    scale=2.0, mode="fan_in", distribution="truncated_normal"
)


def sinusoidal_encoding(seq_len, dim):
    position = jnp.arange(seq_len)[:, None]
    # Calculate number of sin/cos pairs needed
    num_pairs = (dim + 1) // 2  # Round up for odd dimensions
    div_term = jnp.exp(jnp.arange(0, num_pairs) * -(jnp.log(10000.0) / dim))
    pe = jnp.zeros((seq_len, dim))
    # For odd dimensions, truncate the last cos value
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term)[:, : dim // 2 + dim % 2])
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term)[:, : dim // 2])
    return pe


class RFFEmbedding(nn.Module):
    """Random Fourier Features input transformation"""

    dim_in: int
    num_frequencies: int  # Number of sinusoidal frequencies
    rff_sigma: float  # Standard deviation of the initialisation for RFF

    def setup(self):
        # Store omega as a non-trainable variable
        self.omega = self.variable(
            "rff_freqs",
            "omega",
            lambda shape: normal()(self.make_rng("params"), shape) * self.rff_sigma,
            (self.dim_in, self.num_frequencies),
        )

    def __call__(self, x):
        x_proj = jnp.matmul(x, self.omega.value)  # Shape: [batch, T, num_frequencies]
        sin_embed = jnp.sin(x_proj)
        cos_embed = jnp.cos(x_proj)
        return jnp.concatenate(
            [sin_embed, cos_embed], axis=-1
        )  # Shape: [batch, T, 2*num_frequencies]


class JointEmbedMLP(nn.Module):
    """Embeds input features using RFF embeddings and an MLP"""

    embed_dim: int
    mlp_dim: int
    x_dim: int
    y_dim: int
    rff_sigma: float

    def setup(self):
        self.y_embed = RFFEmbedding(
            dim_in=self.y_dim,
            num_frequencies=self.embed_dim // 2,
            rff_sigma=self.rff_sigma,
        )
        self.x_embed = RFFEmbedding(
            dim_in=self.x_dim,
            num_frequencies=self.embed_dim // 2,
            rff_sigma=self.rff_sigma,
        )
        self.mlp = nn.Sequential(
            [
                nn.Dense(self.mlp_dim, kernel_init=gelu_init),
                nn.gelu,
                nn.Dense(self.embed_dim, kernel_init=gelu_init),
            ]
        )

    def __call__(self, y, xi):
        y_emb = self.y_embed(y)  # Shape: [batch, T, embed_dim//2]
        xi_emb = self.x_embed(xi)  # Shape: [batch, T, embed_dim//2]
        emb = jnp.concatenate([y_emb, xi_emb], axis=-1)  # Shape: [batch, T, embed_dim]
        return self.mlp(emb)  # Shape: [batch, T, embed_dim]


class TransformerBlock(nn.Module):
    """Single Transformer block with multi-head self attention and MLP."""

    model_dim: int
    num_heads: int

    def setup(self):
        self.norm1 = nn.LayerNorm()
        self.attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)
        self.norm2 = nn.LayerNorm()
        self.mlp = nn.Sequential(
            [
                nn.Dense(
                    self.model_dim * 2, kernel_init=gelu_init
                ),  # 2x expansion instead of 4x
                nn.gelu,
                nn.Dense(
                    self.model_dim, kernel_init=gelu_init
                ),  # Ensure output matches model_dim
            ]
        )

    @check_shapes(
        "z: [*batch_size, T, d_in]",
        "return: [*batch_size, T, d_out]",
    )
    def __call__(self, z, mask=None):
        # mask: [*batch, T] bool where True = valid key position.
        # Reshape to [*batch, 1, 1, T] so it broadcasts over (heads, queries).
        if mask is not None:
            attn_mask = mask[..., None, None, :]
        else:
            attn_mask = None

        attn_out = self.attn(self.norm1(z), mask=attn_mask)
        z = z + attn_out

        mlp_out = self.mlp(self.norm2(z))
        z = z + mlp_out

        return z


class TransformerStack(nn.Module):
    """Transformer block stack that passes an optional mask through every layer."""

    num_layers: int
    model_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, z, mask=None):
        for i in range(self.num_layers):
            z = TransformerBlock(
                self.model_dim, self.num_heads, name=f"block_{i}"
            )(z, mask=mask)
        return z


class HeadBlock(nn.Module):
    """Pre-norm residual MLP block: LayerNorm -> Dense -> GELU -> Dense + skip."""

    model_dim: int

    @nn.compact
    def __call__(self, x):
        h = nn.LayerNorm()(x)
        h = nn.Dense(self.model_dim, kernel_init=gelu_init)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.model_dim, kernel_init=gelu_init)(h)
        return x + h


class ResidualMLPHead(nn.Module):
    """Input projection + N residual blocks + LayerNorm + output projection.

    Drop-in replacement for the default 2-layer head when the per-position
    output must approximate a nonlinear function (e.g. inverting dynamical
    drift terms for cart-pole / double-pendulum).
    """

    num_blocks: int
    model_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.model_dim, kernel_init=gelu_init, name="input_proj")(x)
        for i in range(self.num_blocks):
            x = HeadBlock(model_dim=self.model_dim, name=f"block_{i}")(x)
        x = nn.LayerNorm(name="output_norm")(x)
        x = nn.Dense(self.out_dim, kernel_init=gelu_init, name="output_proj")(x)
        return x


class TransformerModelMLPEmbed(nn.Module):
    num_layers: int
    model_dim: int
    num_heads: int
    embed_mlp_dim: int
    embed_dim: int
    output_shape: int
    x_dim: int
    y_dim: int
    T: int
    rff_sigma: float
    std_stats: dict
    noise_sigma: Array
    positional: bool = False
    head_mode: str = "base"  # unused here...
    y_residual_mode: str = "default"
    noise_condition_mode: str = "none"
    xi_output_mode: str = "none"
    y_output_mode: str = "none"

    def setup(self):
        self.embed = JointEmbedMLP(
            embed_dim=self.embed_dim,
            mlp_dim=self.embed_mlp_dim,
            x_dim=self.x_dim,
            y_dim=self.y_dim,
            rff_sigma=self.rff_sigma,
        )
        self.embed_proj = nn.Dense(self.model_dim, kernel_init=gelu_init)

        if self.positional:
            self.positional_encoding = sinusoidal_encoding(
                seq_len=self.T, dim=self.model_dim
            )  # Shape: [T, model_dim]

        self.transformer_stack = TransformerStack(
            num_layers=self.num_layers,
            model_dim=self.model_dim,
            num_heads=self.num_heads,
        )

        self.final_norm = nn.LayerNorm()  # Final norm for pre-norm architecture
        self.final_layer = nn.Dense(features=self.output_shape, kernel_init=gelu_init)

    @check_shapes(
        "y: [*batch_size, T, d_y]",
        "xi: [*batch_size, T, d_xi]",
        "return: [*batch_size, T, d_out]",
    )
    def __call__(self, y, xi, aux_data: dict | None = None, **kwargs):
        y_scaled = (y - self.std_stats["y_mean"]) / self.std_stats["y_std"]
        xi_scaled = (xi - self.std_stats["xi_mean"]) / self.std_stats["xi_std"]
        z = self.embed(y_scaled, xi_scaled)  # Shape: [batch_size, T, embed_dim]
        z = self.embed_proj(z)  # Shape: [batch_size, T, model_dim]
        if self.positional:
            z += self.positional_encoding

        z = self.transformer_stack(z)
        z = self.final_norm(z)

        out = self.final_layer(z)  # Shape: [batch_size, T, output_dim]

        out = _apply_score_net_parameterisation(
            out,
            y,
            xi,
            aux_data,
            noise_sigma=self.noise_sigma,
            noise_condition_mode=self.noise_condition_mode,
            xi_output_mode=self.xi_output_mode,
            y_output_mode=self.y_output_mode,
            y_residual_mode=self.y_residual_mode,
        )

        return out


class Transformer1Branch(nn.Module):
    transformer_num_layers: int
    model_dim: int
    num_heads: int
    embed_mlp_dim: int
    embed_dim: int
    output_shape: int
    x_dim: int
    y_dim: int
    T: int
    rff_sigma: float
    std_stats: dict
    noise_sigma: Array
    positional: bool = False
    head_mode: str = "base"
    # head_style: "default" = 2-layer MLP (unchanged from before); "residual_mlp" =
    # ResidualMLPHead (LayerNorm + residual blocks, deeper). Gate behind a flag so
    # existing runs reproduce bit-for-bit when head_style is left at "default".
    head_style: str = "default"
    head_num_blocks: int = 2
    y_residual_mode: str = "default"
    noise_condition_mode: str = "none"
    xi_output_mode: str = "none"
    y_output_mode: str = "none"

    def setup(self):
        self.embed = JointEmbedMLP(
            embed_dim=self.embed_dim,
            mlp_dim=self.embed_mlp_dim,
            x_dim=self.x_dim,
            y_dim=self.y_dim,
            rff_sigma=self.rff_sigma,
        )
        self.embed_proj = nn.Dense(self.model_dim, kernel_init=gelu_init)

        # Learnable global token prepended to the sequence (length becomes T + 1).
        self.global_token = self.param(
            "global_token",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.model_dim),
        )

        if self.positional:
            self.positional_encoding = sinusoidal_encoding(
                seq_len=self.T, dim=self.model_dim
            )  # Shape: [T, model_dim]

        self.transformer_stack = TransformerStack(
            num_layers=self.transformer_num_layers,
            model_dim=self.model_dim,
            num_heads=self.num_heads,
        )

        # Fresh embeddings of original (y_t, xi_t) used for head conditioning.
        self.head1_embed = JointEmbedMLP(
            embed_dim=self.embed_dim,
            mlp_dim=self.embed_mlp_dim,
            x_dim=self.x_dim,
            y_dim=self.y_dim,
            rff_sigma=self.rff_sigma,
        )
        self.head1_embed_proj = nn.Dense(self.model_dim, kernel_init=gelu_init)

        self.head2_embed = JointEmbedMLP(
            embed_dim=self.embed_dim,
            mlp_dim=self.embed_mlp_dim,
            x_dim=self.x_dim,
            y_dim=self.y_dim,
            rff_sigma=self.rff_sigma,
        )
        self.head2_embed_proj = nn.Dense(self.model_dim, kernel_init=gelu_init)

        if self.output_shape == self.y_dim + self.x_dim:
            self.head1_dim = self.y_dim
            self.head2_dim = self.x_dim
        else:
            raise ValueError("Incompatible output shape requested")

        # Normalise the concatenated head input so the head's effective gain is
        # independent of how many slabs head_mode concatenates (matters for the
        # conservative-grad wrapper: extra slabs add gradient paths into y_t).
        self.head1_input_norm = nn.LayerNorm()
        self.head2_input_norm = nn.LayerNorm()

        if self.head_style == "default":
            self.head1 = nn.Sequential(
                [
                    nn.Dense(self.model_dim, kernel_init=gelu_init),
                    nn.gelu,
                    nn.Dense(self.head1_dim, kernel_init=gelu_init),
                ]
            )
            self.head2 = nn.Sequential(
                [
                    nn.Dense(self.model_dim, kernel_init=gelu_init),
                    nn.gelu,
                    nn.Dense(self.head2_dim, kernel_init=gelu_init),
                ]
            )
        elif self.head_style == "residual_mlp":
            self.head1 = ResidualMLPHead(
                num_blocks=self.head_num_blocks,
                model_dim=self.model_dim,
                out_dim=self.head1_dim,
            )
            self.head2 = ResidualMLPHead(
                num_blocks=self.head_num_blocks,
                model_dim=self.model_dim,
                out_dim=self.head2_dim,
            )
        else:
            raise ValueError(
                f"Unknown head_style: {self.head_style!r}. "
                "Expected 'default' or 'residual_mlp'."
            )

        self.final_norm = nn.LayerNorm()  # Final norm for pre-norm architecture

    def _scale_y(self, y):
        """Standardise y: ``(y - y_mean) / y_std``."""
        return (y - self.std_stats["y_mean"]) / self.std_stats["y_std"]

    def _trunk(self, y_scaled, xi_scaled, mask=None):
        """Shared embedding + transformer trunk. Returns (z, e1, e2).

        z  : [*batch, T+1, model_dim]  — transformer output (global token first)
        e1 : [*batch, T, model_dim]    — fresh per-timestep embedding for head 1
        e2 : [*batch, T, model_dim]    — fresh per-timestep embedding for head 2
        """
        z = self.embed_proj(self.embed(y_scaled, xi_scaled))  # [*batch, T, model_dim]

        batch_shape = z.shape[:-2]
        global_token = jnp.broadcast_to(
            self.global_token, (*batch_shape, 1, self.model_dim)
        )
        z = jnp.concatenate([global_token, z], axis=-2)  # [*batch, T+1, model_dim]

        if self.positional:
            positional = jnp.concatenate(
                [
                    jnp.zeros((1, self.model_dim), dtype=z.dtype),
                    self.positional_encoding,
                ],
                axis=0,
            )
            z += positional

        # Extend mask to cover global token (always valid).
        if mask is not None:
            gt_mask = jnp.ones((*mask.shape[:-1], 1), dtype=bool)
            full_mask = jnp.concatenate([gt_mask, mask], axis=-1)  # [*batch, T+1]
        else:
            full_mask = None

        z = self.transformer_stack(z, mask=full_mask)
        z = self.final_norm(z)

        e1 = self.head1_embed_proj(self.head1_embed(y_scaled, xi_scaled))
        e2 = self.head2_embed_proj(self.head2_embed(y_scaled, xi_scaled))

        return z, e1, e2

    def encode(self, y, xi, mask=None, sigma=None):
        """Return the global-token posterior representation.

        y:    [*batch, T, d_y]  — positions ≥ t should be zero-padded
        xi:   [*batch, T, d_x]
        mask: [*batch, T] bool  — True = valid timestep (None means all valid)
        Returns: [*batch, model_dim]
        """
        y_scaled = self._scale_y(y)
        xi_scaled = (xi - self.std_stats["xi_mean"]) / self.std_stats["xi_std"]
        z, _, _ = self._trunk(y_scaled, xi_scaled, mask=mask)
        return z[..., 0, :]  # global token

    def __call__(
        self, y, xi, mask=None, aux_data: dict | None = None, sigma=None, **kwargs
    ):
        y_scaled = self._scale_y(y)
        xi_scaled = (xi - self.std_stats["xi_mean"]) / self.std_stats["xi_std"]

        z, e1, e2 = self._trunk(y_scaled, xi_scaled, mask=mask)

        g = z[..., :1, :]
        g_rep = jnp.broadcast_to(g, (*g.shape[:-2], self.T, g.shape[-1]))

        if self.positional:
            pos = jnp.broadcast_to(self.positional_encoding, e1.shape)

        if self.head_mode == "base":
            if self.positional:
                head1_in = jnp.concatenate([g_rep, e1, pos], axis=-1)
                head2_in = jnp.concatenate([g_rep, e2, pos], axis=-1)
            else:
                head1_in = jnp.concatenate([g_rep, e1], axis=-1)
                head2_in = jnp.concatenate([g_rep, e2], axis=-1)

        elif self.head_mode == "shared_e_t":
            if self.positional:
                head1_in = jnp.concatenate([g_rep, e1, z[..., 1:, :], pos], axis=-1)
                head2_in = jnp.concatenate([g_rep, e2, z[..., 1:, :], pos], axis=-1)
            else:
                head1_in = jnp.concatenate([g_rep, e1, z[..., 1:, :]], axis=-1)
                head2_in = jnp.concatenate([g_rep, e2, z[..., 1:, :]], axis=-1)

        elif self.head_mode == "shared_e_t_window":
            # Markov window: per-head input includes fresh embeddings of
            # (y_{t-1}, x_{t-1}) and (y_{t+1}, x_{t+1}) in addition to (y_t, x_t).
            # Motivated by Markovian likelihoods p(y_t | y_{t-1}, x_{t-1}, theta):
            # the conditional score at time t has explicit dependence on the
            # one-step Markov neighbours, so providing them directly to the
            # head saves the trunk from having to reconstruct them via attention.
            zero_pad = jnp.zeros((*e1.shape[:-2], 1, e1.shape[-1]), dtype=e1.dtype)
            e1_prev = jnp.concatenate([zero_pad, e1[..., :-1, :]], axis=-2)
            e1_next = jnp.concatenate([e1[..., 1:, :], zero_pad], axis=-2)
            e2_prev = jnp.concatenate([zero_pad, e2[..., :-1, :]], axis=-2)
            e2_next = jnp.concatenate([e2[..., 1:, :], zero_pad], axis=-2)
            if self.positional:
                head1_in = jnp.concatenate(
                    [g_rep, e1_prev, e1, e1_next, z[..., 1:, :], pos], axis=-1
                )
                head2_in = jnp.concatenate(
                    [g_rep, e2_prev, e2, e2_next, z[..., 1:, :], pos], axis=-1
                )
            else:
                head1_in = jnp.concatenate(
                    [g_rep, e1_prev, e1, e1_next, z[..., 1:, :]], axis=-1
                )
                head2_in = jnp.concatenate(
                    [g_rep, e2_prev, e2, e2_next, z[..., 1:, :]], axis=-1
                )

        out1 = self.head1(self.head1_input_norm(head1_in))
        out2 = self.head2(self.head2_input_norm(head2_in))
        out = jnp.concatenate([out1, out2], axis=-1)

        out = _apply_score_net_parameterisation(
            out,
            y,
            xi,
            aux_data,
            noise_sigma=self.noise_sigma,
            noise_condition_mode=self.noise_condition_mode,
            xi_output_mode=self.xi_output_mode,
            y_output_mode=self.y_output_mode,
            y_residual_mode=self.y_residual_mode,
        )

        return out


class ConservativeScoreNetwork(nn.Module):
    """Hydra-friendly wrapper that turns any base score net into a conservative score.

    The base net is used as a feature map, projected to a scalar potential, then
    differentiated w.r.t. (y, xi).

    ``y_residual_mode='y_t_diff'`` adds a Gaussian-Markov-chain potential to φ:

        φ_GM(y) = − ½ Σ_t (y_t − y_{t-1})² / σ_y²    (y_{-1} = aux_data['y_init'])

    Its gradient w.r.t. y_t recovers the leading-order Markov-likelihood score
    (the t-1 *and* t+1 couplings appear automatically via autograd of the
    summed-over-t potential), so the base network only has to learn the small
    O(dt)·drift correction. Requires aux_data['y_init'] at every call site.
    """

    base_score_network: dict[str, Any]
    output_shape: int
    x_dim: int
    y_dim: int
    T: int
    std_stats: dict
    noise_sigma: Array
    base_output_shape: int | tuple[int, int] | None = None
    y_output_mode: str = "none"
    xi_output_mode: str = "none"
    y_residual_mode: str = "default"
    noise_condition_mode: str = "none"
    # Hidden width of the scalar-potential head. The potential LayerNorm acts over
    # the base net's output axis (size d_y + d_x); for d_y + d_x == 2 (scalar y AND
    # scalar design) LayerNorm over 2 elements collapses to a
    # near-constant sign, killing the conservative score's input-gradient (flat
    # loss / ~0 param-grads). Set this to up-project the base output to a wider
    # vector before the LayerNorm so the potential gradient is well defined. None
    # keeps the original (no up-projection) behaviour, unchanged for d_y + d_x >= 3.
    potential_hidden_dim: int | None = None

    def setup(self):
        base_cfg = dict(self.base_score_network)
        # Keep the base network in raw-output mode; residual/conditioning is applied
        # at the conservative wrapper level through the potential definition below.
        base_cfg["y_output_mode"] = "none"
        base_cfg["xi_output_mode"] = "none"
        base_cfg["y_residual_mode"] = "default"
        base_cfg["noise_condition_mode"] = "none"
        self._base_score_net: nn.Module = instantiate(  # type: ignore
            base_cfg,
            output_shape=(
                self.base_output_shape
                if self.base_output_shape is not None
                else self.y_dim + self.x_dim
            ),
            x_dim=self.x_dim,
            y_dim=self.y_dim,
            T=self.T,
            std_stats=self.std_stats,
            noise_sigma=self.noise_sigma,
        )
        # Optional up-projection so the potential LayerNorm acts on a wide,
        # non-degenerate axis (needed when d_y + d_x is tiny, e.g. == 2).
        self.potential_in = (
            nn.Dense(self.potential_hidden_dim, kernel_init=gelu_init)
            if self.potential_hidden_dim is not None
            else None
        )
        self.layer_norm = nn.LayerNorm()
        self.output_projection = nn.Dense(1, kernel_init=gelu_init)

    def encode(self, y, xi, mask=None, sigma=None):
        """Delegate to the base network's posterior-representation encoder."""
        return self._base_score_net.encode(y, xi, mask=mask, sigma=sigma)

    @check_shapes(
        "y: [*batch_size, T, d_y]",
        "xi: [*batch_size, T, d_xi]",
        "return: [*batch_size, T, d_out]",
    )
    def __call__(
        self, y, xi, mask=None, aux_data: dict | None = None, sigma=None, **kwargs
    ):
        aux_data = {} if aux_data is None else aux_data

        use_y_t_diff = self.y_residual_mode == "y_t_diff"
        if use_y_t_diff:
            if "y_init" not in aux_data:
                raise ValueError(
                    "ConservativeScoreNetwork with y_residual_mode='y_t_diff' "
                    "requires aux_data['y_init']."
                )
            y_init = aux_data["y_init"]
        else:
            y_init = None

        def base_potential_fn(y_in, xi_in, mask_in, y_init_in, sigma_in):
            y_batched = jnp.expand_dims(y_in, axis=0)
            xi_batched = jnp.expand_dims(xi_in, axis=0)
            mask_batched = jnp.expand_dims(mask_in, axis=0)  # [1, T]
            vec_out = self._base_score_net(
                y_batched,
                xi_batched,
                mask=mask_batched,
                aux_data=aux_data,
                sigma=None,
                **kwargs,
            )
            if self.potential_in is not None:
                vec_out = nn.gelu(self.potential_in(vec_out))
            vec_out = self.layer_norm(vec_out)
            # Potential must be scalar per-example for jax.grad.
            f = jnp.sum(self.output_projection(vec_out))
            if self.y_output_mode == "plus_y":
                f += 0.5 * jnp.sum(y_in**2)
            elif self.y_output_mode == "minus_y":
                f -= 0.5 * jnp.sum(y_in**2)
            if self.xi_output_mode == "plus_x":
                f += 0.5 * jnp.sum(xi_in**2)
            elif self.xi_output_mode == "minus_x":
                f -= 0.5 * jnp.sum(xi_in**2)

            if use_y_t_diff:
                # φ_GM = -½ Σ_t (y_t - y_{t-1})² / σ_y², with y_{-1} = y_init.
                # ∇_{y_t} φ_GM = -(y_t - y_{t-1})/σ² + (y_{t+1} - y_t)/σ² at interior t,
                # and just -(y_{T-1} - y_{T-2})/σ² at t = T-1 (no t+1 term in the sum),
                # matching the leading-order Markov-likelihood score exactly.
                y_prev = jnp.concatenate([y_init_in[None, :], y_in[:-1, :]], axis=-2)
                y_res = y_in - y_prev  # [T, d_y]
                # Mask padded positions so the valid → invalid boundary doesn't
                # produce a spurious -y_{T*} residual (when y is zero-padded).
                valid = mask_in[:, None].astype(y_res.dtype)
                f -= 0.5 * jnp.sum(valid * y_res**2)
            return f

        score_full = _conservative_score_from_potential_fn(
            base_potential_fn,
            y,
            xi,
            mask=mask,
            y_init=y_init,
        )

        if not isinstance(self.noise_sigma, float):
            assert self.noise_sigma.shape == (y.shape[-1], y.shape[-1])

        if isinstance(self.noise_sigma, float):
            diag_noise_sigma = self.noise_sigma * jnp.ones((y.shape[-1],))
        else:
            diag_noise_sigma = jnp.diag(self.noise_sigma)

        if self.noise_condition_mode == "none":
            full_noise_conditioning = jnp.ones((y.shape[-1] + xi.shape[-1],))
        elif self.noise_condition_mode == "y_only":
            full_noise_conditioning = jnp.concatenate(
                [diag_noise_sigma, jnp.ones((xi.shape[-1],))], axis=-1
            )
        elif self.noise_condition_mode == "y_and_xi_mean":
            mean_noise_sigma = jnp.mean(diag_noise_sigma)
            xi_conditioning = mean_noise_sigma * jnp.ones((xi.shape[-1],))
            full_noise_conditioning = jnp.concatenate(
                [diag_noise_sigma, xi_conditioning], axis=-1
            )

        score_full = score_full / (full_noise_conditioning)

        full_dim = self.y_dim + self.x_dim
        if self.output_shape == full_dim:
            return score_full
        if self.output_shape == self.y_dim:
            return score_full[..., : self.y_dim]
        if self.output_shape == self.x_dim:
            return score_full[..., self.y_dim :]
        raise ValueError(
            "ConservativeScoreNetwork expects output_shape equal to y_dim, x_dim, "
            "or y_dim + x_dim"
        )


def _conservative_score_from_potential_fn(
    potential_fn: Callable,
    y: jnp.ndarray,
    xi: jnp.ndarray,
    mask: jnp.ndarray | None = None,
    y_init: jnp.ndarray | None = None,
    sigma: jnp.ndarray | None = None,
) -> jnp.ndarray:
    # jax.grad requires scalar outputs, so compute per-example grads via vmap.
    y_flat = y.reshape((-1, *y.shape[-2:]))
    xi_flat = xi.reshape((-1, *xi.shape[-2:]))
    if mask is None:
        mask_flat = jnp.ones((y_flat.shape[0], y.shape[-2]), dtype=bool)
    else:
        mask_flat = mask.reshape((-1, mask.shape[-1]))
    if y_init is None:
        y_init_flat = jnp.zeros((y_flat.shape[0], y.shape[-1]), dtype=y.dtype)
    else:
        y_init_flat = y_init.reshape((-1, y_init.shape[-1]))
    # Per-example noise multiplier (lambda) threaded through the vmap like y_init,
    # so each trajectory's potential sees its own sigma (not the whole batch).
    if sigma is None:
        sigma_flat = jnp.ones((y_flat.shape[0],), dtype=y.dtype)
    else:
        sigma_flat = jnp.broadcast_to(sigma, y.shape[:-2]).reshape((-1,))
    # argnums=(0, 1): grad only w.r.t. y and xi; mask / y_init / sigma flow through
    # as constants.
    grad_fn = jax.vmap(jax.grad(potential_fn, argnums=(0, 1)))
    grad_y_flat, grad_xi_flat = grad_fn(
        y_flat, xi_flat, mask_flat, y_init_flat, sigma_flat
    )
    grad_y = grad_y_flat.reshape(y.shape)
    grad_xi = grad_xi_flat.reshape(xi.shape)
    return jnp.concatenate([grad_y, grad_xi], axis=-1)
