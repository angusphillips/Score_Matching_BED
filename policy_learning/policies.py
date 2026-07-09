from collections.abc import Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array

from policy_learning.utils.activations import activations

##### Policy Networks ######

relu_init = nn.initializers.variance_scaling(
    scale=2.0, mode="fan_in", distribution="truncated_normal"
)

pytorch_default_init = nn.initializers.variance_scaling(
    scale=1.0, mode="fan_in", distribution="uniform"
)


class DenseTorchInit(nn.Module):
    features: int
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32
    zero_init: bool = False

    @nn.compact
    def __call__(self, x):
        fan_in = x.shape[-1]

        # PyTorch default: U(-1/sqrt(fan_in), 1/sqrt(fan_in))
        k = jnp.sqrt(1.0 / fan_in)

        if self.zero_init:

            def kernel_init(key, shape, dtype=self.dtype):
                return jax.random.uniform(
                    key, shape, minval=-5e-2, maxval=5e-2, dtype=dtype
                )

            def bias_init(key, shape, dtype=self.dtype):
                return jnp.zeros(shape, dtype)
        else:

            def kernel_init(key, shape, dtype=self.dtype):
                return jax.random.uniform(key, shape, minval=-k, maxval=k, dtype=dtype)

            def bias_init(key, shape, dtype=self.dtype):
                return kernel_init(key, shape, dtype)

        kernel = self.param(
            "kernel",
            kernel_init,
            (fan_in, self.features),
        )

        y = jnp.dot(x, kernel)

        if self.use_bias:
            bias = self.param(
                "bias",
                bias_init,
                (self.features,),
            )
            y = y + bias

        return y


class StaticDesignNet(nn.Module):
    shape: Sequence[
        int
    ]  # Usually I think this will be [T, d] where T is the experiment time span and d is the dimension of the design space.
    encoding_dim = 1
    output_activation: Callable = lambda x: x

    def setup(self):
        self.params = self.param(
            name="static_designs",
            init_fn=jax.random.normal,
            shape=self.shape,
        )  # pyright: ignore[reportCallIssue]

    def __call__(
        self,
        y: Array | None = None,
        x: Array | None = None,
        z: Array | None = None,
        **kwargs,
    ):
        return self.output_activation(self.params)


class DADNet(nn.Module):
    encoder_shapes: Sequence[int]
    encoding_dim: int
    enc_act: str
    emit_act: str
    emitter_shapes: Sequence[int]
    shape: Sequence[
        int
    ]  # Usually I think this will be [T, d] where T is the experiment time span and d is the dimension of the design space.
    concat: bool = False
    empty_value: float = 0.01
    output_activation: Callable = lambda x: x
    zero_init_final_layer: bool = False

    def setup(self):
        enc_layers = [
            a
            for i in self.encoder_shapes
            for a in (
                DenseTorchInit(features=i),
                activations[self.enc_act],
            )
        ]
        enc_layers.append(DenseTorchInit(features=self.encoding_dim))
        self.encoder_net = nn.Sequential(enc_layers)
        emit_layers = [
            a
            for i in self.emitter_shapes
            for a in (
                DenseTorchInit(features=i),
                activations[self.emit_act],
            )
        ]
        emit_layers.append(
            DenseTorchInit(
                features=self.shape[-1], zero_init=self.zero_init_final_layer
            )
        )
        self.emitter_net = nn.Sequential(emit_layers)

    def encoder(self, y: Array, x: Array, **kwargs):
        enc_in = jnp.concatenate([y, x], axis=-1)  # [*batch, d]
        return self.encoder_net(enc_in)

    def emitter(self, enc: Array, z: Array | None = None, **kwargs):
        return self.output_activation(self.emitter_net(enc))

    def __call__(self, y: Array, x: Array, z: Array | None = None, **kwargs):
        encoding = self.encoder(y, x, **kwargs)
        return self.emitter(encoding)


class SelfAttentionLayer(nn.Module):
    encoding_dim: int
    num_heads: int
    ff_dim: int

    @nn.compact
    def __call__(self, x: Array, mask: Array | None = None) -> Array:
        # x: [batch, T, encoding_dim]
        # mask (optional): broadcastable to [batch, num_heads, q_len, kv_len];
        # True = attend, False = ignore. Used for causal/history masking so the
        # attention runs over the time axis only, never across the batch.
        h = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.encoding_dim,
            out_features=self.encoding_dim,
            kernel_init=pytorch_default_init,
            use_bias=True,
            deterministic=True,
        )(x, mask=mask)

        # Transformer-style residual + layer norm
        h = nn.LayerNorm()(h + x)
        # Position-wise feed-forward network: expand to ``ff_dim`` then project
        # back to the model dimension (standard transformer block).
        ff = DenseTorchInit(features=self.ff_dim)(h)
        ff = nn.gelu(ff)
        ff = DenseTorchInit(features=self.encoding_dim)(ff)
        h = nn.LayerNorm()(ff + h)
        return h


class AttentionDesignNet(nn.Module):
    encoder_shapes: Sequence[int]
    encoding_dim: int
    num_heads: int
    enc_act: str
    emit_act: str
    emitter_shapes: Sequence[int]
    shape: Sequence[int]  # [T, d_xi]
    n_transformer_layers: int = 2
    transformer_ff_dim: int = 128
    concat: bool = True
    empty_value: float = 0.0
    output_activation: Callable = lambda x: x
    zero_init_final_layer: bool = True
    positional: bool = False

    def _mlp(self, shapes: Sequence[int], act: str, out_features: int) -> nn.Sequential:
        layers = [
            a
            for i in shapes
            for a in (DenseTorchInit(features=i), activations[act])
        ]
        layers.append(DenseTorchInit(features=out_features))
        return nn.Sequential(layers)

    def setup(self):
        # --------------------------------------------------
        # Separate design / outcome encoders, then a fusion module producing the
        # per-step pair representation r_t (TNP-style):
        #     e^xi_t = f_xi(xi_t),  e^y_t = f_y(y_t),
        #     r_t    = f_pair([e^xi_t ; e^y_t]).
        # --------------------------------------------------
        self.design_encoder_net = self._mlp(
            self.encoder_shapes, self.enc_act, self.encoding_dim
        )
        self.outcome_encoder_net = self._mlp(
            self.encoder_shapes, self.enc_act, self.encoding_dim
        )
        self.fusion_net = self._mlp(
            self.encoder_shapes, self.enc_act, self.encoding_dim
        )

        # --------------------------------------------------
        # Input adapter: linear projection of the pair representation r_t into the
        # transformer (history-aggregator) embedding dimension.
        # --------------------------------------------------
        self.input_adapter = DenseTorchInit(features=self.encoding_dim)

        # --------------------------------------------------
        # Learned type embedding marking history tokens. The decision token is a
        # separate learned vector (below), so it implicitly carries its own type.
        # --------------------------------------------------
        self.history_type_embedding = self.param(
            "history_type_embedding",
            nn.initializers.normal(stddev=0.02),
            (self.encoding_dim,),
        )

        # --------------------------------------------------
        # Learned time embeddings over buffer positions (length T). Buffer index s
        # always holds experiment step s-1 (index 0 = decision token) in both the
        # rollout scan and init, so absolute indexing by buffer position is
        # consistent everywhere. Only used when ``positional`` is enabled.
        # --------------------------------------------------
        if self.positional:
            self.time_embedding = self.param(
                "time_embedding",
                nn.initializers.normal(stddev=0.02),
                (self.shape[0], self.encoding_dim),
            )

        # --------------------------------------------------
        # Learned decision ("CLS"/query) token. A single shared vector placed at
        # buffer index 0 at every emission step; the design is read out from its
        # contextualised attention output rather than by pooling the history.
        # --------------------------------------------------
        self.decision_token = self.param(
            "decision_token",
            nn.initializers.normal(stddev=0.02),
            (self.encoding_dim,),
        )

        # --------------------------------------------------
        # Pre-transformer normalisation of the token sequence, then a stack of
        # ``n_transformer_layers`` self-attention encoder layers.
        # --------------------------------------------------
        self.input_norm = nn.LayerNorm()
        self.attn_layers = [
            SelfAttentionLayer(
                self.encoding_dim, self.num_heads, self.transformer_ff_dim
            )
            for _ in range(self.n_transformer_layers)
        ]

        # --------------------------------------------------
        # Emitter MLP
        # --------------------------------------------------
        emit_layers = [
            a
            for i in self.emitter_shapes
            for a in (
                DenseTorchInit(
                    features=i,
                ),
                # nn.LayerNorm(),
                activations[self.emit_act],
            )
        ]
        emit_layers.append(
            DenseTorchInit(
                features=self.shape[-1], zero_init=self.zero_init_final_layer
            )
        )
        self.emitter_net = nn.Sequential(emit_layers)

    # ------------------------------------------------------
    # Encoder: per-(xi, yi) pair representation only (no attention here).
    #
    # iDAD is rolled out through the buffered "concat" reparam sampler: the
    # sampler maintains a buffer of these per-step encodings and calls
    # ``emitter(enc=buffer, t=...)`` to emit one design at a time. The
    # self-attention over *history* therefore lives in the emitter, where the
    # full buffer (time axis) is available -- NOT here, where a per-step call
    # would otherwise make nn.SelfAttention attend over the batch axis.
    # ------------------------------------------------------
    def encoder(self, y: Array, x: Array, **kwargs) -> Array:
        """y: [*batch, dy], x: [*batch, dx] -> [*batch, encoding_dim] (r_t)."""
        e_xi = self.design_encoder_net(x)  # [*batch, encoding_dim]
        e_y = self.outcome_encoder_net(y)  # [*batch, encoding_dim]
        return self.fusion_net(jnp.concatenate([e_xi, e_y], axis=-1))

    # ------------------------------------------------------
    # Emitter: causal-masked self-attention over [decision token, history], then
    # read out the decision token's contextualised output.
    # ------------------------------------------------------
    def emitter(self, enc: Array, t: Array, z: Array | None = None, **kwargs) -> Array:
        """Emit the design at step ``t`` from the encoding buffer.

        enc: [*batch, T, encoding_dim] buffer of pair representations r_t; the
        sampler writes the encoding of step ``s`` at buffer index ``s + 1`` (index
        0 is reserved). We project the buffer into the transformer dimension, add a
        learned history-type embedding, then overwrite index 0 with the learned
        ``decision_token`` so the attended set when emitting design ``t`` is the
        decision token (index 0) plus the ``t`` history encodings (indices
        ``1..t``) -- i.e. ``t + 1`` active tokens. We self-attend over those
        (masking index > t and never across the batch) and feed the decision
        token's contextualised output to the emitter MLP. This is always defined
        (even at ``t = 0`` with no history), so it is NaN-safe. ``z`` is unused
        (iDAD is deterministic).
        """
        T = enc.shape[-2]
        # Input adapter: project pair representations into the transformer dim.
        h = self.input_adapter(enc)  # [*batch, T, encoding_dim]
        # Learned history-type embedding on every token (broadcasts over *batch)...
        h = h + self.history_type_embedding
        # ...then overwrite index 0 with the learned decision (query) token. This
        # replaces the history-type embedding at that slot, so the decision token
        # implicitly carries its own type information.
        h = h.at[..., 0, :].set(self.decision_token)
        # Add learned time embeddings (history order matters for trajectory
        # models). Buffer position p >= 1 holds experiment step p - 1, so it gets
        # ``time_embedding[p - 1]``; the decision token (index 0) gets the time
        # index of the next decision step, ``t``.
        if self.positional:
            te = self.time_embedding  # [T, encoding_dim]
            base = jnp.concatenate(
                [jnp.zeros((1, self.encoding_dim)), te[: T - 1]], axis=0
            )  # [T, encoding_dim]; position 0 filled in below
            base = jnp.broadcast_to(base, h.shape)
            base = base.at[..., 0, :].set(te[jnp.asarray(t).astype(jnp.int32)])
            h = h + base
        # Pre-transformer normalisation of the token sequence.
        h = self.input_norm(h)
        positions = jnp.arange(T)
        # [*batch, T] mask: decision token (index 0) .. t inclusive.
        mask = positions <= jnp.asarray(t)[..., None]
        # Key mask broadcast over (num_heads, query) -> [*batch, 1, 1, T].
        attn_mask = mask[..., None, None, :]
        for layer in self.attn_layers:
            h = layer(h, mask=attn_mask)  # [*batch, T, encoding_dim]
        context = h[..., 0, :]  # decision-token output, [*batch, encoding_dim]
        return self.output_activation(self.emitter_net(context))  # [*batch, d_xi]

    # ------------------------------------------------------
    # Forward (only used to initialise parameters).
    # ------------------------------------------------------
    def __call__(
        self,
        y: Array,
        x: Array,
        z: Array | None = None,
        t: Array | None = None,
        **kwargs,
    ) -> Array:
        enc = self.encoder(y, x)  # [*batch, T, encoding_dim] at init
        if t is None:
            t = jnp.zeros(enc.shape[:-2], dtype=jnp.int32)
        return self.emitter(enc, t=t, z=z)


class IqbalGRUNet(nn.Module):
    encoder_shapes: Sequence[int]
    encoding_dim: int
    rnn_hidden_dim: int
    rnn_layers: int
    enc_act: str
    emit_act: str
    emitter_shapes: Sequence[int]
    shape: Sequence[int]  # [T, d_xi]
    concat: bool = True
    output_activation: Callable = lambda x: x
    empty_value: float = 0.0
    min_std: float = 1e-4
    deterministic: bool = False

    def setup(self):
        enc_layers = [
            a
            for i in self.encoder_shapes
            for a in (
                DenseTorchInit(features=i),
                activations[self.enc_act],
            )
        ]
        enc_layers.append(DenseTorchInit(features=self.encoding_dim))
        self.encoder_net = nn.Sequential(enc_layers)

        self.gru_cells = [
            nn.GRUCell(self.rnn_hidden_dim) for _ in range(self.rnn_layers)
        ]

        emit_layers = [
            a
            for i in self.emitter_shapes
            for a in (
                DenseTorchInit(features=i),
                activations[self.emit_act],
            )
        ]
        emit_layers.append(DenseTorchInit(features=self.shape[-1]))
        self.emitter_net = nn.Sequential(emit_layers)

        self.log_std = self.param(
            "log_std",
            nn.initializers.zeros,
            (self.shape[-1],),
        )

    def init_gru_state(self, batch_shape: Sequence[int]) -> tuple[Array, ...]:
        return tuple(
            jnp.zeros((*batch_shape, self.rnn_hidden_dim))
            for _ in range(self.rnn_layers)
        )

    def gru_step(
        self, state: tuple[Array, ...], r_t: Array
    ) -> tuple[tuple[Array, ...], Array]:
        h = r_t
        new_state = []
        for cell, h_prev in zip(self.gru_cells, state, strict=True):
            new_h, h = cell(h_prev, h)
            new_state.append(new_h)
        return tuple(new_state), h

    def emitter_from_state(self, h: Array, z: Array | None = None) -> Array:
        mean = self.emitter_net(h)
        if z is None:
            z = jnp.zeros_like(mean)
        if self.deterministic:
            z = jnp.zeros_like(z)
        std = jnp.exp(self.log_std) + self.min_std
        return self.output_activation(mean + std * z)

    def encoder(self, y: Array, x: Array, **kwargs) -> Array:
        enc_in = jnp.concatenate([y, x], axis=-1)
        return self.encoder_net(enc_in)

    def emitter(self, enc: Array, t: Array, z: Array | None = None, **kwargs) -> Array:
        enc_inputs = enc[..., 1:, :]
        pad = jnp.zeros_like(enc[..., :1, :])
        enc_inputs = jnp.concatenate([enc_inputs, pad], axis=-2)
        enc_inputs = jnp.moveaxis(enc_inputs, -2, 0)
        batch_shape = enc_inputs.shape[1:-1]
        init_state = self.init_gru_state(batch_shape)
        if z is None:
            z = jnp.zeros((*enc.shape[:-2], enc.shape[-2], self.shape[-1]))
        if self.deterministic:
            z = jnp.zeros_like(z)
        z_inputs = jnp.moveaxis(z, -2, 0)

        def step(carry, inputs):
            state = carry
            r_t, z_t = inputs
            h_top = state[-1]
            x_t = self.emitter_from_state(h_top, z=z_t)
            new_state, _ = self.gru_step(state, r_t)
            return new_state, x_t

        _, x_seq = jax.lax.scan(step, init_state, (enc_inputs, z_inputs))
        x_seq = jnp.moveaxis(x_seq, 0, -2)
        time_len = x_seq.shape[-2]
        idx = jnp.clip(t, 0, time_len - 1).astype(jnp.int32)
        idx = idx[(None,) * (x_seq.ndim - 2) + (None,)]
        return jnp.take_along_axis(x_seq, idx, axis=-2).squeeze(axis=-2)

    def __call__(self, y: Array, x: Array, z: Array | None = None, **kwargs) -> Array:
        enc = self.encoder(y, x, **kwargs)
        if self.is_mutable_collection("params"):
            batch_shape = enc.shape[:-2]
            _ = self.emitter_net(jnp.zeros((*batch_shape, self.rnn_hidden_dim)))
            _ = self.gru_step(
                self.init_gru_state(batch_shape),
                jnp.zeros((*batch_shape, self.encoding_dim)),
            )
            return jnp.zeros((*batch_shape, enc.shape[-2], self.shape[-1]))
        enc_inputs = enc
        enc_inputs = jnp.moveaxis(enc_inputs, -2, 0)
        batch_shape = enc_inputs.shape[1:-1]
        state = self.init_gru_state(batch_shape)
        if z is None:
            z = jnp.zeros((*enc.shape[:-2], enc.shape[-2], self.shape[-1]))
        if self.deterministic:
            z = jnp.zeros_like(z)
        z_inputs = jnp.moveaxis(z, -2, 0)

        def step(carry, inputs):
            state = carry
            r_t, z_t = inputs
            h_top = state[-1]
            x_t = self.emitter_from_state(h_top, z=z_t)
            new_state, _ = self.gru_step(state, r_t)
            return new_state, x_t

        _, x_seq = jax.lax.scan(step, state, (enc_inputs, z_inputs))
        return jnp.moveaxis(x_seq, 0, -2)
