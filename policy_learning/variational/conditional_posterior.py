"""Conditional posterior approximation for policy_learning.

A transformer-based conditional MoG posterior that predicts mixture of gaussians
parameters conditioned on the observed rollout history (y_1:T, x_1:T).
"""

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import PRNGKeyArray as Key

from policy_learning.score_nets import (
    JointEmbedMLP,
    TransformerBlock,
    gelu_init,
)


def _order_sources_by_first_dim(theta_kd: Array) -> Array:
    """Sort source axis by first coordinate; theta_kd has shape [..., k, d]."""
    source_order = jnp.argsort(theta_kd[..., :, 0], axis=-1)
    gather_index = source_order[..., :, None]
    return jnp.take_along_axis(theta_kd, gather_index, axis=-2)


class MoGComponentHead(nn.Module):
    """Single mixture-component head producing mean, log-scale, and logit."""

    dim_theta: int

    def setup(self):
        self.mean_head = nn.Dense(self.dim_theta, kernel_init=gelu_init)
        self.log_scale_head = nn.Dense(self.dim_theta, kernel_init=gelu_init)
        self.logit_head = nn.Dense(1, kernel_init=gelu_init)

    def __call__(self, context: Array) -> tuple[Array, Array, Array]:
        mean_flat = self.mean_head(context)
        log_scale_flat = self.log_scale_head(context)
        logit = jnp.squeeze(self.logit_head(context), axis=-1)
        return mean_flat, log_scale_flat, logit


class MultiHeadAttentionPooling(nn.Module):
    """Learned multi-head attention pooling from sequence tokens to one context token."""

    model_dim: int
    num_heads: int

    def setup(self):
        self.query_token = self.param(
            "query_token",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.model_dim),
        )
        self.pool_attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)

    def __call__(self, z: Array) -> Array:
        # z has shape [..., T, model_dim]. Build one learned query per batch element.
        batch_shape = z.shape[:-2]
        query = jnp.broadcast_to(self.query_token, (*batch_shape, 1, self.model_dim))
        pooled = self.pool_attn(query, z)  # [..., 1, model_dim]
        return pooled[..., 0, :]


class TransformerConditionalMoGPosterior(nn.Module):
    """Conditional mixture of Gaussians posterior q(theta | y, xi).

    Uses a transformer backbone to embed the rollout history (y_1:T, x_1:T),
    then predicts MoG parameters via a prediction head.

    Args:
        dim_theta: Dimension of theta (posterior event dimension)
        num_components: Number of mixture components
        embedding_dim: Dimension of RFF embeddings
        embed_mlp_dim: MLP dimension for initial embedding
        model_dim: Transformer model dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        x_dim: Dimension of x (design)
        y_dim: Dimension of y (observations)
        T: Time horizon
        std_stats: Dictionary with y_mean, y_std, xi_mean, xi_std
        rff_sigma: RFF initialization std
        positional: Whether to use positional encoding
    """

    dim_theta: int
    theta_shape: Sequence[int]
    num_components: int
    embedding_dim: int
    embed_mlp_dim: int
    model_dim: int
    num_layers: int
    num_heads: int
    x_dim: int
    y_dim: int
    T: int
    std_stats: dict
    rff_sigma: float = 1.0
    positional: bool = False
    canonicalize_sources: bool = False

    def _maybe_canonicalize(self, theta: Array) -> Array:
        """Sort exchangeable sources on coordinate 0 (location-finding only).

        For models whose theta entries are *not* exchangeable (e.g. the
        dynamical-system parameter vectors) this must be a no-op, otherwise it
        scrambles the parameter axis. Gated by `canonicalize_sources`.
        """
        if self.canonicalize_sources and theta.ndim >= 3:
            return _order_sources_by_first_dim(theta)
        return theta

    def setup(self):
        # Input embedding
        self.embed = JointEmbedMLP(
            embed_dim=self.embedding_dim,
            mlp_dim=self.embed_mlp_dim,
            x_dim=self.x_dim,
            y_dim=self.y_dim,
            rff_sigma=self.rff_sigma,
        )
        self.embed_proj = nn.Dense(self.model_dim, kernel_init=gelu_init)

        # self.global_token = self.param(
        #     "global_token",
        #     nn.initializers.normal(stddev=0.02),
        #     (1, 1, self.model_dim),
        # )

        # Positional encoding if requested
        if self.positional:
            self.positional_encoding = self._get_positional_encoding()

        # Transformer blocks
        self.transformer_blocks = nn.Sequential(
            [
                TransformerBlock(self.model_dim, self.num_heads)
                for _ in range(self.num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm()
        self.attn_pool = MultiHeadAttentionPooling(
            model_dim=self.model_dim,
            num_heads=self.num_heads,
        )

        # Use K independent heads, one for each mixture component.
        self.component_heads = [
            MoGComponentHead(self.dim_theta) for _ in range(self.num_components)
        ]

    def _get_positional_encoding(self) -> Array:
        """Create sinusoidal positional encoding."""
        position = jnp.arange(self.T)[:, None]
        num_pairs = (self.model_dim + 1) // 2
        div_term = jnp.exp(
            jnp.arange(0, num_pairs) * -(jnp.log(10000.0) / self.model_dim)
        )
        pe = jnp.zeros((self.T, self.model_dim))
        pe = pe.at[:, 0::2].set(
            jnp.sin(position * div_term)[:, : self.model_dim // 2 + self.model_dim % 2]
        )
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term)[:, : self.model_dim // 2])
        return pe

    def __call__(self, y: Array, xi: Array) -> tuple[Array, Array, Array]:
        """Encode rollout history and predict MoG parameters.

        Args:
            y: Observations [batch, T, d_y]
            xi: Designs [batch, T, d_xi]

        Returns:
            means: [batch, num_components, dim_theta]
            log_scales: [batch, num_components, dim_theta]
            logits: [batch, num_components]
        """
        # Standardize inputs
        y_scaled = (y - self.std_stats["y_mean"]) / self.std_stats["y_std"]
        xi_scaled = (xi - self.std_stats["xi_mean"]) / self.std_stats["xi_std"]

        # Embed
        z = self.embed(y_scaled, xi_scaled)  # [batch, T, embedding_dim]
        z = self.embed_proj(z)  # [batch, T, model_dim]

        # batch_shape = z.shape[:-2]
        # global_token = jnp.broadcast_to(
        #     self.global_token,
        #     (*batch_shape, 1, self.model_dim),
        # )
        # z = jnp.concatenate(
        #     [global_token, z], axis=-2
        # )  # [*batch_size, T + 1, model_dim]

        # Add positional encoding
        if self.positional:
            # Positional encoding shape: [T, model_dim]
            # z shape: [batch, T, model_dim]
            # Broadcast to match z
            z = z + self.positional_encoding[None, :, :]

        # Transformer
        z = self.transformer_blocks(z)  # [batch, T, model_dim]
        z = self.final_norm(z)

        # Attention pool over time to get context.
        context = self.attn_pool(z)  # [batch, model_dim]

        component_outputs = [head(context) for head in self.component_heads]
        means_flat = jnp.stack([out[0] for out in component_outputs], axis=-2)
        log_scales_flat = jnp.stack([out[1] for out in component_outputs], axis=-2)
        logits = jnp.stack([out[2] for out in component_outputs], axis=-1)

        batch_shape = context.shape[:-1]
        means = means_flat.reshape(*batch_shape, self.num_components, *self.theta_shape)
        log_scales = log_scales_flat.reshape(
            *batch_shape, self.num_components, *self.theta_shape
        )

        return means, log_scales, logits

    def sample(self, key: Key, y: Array, xi: Array) -> Array:
        """Sample from posterior q(theta | y, xi).

        Args:
            key: Random key
            y: Observations [batch, T, d_y]
            xi: Designs [batch, T, d_xi]

        Returns:
            theta: [batch, dim_theta]
        """
        means, log_scales, logits = self(
            y, xi
        )  # shapes: [batch, K, k, d], [batch, K, k, d], [batch, K, k, d]

        key_component, key_sample = jax.random.split(key)

        # Sample component index from mixture
        batch_shape = logits.shape[:-1]
        component_indices = jax.random.categorical(
            key_component, logits, shape=batch_shape
        )  # [batch]

        # Sample from standard normal
        noise = jax.random.normal(key_sample, shape=(*batch_shape, *self.theta_shape))

        # Get scales
        scales = jnp.exp(log_scales)  # [batch, K, d]

        # Select means/scales for the chosen component. The component (K) axis
        # sits just before *theta_shape; broadcast component_indices with one
        # singleton per (K + theta_shape) axis so this works for any theta_shape
        k_axis = -(len(self.theta_shape) + 1)
        idx = component_indices
        for _ in range(len(self.theta_shape) + 1):
            idx = idx[..., None]
        selected_means = jnp.take_along_axis(means, idx, axis=k_axis).squeeze(k_axis)
        selected_scales = jnp.take_along_axis(scales, idx, axis=k_axis).squeeze(k_axis)

        # Reparameterized sample
        theta = selected_means + selected_scales * noise
        return self._maybe_canonicalize(theta)

    def log_prob(self, theta: Array, y: Array, xi: Array) -> Array:
        """Compute log probability log q(theta | y, xi).

        Args:
            theta: Samples [batch, T, theta_shape]
            y: Observations [batch, T, d_y]
            xi: Designs [batch, T, d_xi]

        Returns:
            log_prob: [batch]
        """
        means, log_scales, logits = self(
            y, xi
        )  # [batch, K, *theta_shape], [batch, K, *theta_shape], [batch, K]

        scales = jnp.exp(log_scales)  # [batch, K, *theta_shape]

        # theta arrives as [*batch, T, *theta_shape] and is constant over T;
        # take the t=0 slice along the T axis generically for any theta_shape.
        theta = self._maybe_canonicalize(theta)
        t_axis = -(len(self.theta_shape) + 1)
        theta_single_t = jnp.take(theta, 0, axis=t_axis)  # [*batch, *theta_shape]
        # Insert the mixture-component axis so it broadcasts against means.
        theta_expanded = jnp.expand_dims(theta_single_t, axis=t_axis)

        # Compute log prob for each component
        log_probs_per_dim = (
            -0.5 * jnp.log(2 * jnp.pi)
            - log_scales
            - 0.5 * ((theta_expanded - means) / scales) ** 2
        )  # [batch, K, *theta_shape]

        # Sum over the theta event dimensions only (the last len(theta_shape)
        # axes), leaving the [batch, K] component axis intact.
        theta_axes = tuple(range(-len(self.theta_shape), 0))
        log_probs_components = log_probs_per_dim.sum(axis=theta_axes)  # [batch, K]

        # Add log weights
        log_weights = jax.nn.log_softmax(logits)  # [batch, K]
        log_probs_weighted = log_probs_components + log_weights

        # Logsumexp over components
        log_prob = jax.scipy.special.logsumexp(log_probs_weighted, axis=-1)  # [batch]

        return log_prob

    def score(self, theta: Array, y: Array, xi: Array) -> Array:
        """Compute score function ∇_theta log q(theta | y, xi).

        Args:
            theta: Samples [batch, dim_theta]
            y: Observations [batch, T, d_y]
            xi: Designs [batch, T, d_xi]

        Returns:
            score: [batch, dim_theta]
        """

        def log_prob_single(theta_single):
            return self.log_prob(theta_single[None], y, xi)[0]

        grad_fn = jax.grad(log_prob_single)
        score = jax.vmap(grad_fn)(theta)

        return score
