"""Variational distribution families for approximating joint marginal p(y, xi).

All distributions are built on flowjax/equinox for a unified training interface.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import AbstractDistribution, Normal
from flowjax.flows import masked_autoregressive_flow
from jaxtyping import Array
from jaxtyping import PRNGKeyArray as Key


class VariationalDistribution(eqx.Module):
    """Wrapper around flowjax distributions for (y, xi) joint marginal.

    Provides a consistent interface for sampling, log_prob, and score computation,
    handling the reshaping between flat vectors and (y, xi) structured arrays.
    """

    dist: AbstractDistribution
    dim_y: int = eqx.field(static=True)
    dim_xi: int = eqx.field(static=True)
    T: int = eqx.field(static=True)

    @property
    def total_dim(self) -> int:
        return self.T * (self.dim_y + self.dim_xi)

    def _split_to_yx(self, z: Array) -> tuple[Array, Array]:
        """Split concatenated z into y and xi components."""
        batch_shape = z.shape[:-1]
        z_flat = z.reshape(*batch_shape, self.T, self.dim_y + self.dim_xi)
        y = z_flat[..., : self.dim_y]
        xi = z_flat[..., self.dim_y :]
        return y, xi

    def _concat_yx(self, y: Array, xi: Array) -> Array:
        """Concatenate y and xi into single vector."""
        batch_shape = y.shape[:-2]
        z = jnp.concatenate([y, xi], axis=-1)  # (*batch, T, dim_y + dim_xi)
        return z.reshape(*batch_shape, -1)  # (*batch, T * (dim_y + dim_xi))

    def sample(self, key: Key, batch_shape: tuple[int, ...]) -> tuple[Array, Array]:
        """Sample from the variational distribution.

        Returns:
            y: Array of shape (*batch_shape, T, dim_y)
            xi: Array of shape (*batch_shape, T, dim_xi)
        """
        z = self.dist.sample(key, batch_shape)
        return self._split_to_yx(z)

    def log_prob(self, y: Array, xi: Array) -> Array:
        """Compute log probability log q(y, xi).

        Args:
            y: Data samples of shape (*batch, T, dim_y)
            xi: Design samples of shape (*batch, T, dim_xi)

        Returns:
            Log probabilities of shape (*batch,)
        """
        z = self._concat_yx(y, xi)
        return self.dist.log_prob(z)

    def score(self, y: Array, xi: Array) -> tuple[Array, Array]:
        """Compute score function ∇ log q(y, xi).

        Returns:
            score_y: Gradient w.r.t. y of shape (*batch, T, dim_y)
            score_xi: Gradient w.r.t. xi of shape (*batch, T, dim_xi)
        """

        def log_prob_single(y_single, xi_single):
            return self.log_prob(y_single[None], xi_single[None])[0]

        # Vectorize over batch dimensions
        grad_fn = jax.vmap(jax.grad(log_prob_single, argnums=(0, 1)))
        score_y, score_xi = grad_fn(y, xi)
        return score_y, score_xi


def create_mean_field_gaussian(
    dim_y: int,
    dim_xi: int,
    T: int,
) -> VariationalDistribution:
    """Create a mean-field (diagonal) Gaussian variational distribution.

    q(y, xi) = N([y, xi]; 0, I)

    Note: The parameters (mean, std) will be learned during training via
    the equinox parameter updates.

    Args:
        dim_y: Dimension of y per time step
        dim_xi: Dimension of xi per time step
        T: Number of time steps

    Returns:
        VariationalDistribution wrapping a diagonal Gaussian
    """
    total_dim = T * (dim_y + dim_xi)

    # Create diagonal Gaussian with learnable loc and scale
    # flowjax Normal takes loc and scale arrays
    dist = Normal(loc=jnp.zeros(total_dim), scale=jnp.ones(total_dim))

    return VariationalDistribution(
        dist=dist,
        dim_y=dim_y,
        dim_xi=dim_xi,
        T=T,
    )


class MixtureOfGaussians(eqx.Module):
    """Mixture of diagonal Gaussians distribution.

    q(z) = sum_k w_k * N(z; mu_k, diag(sigma_k^2))

    Properly samples from one component at a time, weighted by mixture weights.
    """

    locs: Array  # (num_components, dim)
    log_scales: Array  # (num_components, dim)
    logits: Array  # (num_components,) - unnormalized log weights

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.locs.shape[-1],)

    @property
    def num_components(self) -> int:
        return self.locs.shape[0]

    @property
    def weights(self) -> Array:
        """Normalized mixture weights."""
        return jax.nn.softmax(self.logits)

    @property
    def scales(self) -> Array:
        """Component scales (standard deviations)."""
        return jnp.exp(self.log_scales)

    def sample(self, key: Key, sample_shape: tuple[int, ...] = ()) -> Array:
        """Sample from the mixture distribution."""
        key1, key2 = jax.random.split(key)

        # Sample component indices according to weights
        component_indices = jax.random.categorical(
            key1, self.logits, shape=sample_shape
        )

        # Sample from all components, then select
        # Shape: (*sample_shape, num_components, dim)
        eps = jax.random.normal(key2, shape=(*sample_shape, *self.locs.shape))
        all_samples = self.locs + eps * self.scales

        # Select the sample from the chosen component
        # component_indices: (*sample_shape,)
        # all_samples: (*sample_shape, num_components, dim)
        samples = jnp.take_along_axis(
            all_samples,
            component_indices[..., None, None],
            axis=-2,
        ).squeeze(-2)

        return samples

    def log_prob(self, x: Array) -> Array:
        """Compute log probability using log-sum-exp over components."""
        # x: (*batch, dim)
        # Expand for broadcasting: (*batch, 1, dim)
        x_expanded = x[..., None, :]

        # Compute log prob for each component
        # locs, scales: (num_components, dim)
        log_probs_per_dim = (
            -0.5 * jnp.log(2 * jnp.pi)
            - self.log_scales
            - 0.5 * ((x_expanded - self.locs) / self.scales) ** 2
        )
        # Sum over dimensions: (*batch, num_components)
        log_probs_components = log_probs_per_dim.sum(axis=-1)

        # Add log weights and logsumexp over components
        log_weights = jax.nn.log_softmax(self.logits)
        # (*batch, num_components) + (num_components,) -> (*batch, num_components)
        log_probs_weighted = log_probs_components + log_weights

        # Marginalize over components: (*batch,)
        return jax.scipy.special.logsumexp(log_probs_weighted, axis=-1)


def create_mixture_of_gaussians(
    key: Key,
    dim_y: int,
    dim_xi: int,
    T: int,
    num_components: int = 10,
) -> VariationalDistribution:
    """Create a mixture of Gaussians variational distribution.

    q(y, xi) = sum_k w_k * N([y, xi]; mu_k, diag(sigma_k^2))

    Args:
        key: Random key for initialization
        dim_y: Dimension of y per time step
        dim_xi: Dimension of xi per time step
        T: Number of time steps
        num_components: Number of mixture components

    Returns:
        VariationalDistribution wrapping a mixture of Gaussians
    """
    total_dim = T * (dim_y + dim_xi)

    # Initialize with different means for diversity
    locs = 0.5 * jax.random.normal(key, shape=(num_components, total_dim))
    log_scales = jnp.zeros((num_components, total_dim))

    # Uniform mixture weights to start (as logits)
    logits = jnp.zeros(num_components)

    dist = MixtureOfGaussians(locs=locs, log_scales=log_scales, logits=logits)

    return VariationalDistribution(
        dist=dist,  # type: ignore
        dim_y=dim_y,
        dim_xi=dim_xi,
        T=T,
    )


def create_maf(
    key: Key,
    dim_y: int,
    dim_xi: int,
    T: int,
    flow_layers: int = 10,
    nn_width: int = 64,
    nn_depth: int = 3,
    num_bins: int = 8,
) -> VariationalDistribution:
    """Create a Masked Autoregressive Flow variational distribution.

    Uses rational quadratic spline transformations for flexible density estimation.

    Args:
        key: Random key for initialization
        dim_y: Dimension of y per time step
        dim_xi: Dimension of xi per time step
        T: Number of time steps
        flow_layers: Number of MAF layers
        nn_width: Width of conditioner networks
        nn_depth: Depth of conditioner networks
        num_bins: Number of spline bins

    Returns:
        VariationalDistribution wrapping a MAF
    """
    total_dim = T * (dim_y + dim_xi)

    # Create base distribution
    base_dist = Normal(jnp.zeros(total_dim), jnp.ones(total_dim))

    # Create masked autoregressive flow with rational quadratic spline
    flow = masked_autoregressive_flow(
        key=key,
        base_dist=base_dist,
        transformer=RationalQuadraticSpline(knots=num_bins, interval=4),
        nn_depth=nn_depth,
        nn_width=nn_width,
        flow_layers=flow_layers,
    )

    return VariationalDistribution(
        dist=flow,
        dim_y=dim_y,
        dim_xi=dim_xi,
        T=T,
    )


def create_flow_model(
    model_type: str,
    key: Key | None = None,
    dim_y: int = 1,
    dim_xi: int = 1,
    T: int = 1,
    # Mixture of Gaussians parameters
    num_components: int = 10,
    # MAF parameters
    flow_layers: int = 10,
    nn_width: int = 64,
    nn_depth: int = 3,
    num_bins: int = 8,
) -> VariationalDistribution:
    """Factory function to create any variational flow model.

    Args:
        model_type: Type of model to create. One of:
            - "mean_field_gaussian": Diagonal Gaussian
            - "mixture_of_gaussians": Mixture of diagonal Gaussians
            - "maf": Masked Autoregressive Flow
        key: Random key for initialization (required for mixture_of_gaussians and maf)
        dim_y: Dimension of y per time step
        dim_xi: Dimension of xi per time step
        T: Number of time steps
        num_components: Number of components (for mixture_of_gaussians)
        flow_layers: Number of flow layers (for maf)
        nn_width: Width of conditioner networks (for maf)
        nn_depth: Depth of conditioner networks (for maf)
        num_bins: Number of spline bins (for maf)

    Returns:
        VariationalDistribution of the specified type

    Examples:
        >>> key = jax.random.PRNGKey(0)
        >>> # Create mean field Gaussian (no key needed)
        >>> dist = create_flow_model("mean_field_gaussian", dim_y=2, dim_xi=3, T=10)
        >>> # Create mixture of Gaussians
        >>> dist = create_flow_model("mixture_of_gaussians", key, dim_y=2, dim_xi=3, T=10, num_components=5)
        >>> # Create MAF
        >>> dist = create_flow_model("maf", key, dim_y=2, dim_xi=3, T=10, flow_layers=8)
    """
    if model_type == "mean_field_gaussian":
        return create_mean_field_gaussian(dim_y=dim_y, dim_xi=dim_xi, T=T)

    elif model_type == "mixture_of_gaussians":
        if key is None:
            raise ValueError("key is required for mixture_of_gaussians model")
        return create_mixture_of_gaussians(
            key=key,
            dim_y=dim_y,
            dim_xi=dim_xi,
            T=T,
            num_components=num_components,
        )

    elif model_type == "maf":
        if key is None:
            raise ValueError("key is required for maf model")
        return create_maf(
            key=key,
            dim_y=dim_y,
            dim_xi=dim_xi,
            T=T,
            flow_layers=flow_layers,
            nn_width=nn_width,
            nn_depth=nn_depth,
            num_bins=num_bins,
        )

    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Must be one of: 'mean_field_gaussian', 'mixture_of_gaussians', 'maf'"
        )
