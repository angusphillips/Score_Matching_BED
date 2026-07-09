import math
from collections.abc import Callable, Sequence
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
from check_shapes import check_shapes
from jaxtyping import Array, PyTree
from jaxtyping import PRNGKeyArray as Key

from policy_learning.bed_models.base import (
    BEDModel,
    fast_sampler,
    get_reparam_sampler,
)

Shape = Sequence[int]

# Probability-integral-transform helpers (standard-normal CDF and its inverse).
_Phi = jax.scipy.stats.norm.cdf
_Phi_inv = jax.scipy.special.ndtri
_PPF_EPS = 1e-6  # keep ndtri arguments strictly inside (0, 1)

# Default per-type mixture weights for random_sampler. theta is forwarded to
# every candidate (theta-independent ones ignore it), so the theta-dependent
# "adaptive" sampler can be mixed in. It is opt-in: weight 0 by default and not
# in the default candidate set, so it only participates when listed explicitly
# in candidate_samplers with a positive weight (and theta is available).
SAMPLER_WEIGHTS = {
    "isotropic_gaussian": 1.0,
    "random_scale_gaussian": 1.0,
    "adaptive": 0.0,
}


class Gravimetry(BEDModel):
    r"""Microgravity point-source (buried sphere / void) detection.

    A buried spherical anomaly produces a surface gravity reading (in microGal)

    .. math::
        g(\xi; \theta) = \kappa\, \frac{z}{\big(\|\xi - x_{\!\perp}\|^2 + z^2\big)^{3/2}},

    where ``z`` is the source depth, ``x_perp`` the horizontal source location and
    ``kappa`` the source strength (microGal*m^2, negative for a void). The physical
    mapping is ``kappa = G * (4/3) pi R^3 * delta_rho / 1e-8`` (i.e. ``G * Delta M``
    rescaled m/s^2 -> microGal).

    The model is iid with explicit additive Gaussian noise and a non-linear mean:

    .. math:: y_i \mid \theta, \xi_i \sim \mathcal N(g(\xi_i; \theta), \sigma^2).

    Two search-space dimensionalities (``search_dim``, = the design dimension):

    * ``search_dim=1`` (default): ``theta = (x0, z, kappa)``, design ``xi = x_i in R``.
    * ``search_dim=2``: ``theta = (x0, y0, z, kappa)``, design ``xi = (x_i, y_i) in R^2``.

    Priors (pushed forward from a standard-normal latent via ``theta_transform``):

    * ``x0`` (and ``y0`` when ``search_dim=2``): ``N(0, tau^2)``.
    * ``z``: ``log z ~ N(z_log_mean, z_log_std^2)`` with a floor ``z >= z_floor``.
    * ``kappa`` (``kappa_prior``):
        - ``"generative"``: ``R ~ U(R_min, R_max)`` and ``kappa = c R^3`` with
          ``c = G (4/3) pi delta_rho / 1e-8`` (so ``kappa < 0`` for ``delta_rho < 0``);
        - ``"direct"``: ``kappa ~ N(kappa_mean, kappa_std^2)`` truncated to ``kappa < 0``.

    Only ``kappa`` (i.e. ``R^3 delta_rho``) is identifiable -- ``delta_rho`` is fixed,
    never inferred jointly with ``R``.
    """

    def __init__(
        self,
        T: int = 20,
        search_dim: int = 1,  # design / search-space dimension (1 or 2)
        init_seed: int = 0,
        d_y: int = 1,
        sigma: float = 1.0,
        delta_rho: float = -2200.0,  # air-filled void; use -1500 for water-filled
        kappa_prior: str = "generative",  # "generative" | "direct"
        tau: float = 30.0,  # horizontal-location prior std (m)
        z_log_mean: float = math.log(10.0),
        z_log_std: float = 0.5,
        z_floor: float = 3.0,
        R_min: float = 1.5,
        R_max: float = 4.0,
        kappa_mean: float = -1200.0,
        kappa_std: float = 800.0,
        design_sigma: float = 40.0,  # design-prior / sampler spatial scale (m)
        G: float = 6.674e-11,
        dtype: jnp.dtype = jnp.float32,
        standardise: bool = True,
        constant_additive_noise: bool = True,
        **kwargs,
    ):
        if kappa_prior not in ("generative", "direct"):
            raise ValueError(
                f"kappa_prior must be 'generative' or 'direct', got {kappa_prior!r}"
            )
        if search_dim not in (1, 2):
            raise ValueError(f"search_dim must be 1 or 2, got {search_dim!r}")
        self.key = jax.random.key(init_seed)
        self.init_seed = init_seed
        self.T = T
        self.search_dim = search_dim
        self.d = search_dim  # design = horizontal search-space dimension
        self.k = 1  # single source (interface parity)
        self.d_y = d_y
        # theta = (horizontal coords..., z, kappa) -> search_dim + 2 components.
        self.theta_dim = search_dim + 2
        self.theta_shape = (self.theta_dim,)
        self.sigma = sigma
        self.delta_rho = delta_rho
        self.kappa_prior = kappa_prior
        self.tau = tau
        self.z_log_mean = z_log_mean
        self.z_log_std = z_log_std
        self.z_floor = z_floor
        self.R_min = R_min
        self.R_max = R_max
        self.kappa_mean = kappa_mean
        self.kappa_std = kappa_std
        self.design_sigma = design_sigma
        self.G = G
        self.dtype = dtype

        # kappa = c * R^3 (generative). c < 0 for delta_rho < 0.
        self.kappa_coeff = G * (4.0 / 3.0) * math.pi * delta_rho / 1e-8

        dim = T * (self.d + self.d_y)
        super().__init__(dim)
        self.constant_additive_noise = constant_additive_noise

        self.design_scale = design_sigma
        # Helpful so the policy network can output O(1) while actual designs need to be at a much larger scale in metres
        self.design_bijector = lambda x: self.design_scale * x  # network -> metres
        self.design_bijector_inv = lambda x: x / self.design_scale  # metres -> network
        self._policy_parameterisation = "no_param"

        self.theta_transform = self._theta_transform
        self.theta_transform_inv = self._theta_transform_inv

        self._fast_design_sampler_cache: dict[
            tuple[str | None, tuple[tuple[str, str], ...]], Callable
        ] = {}

        if standardise:
            self.compute_standardisation(10000)

    # ------------------------------------------------------------------
    # Latent <-> theta transforms (standard-normal latent of dim theta_dim)
    # ------------------------------------------------------------------
    def _theta_transform(self, u: Array) -> Array:
        """Map a standard-normal latent ``u`` to physical ``theta``.

        Layout: ``u[..., :search_dim]`` -> horizontal coords, ``u[..., search_dim]``
        -> depth, ``u[..., search_dim+1]`` -> kappa.
        """
        sd = self.search_dim
        horiz = self.tau * u[..., :sd]  # [..., sd] horizontal locations
        u_z = u[..., sd]
        u_k = u[..., sd + 1]

        z = jnp.exp(self.z_log_mean + self.z_log_std * u_z)
        z = jnp.maximum(z, self.z_floor)

        if self.kappa_prior == "generative":
            R = self.R_min + (self.R_max - self.R_min) * _Phi(u_k)
            kappa = self.kappa_coeff * R**3
        else:  # direct: N(kappa_mean, kappa_std^2) truncated to kappa < 0
            phi_b = _Phi((0.0 - self.kappa_mean) / self.kappa_std)
            p = jnp.clip(_Phi(u_k) * phi_b, _PPF_EPS, 1.0 - _PPF_EPS)
            kappa = self.kappa_mean + self.kappa_std * _Phi_inv(p)

        return jnp.concatenate([horiz, z[..., None], kappa[..., None]], axis=-1)

    def _theta_transform_inv(self, theta: Array) -> Array:
        """Map physical ``theta`` back to the standard-normal latent ``u``."""
        sd = self.search_dim
        u_h = theta[..., :sd] / self.tau  # [..., sd]
        z = theta[..., sd]
        kappa = theta[..., sd + 1]

        u_z = (jnp.log(z) - self.z_log_mean) / self.z_log_std

        if self.kappa_prior == "generative":
            R = jnp.cbrt(kappa / self.kappa_coeff)
            frac = jnp.clip(
                (R - self.R_min) / (self.R_max - self.R_min), _PPF_EPS, 1.0 - _PPF_EPS
            )
            u_k = _Phi_inv(frac)
        else:
            phi_b = _Phi((0.0 - self.kappa_mean) / self.kappa_std)
            p = jnp.clip(
                _Phi((kappa - self.kappa_mean) / self.kappa_std) / phi_b,
                _PPF_EPS,
                1.0 - _PPF_EPS,
            )
            u_k = _Phi_inv(p)

        return jnp.concatenate([u_h, u_z[..., None], u_k[..., None]], axis=-1)

    # ------------------------------------------------------------------
    # Forward (mean) model
    # ------------------------------------------------------------------
    @check_shapes("xi: [*batch, d]", "theta: [*batch, p]", "return: [*batch, 1]")
    def _forward(self, xi: Array, theta: Array) -> Array:
        """Surface gravity anomaly ``g(xi; theta)`` (microGal)."""
        sd = self.search_dim
        horiz = theta[..., :sd]  # source horizontal coords
        z = theta[..., sd]
        kappa = theta[..., sd + 1]
        r2 = jnp.sum((xi - horiz) ** 2, axis=-1)  # squared horizontal distance
        g = kappa * z / (r2 + z**2) ** 1.5
        return g[..., None]

    def get_obvs_sigma(self) -> float:
        # Constant additive Gaussian noise -> a single scalar std (float contract).
        return float(self.sigma)

    # ------------------------------------------------------------------
    # Prior / latent
    # ------------------------------------------------------------------
    def sample_prior(self, key: Key, batch_shape: Shape) -> tuple[Array, Key]:
        key, subkey = jax.random.split(key)
        u = jax.random.normal(
            subkey, (*batch_shape, 1, self.theta_dim), dtype=self.dtype
        )
        theta = self._theta_transform(u)
        theta = jnp.broadcast_to(theta, (*batch_shape, self.T, self.theta_dim))
        return theta, key

    @check_shapes("theta: [*batch, T, p]", "return: [*batch]")
    def prior_log_lik(self, theta: Array) -> Array:
        # Latent-space standard-normal density of u = theta_transform_inv(theta)
        # (matches the StochasticPendulum convention; consistent
        # with the z-space posterior samplers which add log N(u; 0, I)).
        u = self._theta_transform_inv(theta[..., 0, :])  # [*batch, theta_dim]
        return jnp.array(
            jax.scipy.stats.multivariate_normal.logpdf(
                u,
                mean=jnp.zeros(self.theta_dim, dtype=self.dtype),
                cov=jnp.eye(self.theta_dim, dtype=self.dtype),
            )
        )

    # ------------------------------------------------------------------
    # Noise / data sampling
    # ------------------------------------------------------------------
    def sample_epsilon(self, key: Key, batch_shape: Shape) -> tuple[Array, Key]:
        key, subkey = jax.random.split(key)
        epsilon = jax.random.normal(
            subkey, (*batch_shape, self.T, self.d_y), dtype=self.dtype
        )
        return epsilon, key

    def sample_policy_epsilon(self, key: Key, batch_shape: Shape) -> tuple[Array, Key]:
        key, subkey = jax.random.split(key)
        policy_epsilon = jax.random.normal(
            subkey, (*batch_shape, self.T, self.d), dtype=self.dtype
        )
        return policy_epsilon, key

    @check_shapes(
        "xi: [*batch, T, d]",
        "theta: [*batch, T, p]",
        "return[0]: [*batch, T, 1]",
    )
    def sample_data(
        self,
        key: Key,
        xi: Array,
        theta: Array,
        noise_mult: Array | None = None,
    ) -> tuple[Array, dict, Key]:
        g = self._forward(self.design_bijector(xi), theta)  # xi is network-space
        scale = self.sigma
        if noise_mult is not None:
            scale = self.sigma * jnp.asarray(noise_mult, dtype=g.dtype)[..., None, None]
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, g.shape, dtype=self.dtype)
        return g + scale * z, {}, key

    # ------------------------------------------------------------------
    # Design sampling. Designs are produced in O(1) NETWORK space (the
    # design_bijector applies the metre scale downstream); a unit network scale
    # therefore corresponds to a physical spread of ~design_scale metres.
    # ------------------------------------------------------------------
    def _sample_designs_isotropic_gaussian(
        self,
        key: Key,
        batch_shape: Shape,
        scale: float = 1.0,
        theta: Array | None = None,
    ) -> tuple[Array, dict, Key]:
        del theta  # theta-independent; accepted for a uniform sampler interface
        key, subkey = jax.random.split(key)
        xi = scale * jax.random.normal(
            subkey, (*batch_shape, self.T, self.d), dtype=self.dtype
        )
        return xi, {}, key

    def _sample_designs_random_scale_gaussian(
        self,
        key: Key,
        batch_shape: Shape,
        scale_min: float = 0.3,
        scale_max: float = 1.2,
        theta: Array | None = None,
    ) -> tuple[Array, dict, Key]:
        del theta  # theta-independent; accepted for a uniform sampler interface
        key, scale_key, z_key = jax.random.split(key, 3)
        scale = jax.random.uniform(
            scale_key,
            (*batch_shape, 1, 1),
            minval=scale_min,
            maxval=scale_max,
            dtype=self.dtype,
        )
        xi = scale * jax.random.normal(
            z_key, (*batch_shape, self.T, self.d), dtype=self.dtype
        )
        return xi, {}, key

    def _sample_designs_adaptive(
        self,
        key: Key,
        batch_shape: Shape,
        theta: Array,
        scale_min: float = 0.1,
        scale_max: float = 0.5,
    ) -> tuple[Array, dict, Key]:
        """Theta-dependent designs: a random-scale Gaussian centred on the source.

        The horizontal source location ``theta[..., :search_dim]`` (metres) is
        mapped into O(1) network space via ``design_bijector_inv`` and used as the
        per-design centre; an isotropic Gaussian with a per-design random network
        std in ``[scale_min, scale_max]`` is added. Network std 1.0 corresponds to
        a physical spread of ``design_scale`` metres, so the defaults place designs
        within a few-to-twenty metre cloud around the source.

        ``theta`` has shape ``[*batch, T, theta_dim]`` (the source is shared across
        time, so the centre is constant over ``t``).
        """
        if theta is None:
            raise ValueError("adaptive sampler requires theta but received None")
        theta = jnp.asarray(theta, dtype=self.dtype)
        # Source horizontal location (metres) -> O(1) network space.
        centre = self.design_bijector_inv(theta[..., : self.search_dim])

        key, scale_key, z_key = jax.random.split(key, 3)
        scale = jax.random.uniform(
            scale_key,
            (*batch_shape, self.T, 1),
            minval=scale_min,
            maxval=scale_max,
            dtype=self.dtype,
        )
        z = jax.random.normal(z_key, (*batch_shape, self.T, self.d), dtype=self.dtype)
        xi = centre + scale * z
        return xi, {"random_scale": scale}, key

    def _sample_designs_random_sampler(
        self,
        key: Key,
        batch_shape: Shape,
        theta: Array | None = None,
        candidate_samplers: Sequence[str] | None = None,
        sampler_kwargs_by_type: dict[str, dict] | None = None,
        sampler_weights_by_type: dict[str, float] = SAMPLER_WEIGHTS,
    ) -> tuple[Array, dict, Key]:
        """Per-sample mixture over candidate design samplers.

        Each batch element draws a candidate sampler index from the (normalised)
        ``sampler_weights_by_type`` and takes its design from that candidate.
        ``theta`` is forwarded to every candidate; theta-independent samplers
        ignore it. The theta-dependent "adaptive" sampler is a valid candidate but
        is opt-in: it is excluded from the default candidate set and, when listed,
        requires ``theta`` to be provided (and a positive weight) to participate.
        """
        canonical_sampler_map = {
            "isotropic_gaussian": self._sample_designs_isotropic_gaussian,
            "random_scale_gaussian": self._sample_designs_random_scale_gaussian,
            "adaptive": self._sample_designs_adaptive,
        }
        theta_dependent_samplers = {"adaptive"}
        sampler_kwargs_by_type = (
            {} if sampler_kwargs_by_type is None else dict(sampler_kwargs_by_type)
        )
        if candidate_samplers is None:
            # Default mix is theta-independent so random_sampler works without a
            # latent; list "adaptive" explicitly (with theta) to mix it in.
            candidate_samplers = [
                name
                for name in canonical_sampler_map
                if name not in theta_dependent_samplers
            ]

        normalized_candidates = []
        for sampler_name in candidate_samplers:
            if sampler_name == "random_sampler":
                raise ValueError("random_sampler cannot contain itself as a candidate")
            if sampler_name not in canonical_sampler_map:
                valid = ", ".join(sorted(canonical_sampler_map))
                raise ValueError(
                    f"Unknown candidate sampler '{sampler_name}'. Valid options: {valid}"
                )
            normalized_candidates.append(sampler_name)

        if len(normalized_candidates) == 0:
            raise ValueError("candidate_samplers must be non-empty")

        candidate_weights = []
        for sampler_name in normalized_candidates:
            weight = float(sampler_weights_by_type.get(sampler_name, 0.0))
            if weight < 0.0:
                raise ValueError(
                    f"sampler weight for '{sampler_name}' must be non-negative"
                )
            candidate_weights.append(weight)

        weight_sum = sum(candidate_weights)
        if weight_sum <= 0.0:
            raise ValueError("At least one candidate sampler must have positive weight")

        # Only run samplers with non-zero weight. Zero-weight samplers are never
        # selected, so running them just wastes work and would trip their guards
        # (e.g. the adaptive theta requirement) for samplers we do not actually use.
        active_samplers = [
            name
            for name, weight in zip(normalized_candidates, candidate_weights, strict=True)
            if weight > 0.0
        ]
        active_weights = [weight for weight in candidate_weights if weight > 0.0]

        if theta is None and any(
            name in theta_dependent_samplers for name in active_samplers
        ):
            raise ValueError(
                "random_sampler mixes a theta-dependent candidate (e.g. 'adaptive') "
                "with positive weight but theta is None; use sample()/fast_sample() "
                "so theta is wired through."
            )

        candidate_probs = jnp.array(active_weights, dtype=self.dtype) / jnp.array(
            weight_sum, dtype=self.dtype
        )

        key, choice_key, sample_key = jax.random.split(key, 3)
        # One sampler index per batch element so each sample can come from a
        # different active candidate sampler.
        sampler_idx = jax.random.choice(
            choice_key,
            a=len(active_samplers),
            shape=batch_shape,
            p=candidate_probs,
        )

        # Run every active candidate sampler over the full batch, then gather per
        # sample.
        sampler_keys = jax.random.split(sample_key, len(active_samplers))
        candidate_designs = []
        for i, sampler_name in enumerate(active_samplers):
            selected_sampler_kwargs = dict(sampler_kwargs_by_type.get(sampler_name, {}))
            sampler_fn = canonical_sampler_map[sampler_name]
            xi_i, _, _ = sampler_fn(
                sampler_keys[i],
                batch_shape,
                theta=theta,
                **selected_sampler_kwargs,
            )
            candidate_designs.append(xi_i)

        stacked = jnp.stack(candidate_designs, axis=0)  # (K, *batch_shape, T, d)
        # idx broadcasts over the (T, d) tail; gather along the sampler axis.
        idx = sampler_idx[None, ..., None, None]  # (1, *batch_shape, 1, 1)
        xi = jnp.take_along_axis(stacked, idx, axis=0)[0]  # (*batch_shape, T, d)
        return xi, {}, key

    def sample_designs(
        self,
        key: Key,
        batch_shape: Shape,
        theta: Array | None = None,
        sampler_type: str | None = None,
        sampler_kwargs: dict | None = None,
    ) -> tuple[Array, dict, Key]:
        if sampler_type is None:
            sampler_type = "random_scale_gaussian"
        if sampler_kwargs is None:
            sampler_kwargs = {}
        sampler_type = sampler_type.lower()

        if sampler_type == "isotropic_gaussian":
            return self._sample_designs_isotropic_gaussian(
                key, batch_shape, **sampler_kwargs
            )
        if sampler_type == "random_scale_gaussian":
            return self._sample_designs_random_scale_gaussian(
                key, batch_shape, **sampler_kwargs
            )
        if sampler_type == "adaptive":
            if theta is None:
                raise ValueError(
                    "sampler_type 'adaptive' requires theta but none was provided. "
                    "Use sample()/fast_sample() so theta is wired through."
                )
            return self._sample_designs_adaptive(
                key, batch_shape, theta, **sampler_kwargs
            )
        if sampler_type == "random_sampler":
            return self._sample_designs_random_sampler(
                key, batch_shape, theta=theta, **sampler_kwargs
            )
        raise ValueError(
            "sampler_type must be one of {'isotropic_gaussian', "
            "'random_scale_gaussian', 'adaptive', 'random_sampler'}, "
            f"got {sampler_type!r}"
        )

    def _freeze_sampler_kwargs(
        self, sampler_kwargs: dict
    ) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((str(k), repr(v)) for k, v in sampler_kwargs.items()))

    def _get_cached_fast_design_sampler(
        self, sampler_type: str | None, sampler_kwargs: dict
    ):
        cache_key = (sampler_type, self._freeze_sampler_kwargs(sampler_kwargs))
        design_sampler = self._fast_design_sampler_cache.get(cache_key)
        if design_sampler is None:
            design_sampler = partial(
                self.sample_designs,
                sampler_type=sampler_type,
                sampler_kwargs=sampler_kwargs,
            )
            self._fast_design_sampler_cache[cache_key] = design_sampler
        return design_sampler

    @check_shapes(
        "theta: [*batch, T, p]",
        "return[0]: [*batch, T, d]",
        "return[1]: [*batch, T, 1]",
    )
    def sample_random_acquisition(
        self,
        key: Key,
        theta: Array,
        batch_shape: Shape,
        strategy: str | None = "random",
        lhc_range: float | None = 2.0,
    ) -> tuple[Array, Array, dict, Key]:
        del lhc_range
        if strategy == "random":
            xi, _, key = self.sample_designs(key, batch_shape)
            y, aux_data, key = self.sample_data(key, xi, theta)
            return xi, y, aux_data, key
        raise ValueError(f"Unknown strategy {strategy}")

    # ------------------------------------------------------------------
    # Joint sampling
    # ------------------------------------------------------------------
    def sample(
        self,
        key: Key,
        batch_shape: Shape,
        theta: Array | None = None,
        xi: Array | None = None,
        sampler_type: str | None = None,
        sampler_kwargs: dict | None = None,
    ) -> tuple[Array, Array, Array, dict, Key]:
        if sampler_type == "policy_rollout":
            return self._policy_rollout_sample(key, batch_shape)
        if theta is None:
            theta, key = self.sample_prior(key, batch_shape)
        else:
            assert theta.shape == (*batch_shape, self.T, self.theta_dim)
        if xi is None:
            xi, _, key = self.sample_designs(
                key,
                batch_shape,
                theta=theta,
                sampler_type=sampler_type,
                sampler_kwargs=sampler_kwargs,
            )
        else:
            assert xi.shape == (*batch_shape, self.T, self.d)
        y, aux_data, key = self.sample_data(key, xi, theta)
        return y, xi, theta, aux_data, key

    def fast_sample(
        self,
        key: Key,
        batch_shape: Shape,
        sampler_type: str | None = None,
        sampler_kwargs: dict | None = None,
    ) -> tuple[Array, Array, Array, dict, Key]:
        if sampler_type == "policy_rollout":
            return self._policy_rollout_sample(key, batch_shape)
        sampler_kwargs = {} if sampler_kwargs is None else sampler_kwargs
        design_sampler = self._get_cached_fast_design_sampler(
            sampler_type=sampler_type, sampler_kwargs=sampler_kwargs
        )
        return fast_sampler(
            key, batch_shape, self.sample_prior, self.sample_data, design_sampler
        )

    # ------------------------------------------------------------------
    # Reparametrised sampling
    # ------------------------------------------------------------------
    @check_shapes(
        "epsilon: [*batch, 1]",
        "xi: [*batch, d]",
        "theta: [*batch, p]",
        "return[0]: [*batch, 1]",
    )
    def data_reparam(
        self, epsilon: Array, xi: Array, theta: Array
    ) -> tuple[Array, dict]:
        g = self._forward(self.design_bijector(xi), theta)  # xi is network-space
        return g + self.sigma * epsilon, {}

    @check_shapes(
        "theta: [*b, T, p]",
        "epsilon: [*b, T, 1]",
        "return[0]: [*b, T, 1]",
        "return[1]: [*b, T, d]",
    )
    def reparam_sample(
        self,
        params: PyTree,
        policy_net: nn.Module,
        theta: Array,
        epsilon: Array,
        policy_epsilon: Array,
    ) -> tuple[Array, Array, dict]:
        reparam_sampler = self._get_cached_reparam_sampler(
            policy_net=policy_net,
            factory_name="standard",
            sampler_factory=lambda net: get_reparam_sampler(
                net, self.data_reparam, self.T, self.theta_shape
            ),
        )
        return reparam_sampler(params, theta, epsilon, policy_epsilon)

    # ------------------------------------------------------------------
    # Log likelihoods
    # ------------------------------------------------------------------
    @check_shapes(
        "yT: [*batch, T, 1]",
        "xT: [*batch, T, d]",
        "theta: [*batch, T, p]",
        "return: [*batch, T]",
    )
    def per_step_data_log_lik(
        self,
        yT: Array,
        xT: Array,
        theta: Array,
        aux_data: dict | None = None,
        noise_mult: Array | None = None,
    ) -> Array:
        g = self._forward(self.design_bijector(xT), theta)  # xT is network-space
        if noise_mult is None:
            scale = self.sigma
        else:
            scale = self.sigma * jnp.asarray(noise_mult, dtype=yT.dtype)[..., None]
        return jax.scipy.stats.norm.logpdf(
            yT.squeeze(axis=-1), g.squeeze(axis=-1), scale
        )

    def data_log_lik(
        self,
        yT: Array,
        xT: Array,
        theta: Array,
        aux_data: dict | None = None,
        noise_mult: Array | None = None,
    ) -> Array:
        return jnp.sum(
            self.per_step_data_log_lik(yT, xT, theta, aux_data, noise_mult=noise_mult),
            axis=-1,
        )

    @check_shapes(
        "yT: [*batch, T, 1]",
        "xT: [*batch, T, d]",
        "theta: [*batch, T, p]",
        "return: [*batch]",
    )
    def log_lik(
        self,
        yT: Array,
        xT: Array,
        theta: Array,
        aux_data: dict | None = None,
        noise_mult: Array | None = None,
    ) -> Array:
        # Data likelihood only (the z-space posterior samplers add the latent
        # prior separately; including the theta-prior here would double-count it).
        return self.data_log_lik(yT, xT, theta, aux_data, noise_mult=noise_mult)

    # ------------------------------------------------------------------
    # Marginal design distribution. Designs are in O(1) NETWORK space, so the
    # design prior is ~ unit-scale Gaussian (the metre scale lives in the bijector).
    # ------------------------------------------------------------------
    @check_shapes("xi: [*batch, T, d]", "return: [*batch]")
    def marginal_design_log_lik(self, xi: Array, design_sigma: float = 1.0) -> Array:
        xi_flat = jnp.reshape(xi, (*xi.shape[:-2], self.T * self.d))
        return jnp.array(
            jax.scipy.stats.multivariate_normal.logpdf(
                xi_flat,
                mean=jnp.zeros(xi_flat.shape[-1], dtype=self.dtype),
                cov=design_sigma**2 * jnp.eye(xi_flat.shape[-1], dtype=self.dtype),
            )
        )

    def marginal_design_score(self, xT: Array, design_sigma: float = 1.0) -> Array:
        return -xT / design_sigma**2
