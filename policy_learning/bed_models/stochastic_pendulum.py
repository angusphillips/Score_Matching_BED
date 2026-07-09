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
    get_reparam_sampler_dynamical_system,
)

Shape = Sequence[int]

SAMPLER_WEIGHTS = {
    "uniform": 1.0,
    "random_at_bounds": 1.0,
    "pseudo_random": 1.0,
    "temp_corr": 1.0,
    "tanh_normal": 0.0,
    # Unconstrained Gaussian sampler for soft_constraint mode; weight 0 so it is
    # never auto-mixed by random_sampler (select explicitly via sampler_type).
    "gaussian": 0.0,
    # Beyond-bound Fourier sampler, soft_constraint mode only; weight 0 so it is
    # never auto-mixed by random_sampler (select explicitly via sampler_type).
    "fourier": 0.0,
}


class StochasticPendulum(BEDModel):
    """Stochastic Pendulum model from https://arxiv.org/abs/2402.07868
    Note there is an inconsistency in their stated model and their implementation. They state that the acceleration component of the drift function is:
    ddq = -3 * g / (2 * l) * sin(q) + (xi - b * dq) / (m * l**2)
    whereas they implement it as:
    ddq = -3 * g / (2 * l) * sin(q) + 3 * (xi - b * dq) / (m * l**2)
    We implement the latter to maintain consistency with the results in their paper.
    """

    def __init__(
        self,
        T: int = 50,
        init_sigma: float = 1.0,
        grav_accel: float = 9.81,
        damping: float = 0.1,
        dt: float = 0.05,
        diffusion_mat: list | None = None,
        dtype: jnp.dtype = jnp.float32,
        constant_additive_noise: bool = True,
        init_seed: int = 0,
    ):
        self.key = jax.random.key(init_seed)
        self.T = T
        self.d = 1  # Dimension of design space
        self.theta_dim = 2
        self.d_y = 2  # Dimension of observation space
        self.dim = T * (
            self.d + self.d_y
        )  # Dimension of the score network output space
        self.theta_shape = (2,)
        self.init_sigma = init_sigma
        self.grav_accel = grav_accel
        self.damping = damping
        self.dt = dt
        if not diffusion_mat:
            self.diffusion_mat = jnp.diag(jnp.array([0.1, 0.1]))
        else:
            self.diffusion_mat = jnp.diag(jnp.array(diffusion_mat))
        self.dtype = dtype
        self.EPS = 1e-6
        super().__init__(self.dim)
        self.constant_additive_noise = constant_additive_noise
        self.design_bounds = jnp.array([1.0], dtype=self.dtype)
        self.design_bijector = lambda x: jnp.clip(x, -1.0, 1.0)
        self.design_bijector_inv = lambda x: x
        # Designs are box-constrained via a tanh policy output activation
        # (design_bounds * tanh); the model sees the bounded design x.
        self._policy_parameterisation = "x"

        self.theta_transform = lambda x: jnp.exp(jnp.sqrt(0.01) * x)
        self.theta_transform_inv = lambda x: jnp.log(x) / jnp.sqrt(0.01)
        self._fast_design_sampler_cache: dict[
            tuple[str | None, tuple[tuple[str, str], ...]], Callable
        ] = {}

        self.compute_standardisation(100000)

    def get_policy_output_activation(self, activation_type: str) -> Callable:
        if activation_type == "identity":
            return lambda x: x
        elif activation_type == "tanh":
            return lambda x: self.design_bounds * jnp.tanh(x)
        elif activation_type == "sigmoid":
            return lambda x: self.design_bounds * (2 * jax.nn.sigmoid(x) - 1)
        else:
            raise ValueError(f"Unknown activation type: {activation_type!r}")


    def _freeze_sampler_kwargs(
        self, sampler_kwargs: dict
    ) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((str(k), repr(v)) for k, v in sampler_kwargs.items()))

    def _get_cached_fast_design_sampler(
        self,
        sampler_type: str | None,
        sampler_kwargs: dict,
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

    def get_obvs_sigma(self):
        return jnp.sqrt(self.dt) * self.diffusion_mat

    @check_shapes(
        "y: [*batch, 2]",
        "xi: [*batch, 1]",
        "theta: [*batch, 2]",
        "return: [*batch, 2]",
    )
    def mean_dynamics_fn(self, y: Array, xi: Array, theta: Array) -> Array:
        """Mean function of the forward dynamics with EM discretisation"""
        m = theta[..., 0]
        l = theta[..., 1]
        ddq = (-3 * self.grav_accel / (2 * l)) * jnp.sin(y[..., 0]) + 3 * (
            xi[..., 0] - self.damping * y[..., 1]
        ) / (m * l**2)
        drift_term = jnp.concatenate([y[..., 1:2], ddq[..., None]], axis=-1)
        return y + self.dt * drift_term

    # Normal sampling functions
    def sample_prior(self, key: Key, batch_shape: Shape) -> tuple[Array, Key]:
        key, subkey = jax.random.split(key)
        log_theta = jnp.sqrt(0.01) * jax.random.normal(
            subkey, shape=(*batch_shape, 1, self.theta_dim), dtype=self.dtype
        )
        log_theta = jnp.broadcast_to(
            log_theta, (*batch_shape, self.T, self.theta_dim)
        )  # [*batch_shape, self.T, 2]
        return jnp.exp(log_theta), key

    def sample_epsilon(self, key: Key, batch_shape: Shape) -> tuple[Array, Key]:
        key, subkey = jax.random.split(key)
        epsilon = jax.random.normal(
            subkey, (*batch_shape, self.T + 1, 2), dtype=self.dtype
        )
        epsilon = epsilon.at[..., 0, 0].set(
            jnp.zeros_like(epsilon[..., 0, 0])
        )  # fixed initial position
        epsilon = epsilon.at[..., 0, 1].set(
            epsilon[..., 0, 1] * self.init_sigma
        )  # random initial velocity
        return epsilon, key

    def sample_policy_epsilon(self, key: Key, batch_shape: Shape) -> tuple[Array, Key]:
        key, subkey = jax.random.split(key)
        policy_epsilon = jax.random.normal(
            subkey, (*batch_shape, self.T, self.d), dtype=self.dtype
        )
        return policy_epsilon, key

    @check_shapes(
        "xi: [*batch_shape, T, 1]",
        "theta: [*batch_shape, T, 2]",
        "return[0]: [*batch_shape, T, 2]",
    )
    def sample_data(self, key: Key, xi: Array, theta: Array) -> tuple[Array, dict, Key]:
        epsilon, key = self.sample_epsilon(key, (*xi.shape[:-2],))

        @check_shapes(
            "carry_state: [b, 2]",
            "new_state[0]: [b, 1]",
            "new_state[1]: [b, 2]",
            "new_state[2]: [b, 2]",
            "return[1]: [b, 2]",
        )
        def EM_scan_fn(carry_state: Array, new_state: tuple[Array, Array, Array]):
            y = carry_state
            xi, theta, epsilon = new_state
            y_next = self.mean_dynamics_fn(y, xi, theta) + jnp.sqrt(
                self.dt
            ) * jnp.matmul(
                epsilon, self.diffusion_mat
            )  # this holds for full or partial noise
            return y_next, y_next

        if xi.ndim == 2:  # If it's unbatched then just add a batch dimension
            xi = jnp.expand_dims(xi, axis=0)
            epsilon = jnp.expand_dims(epsilon, axis=0)
            theta = jnp.expand_dims(theta, axis=0)
            unbatched = True
        elif xi.ndim == 3:
            unbatched = False
        else:
            raise ValueError("xi must be 2 or 3 dimensional")

        xi = self.design_bijector(xi)  # [b, T, 1] map back to [-1, 1]
        xs = (
            xi.transpose((1, 0, 2)),
            theta.transpose((1, 0, 2)),
            epsilon.transpose((1, 0, 2))[1:],  # First epsilon is used to sample y0
        )
        y_init = jnp.zeros_like(
            epsilon[:, 0, :]
        )  # TODO should be sampling here? Or set y_0 = (0,0)^T
        _, y = jax.lax.scan(EM_scan_fn, y_init, xs=xs)

        y = y.transpose((1, 0, 2))  # [b, T, 2]
        if unbatched:
            y = jnp.squeeze(y, axis=0)
            y_init = jnp.squeeze(y_init, axis=0)

        return (
            y,
            {"y_init": y_init},
            key,
        )

    def _sample_designs_uniform(
        self, key: Key, batch_shape: Shape
    ) -> tuple[Array, dict, Key]:
        """Uniform designs on the constrained bounds."""
        key, subkey = jax.random.split(key)
        if self._policy_parameterisation == "soft_constraint":
            raise ValueError("This sampler not meant for soft constraint")
        xi = jax.random.uniform(
            subkey,
            shape=(*batch_shape, self.T, self.d),
            dtype=self.dtype,
            minval=-self.design_bounds,
            maxval=self.design_bounds,
        )
        return xi, {}, key

    def _sample_designs_gaussian(
        self,
        key: Key,
        batch_shape: Shape,
        scale: float | None = None,
        sigma_min: float=0.4,
        sigma_max: float=1.2,
        **kwargs,
    ) -> tuple[Array, dict, Key]:
        """Unconstrained iid Gaussian designs ``xi ~ N(0, sigma**2)``.

        Intended for ``soft_constraint`` mode, where ``design_bijector_inv`` is
        the identity, so these designs reach the network/policy input space
        unchanged and cover the unbounded region the policy explores. (Not for
        ``"u"`` mode, where ``design_bijector_inv`` is ``arctanh``.)
        """
        mode = getattr(self, "_policy_parameterisation", "x")
        if mode != "soft_constraint":
            raise ValueError(
                "gaussian sampler emits beyond-bound designs and is only valid in "
                f"'soft_constraint' mode; got policy_parameterisation={mode!r}. In "
                "'x' mode sample_data clips the overshoot and in 'u' mode arctanh "
                "distorts it."
            )

        if scale is None:
            key, scale_key = jax.random.split(key)
            scale = jax.random.uniform(
                scale_key,
                (*batch_shape, 1, 1),
                minval=sigma_min,
                maxval=sigma_max,
                dtype=self.dtype,
            )
        key, subkey = jax.random.split(key)
        xi = scale * jax.random.normal(
            subkey,
            shape=(*batch_shape, self.T, self.d),
            dtype=self.dtype,
        )
        return xi, {}, key

    def _sample_designs_random_at_bounds(
        self,
        key: Key,
        batch_shape: Shape,
        u_boundary: float = 5.0,
        jitter_outward_std: float = 1.5,
        jitter_inward_std: float = 1.5,
        **kwargs,
    ) -> tuple[Array, dict, Key]:
        """Designs near saturation in u-space, mapped to x via tanh.

        Per (sample, t, d): pick sign uniformly, set u = sign * u_boundary +
        asymmetric folded-Gaussian jitter. With jitter_outward_std >
        jitter_inward_std, the distribution has a longer tail toward more
        saturation than toward the interior, matching policy behaviour near
        a committed control level. xi = design_bounds * tanh(u).
        """
        if self._policy_parameterisation == "soft_constraint":
            raise ValueError("This sampler not meant for soft constraint")
        if jitter_outward_std < 0 or jitter_inward_std < 0:
            raise ValueError("jitter stds must be non-negative")

        key, sign_key, jitter_key = jax.random.split(key, 3)
        signs = jax.random.choice(
            sign_key,
            jnp.array([-1.0, 1.0], dtype=self.dtype),
            shape=(*batch_shape, self.T, self.d),
        )
        xi = self._u_boundary_jitter(
            signs,
            jitter_key,
            u_boundary=u_boundary,
            jitter_outward_std=jitter_outward_std,
            jitter_inward_std=jitter_inward_std,
        )
        return xi, {}, key

    def _sample_designs_pseudo_random(
        self,
        key: Key,
        batch_shape: Shape,
        switch_rate_min: float = 0.02,
        switch_rate_max: float = 0.1,
        u_boundary: float = 5.0,
        jitter_outward_std: float = 1.5,
        jitter_inward_std: float = 1.5,
        **kwargs,
    ) -> tuple[Array, dict, Key]:
        """Designs near saturation with sign flipping via a Poisson-like process.

        Per (sample, dim): start at ±u_boundary, flip sign at each step with
        probability ``1 - exp(-switch_rate)`` (discrete-time Poisson). Around
        the latent ±u_boundary level, add asymmetric folded-Gaussian jitter
        (see _u_boundary_jitter).
        """
        if self._policy_parameterisation == "soft_constraint":
            raise ValueError("This sampler not meant for soft constraint")
        if switch_rate_min < 0 or switch_rate_max < 0:
            raise ValueError("switch_rate bounds must be non-negative")
        if jitter_outward_std < 0 or jitter_inward_std < 0:
            raise ValueError("jitter stds must be non-negative")
        if switch_rate_min > switch_rate_max:
            raise ValueError("switch_rate_min must be <= switch_rate_max")

        (
            key,
            subkey_init,
            subkey_switch,
            subkey_jitter,
            subkey_switch_rate,
        ) = jax.random.split(key, 5)

        init_sign = jnp.where(
            jax.random.bernoulli(subkey_init, p=0.5, shape=(*batch_shape, 1, self.d)),
            jnp.array(1.0, dtype=self.dtype),
            jnp.array(-1.0, dtype=self.dtype),
        )

        switch_rate = jax.random.uniform(
            subkey_switch_rate,
            shape=(),
            minval=switch_rate_min,
            maxval=switch_rate_max,
            dtype=self.dtype,
        )
        switch_prob = jnp.clip(1.0 - jnp.exp(-switch_rate), 0.0, 1.0)
        switches = jax.random.bernoulli(
            subkey_switch,
            p=switch_prob,
            shape=(*batch_shape, max(self.T - 1, 0), self.d),
        )
        flip_factors = jnp.where(
            switches,
            -jnp.ones_like(switches, dtype=self.dtype),
            jnp.ones_like(switches, dtype=self.dtype),
        )

        signs = jnp.concatenate([init_sign, flip_factors], axis=-2)
        sign_process = jnp.cumprod(signs, axis=-2)
        xi = self._u_boundary_jitter(
            sign_process,
            subkey_jitter,
            u_boundary=u_boundary,
            jitter_outward_std=jitter_outward_std,
            jitter_inward_std=jitter_inward_std,
        )
        return xi, {}, key

    def _sample_designs_temp_corr(
        self,
        key: Key,
        batch_shape: Shape,
        rho: float = 0.3,
        **kwargs,
    ) -> tuple[Array, dict, Key]:
        """Temporally-correlated uniform-ish designs via AR(1) mixing.

        Sample iid uniform increments on [-bounds, bounds], then mix:
            x_0 = eps_0
            x_t = rho * x_{t-1} + (1 - rho) * eps_t   for t > 0

        The result stays in [-bounds, bounds] (convex combination preserves the
        box). Marginals are NOT exactly uniform but approximately so for small
        rho. With rho=0 reduces to iid uniform; as rho->1 designs become
        increasingly persistent across time.

        rho ~ 0.3 mimics the mild temporal correlation that emerges in early
        policy-training rollouts; rho ~ 0.7 approaches pseudo_random-like behaviour.
        """
        if self._policy_parameterisation == "soft_constraint":
            raise ValueError("This sampler not meant for soft constraint")
        if rho < 0.0 or rho >= 1.0:
            raise ValueError(f"rho must be in [0, 1), got {rho}")

        key, subkey = jax.random.split(key)
        eps = jax.random.uniform(
            subkey,
            shape=(*batch_shape, self.T, self.d),
            dtype=self.dtype,
            minval=-self.design_bounds,
            maxval=self.design_bounds,
        )

        # Move T axis to front so we can scan over it.
        eps_t_first = jnp.moveaxis(eps, -2, 0)  # (T, *batch_shape, d)

        def step(prev_x, eps_t):
            new_x = rho * prev_x + (1.0 - rho) * eps_t
            return new_x, new_x

        init = eps_t_first[0]
        _, xi_tail = jax.lax.scan(step, init, eps_t_first[1:])
        xi_t_first = jnp.concatenate([init[None], xi_tail], axis=0)
        xi = jnp.moveaxis(xi_t_first, 0, -2)  # (*batch_shape, T, d)

        return xi, {}, key

    def _sample_designs_tanh_normal(
        self,
        key: Key,
        batch_shape: Shape,
        sigma_min: float = 0.5,
        sigma_max: float = 3.0,
        **kwargs,
    ) -> tuple[Array, dict, Key]:
        """Random-scale centered tanh-normal: u ~ N(0, sigma^2), xi = bounds * tanh(u).

        Per-trajectory sigma is drawn uniformly from [sigma_min, sigma_max].
        Defaults (0.5, 3.0) give p99 |u| ≈ 5 and tails out past |u| = 7.
        Use the bound-saturating samplers (random_at_bounds, pseudo_random) for
        coverage of the strictly-saturating |u| > 6 regime.
        """
        if self._policy_parameterisation == "soft_constraint":
            raise ValueError("This sampler not meant for soft constraint")
        if sigma_min <= 0.0 or sigma_max < sigma_min:
            raise ValueError(
                f"Require 0 < sigma_min <= sigma_max; got ({sigma_min}, {sigma_max})."
            )
        key, sigma_key, u_key = jax.random.split(key, 3)
        sigma = jax.random.uniform(
            sigma_key,
            shape=batch_shape,
            minval=sigma_min,
            maxval=sigma_max,
            dtype=self.dtype,
        )[..., None, None]
        u = sigma * jax.random.normal(
            u_key,
            shape=(*batch_shape, self.T, self.d),
            dtype=self.dtype,
        )
        xi = self.design_bounds * jnp.tanh(u)
        return xi, {}, key

    def _u_boundary_jitter(
        self,
        signs: Array,
        key: Key,
        u_boundary: float,
        jitter_outward_std: float,
        jitter_inward_std: float,
    ) -> Array:
        """Asymmetric folded-Gaussian jitter around sign * u_boundary in u-space.

        Half the elements receive outward jitter (away from 0, std =
        jitter_outward_std); half receive inward jitter (toward 0, std =
        jitter_inward_std). Returns xi = design_bounds * tanh(u).
        """
        mag_key, dir_key = jax.random.split(key, 2)
        z = jnp.abs(jax.random.normal(mag_key, shape=signs.shape, dtype=self.dtype))
        is_outward = jax.random.bernoulli(dir_key, p=0.5, shape=signs.shape)
        jitter_mag = jnp.where(
            is_outward, jitter_outward_std * z, -jitter_inward_std * z
        )
        u = signs * (u_boundary + jitter_mag)
        return self.design_bounds * jnp.tanh(u)

    def _sample_designs_fourier(
        self,
        key: Key,
        batch_shape: Shape,
        n_frequencies: int = 3,
        min_oscillations: float = 0.0,
        max_oscillations: float = 3.0,
        min_amplitude_fraction: float = 0.5,
        max_amplitude_fraction: float = 1.5,
        **kwargs,
    ) -> tuple[Array, dict, Key]:
        """Smooth random designs as a sum of ``n_frequencies`` sinusoids.

        Per design dimension:
            xi(t) = sum_j a_j sin(omega_j t) + b_j cos(omega_j t)
        with iid Gaussian coefficients ``(a_j, b_j)`` and frequencies ``omega_j``
        drawn uniformly so each component completes between ``min_oscillations``
        and ``max_oscillations`` full cycles over the horizon (physical time grid
        ``t = arange(T) * dt``, duration ``T * dt``).

        Per trajectory and dimension the realised signal is rescaled to a target
        peak drawn uniformly in ``[min_amplitude_fraction, max_amplitude_fraction]
        * design_bounds``. So the largest ``|xi|`` observed is at most
        ``max_amplitude_fraction`` of the bound (default 1.5x): designs sit around
        the bounds and are deliberately allowed to overshoot slightly.

        Emits unconstrained physical-space designs, so it is only valid in
        ``soft_constraint`` mode (identity bijector). ``"x"`` mode would clip the
        overshoot inside ``sample_data`` when generating y, and ``"u"`` mode would
        distort it via ``arctanh``; both are rejected.
        """
        mode = getattr(self, "_policy_parameterisation", "x")
        if mode != "soft_constraint":
            raise ValueError(
                "fourier sampler emits beyond-bound designs and is only valid in "
                f"'soft_constraint' mode; got policy_parameterisation={mode!r}. In "
                "'x' mode sample_data clips the overshoot and in 'u' mode arctanh "
                "distorts it."
            )
        if n_frequencies < 1:
            raise ValueError(f"n_frequencies must be >= 1, got {n_frequencies}")
        if not 0.0 <= min_oscillations <= max_oscillations:
            raise ValueError(
                "Require 0 <= min_oscillations <= max_oscillations; got "
                f"({min_oscillations}, {max_oscillations})."
            )
        if not 0.0 <= min_amplitude_fraction <= max_amplitude_fraction:
            raise ValueError(
                "Require 0 <= min_amplitude_fraction <= max_amplitude_fraction; "
                f"got ({min_amplitude_fraction}, {max_amplitude_fraction})."
            )

        J = int(n_frequencies)
        key, omega_key, a_key, b_key, amp_key = jax.random.split(key, 5)

        # Frequency range (rad / unit time) for the requested oscillation counts
        # over the physical horizon.
        duration = self.T * self.dt
        t = jnp.arange(self.T, dtype=self.dtype) * self.dt  # (T,)
        two_pi = 2.0 * jnp.pi
        omega_min = two_pi * min_oscillations / duration
        omega_max = two_pi * max_oscillations / duration

        coeff_shape = (*batch_shape, self.d, J)
        omega = jax.random.uniform(
            omega_key, coeff_shape, minval=omega_min, maxval=omega_max,
            dtype=self.dtype,
        )
        a = jax.random.normal(a_key, coeff_shape, dtype=self.dtype)
        b = jax.random.normal(b_key, coeff_shape, dtype=self.dtype)

        # phase[..., d, t, j] = omega[..., d, j] * t[t]  -> (*batch, d, T, J)
        phase = omega[..., None, :] * t[:, None]
        signal = jnp.sum(
            a[..., None, :] * jnp.sin(phase) + b[..., None, :] * jnp.cos(phase),
            axis=-1,
        )  # (*batch, d, T)

        # Rescale per (trajectory, dim) so the peak matches a random target in
        # [min_frac, max_frac] * bound. design_bounds is (d,).
        peak = jnp.max(jnp.abs(signal), axis=-1)  # (*batch, d)
        target_fraction = jax.random.uniform(
            amp_key, (*batch_shape, self.d),
            minval=min_amplitude_fraction, maxval=max_amplitude_fraction,
            dtype=self.dtype,
        )
        target_peak = target_fraction * self.design_bounds  # (*batch, d)
        scale = target_peak / (peak + 1e-8)
        signal = signal * scale[..., None]  # (*batch, d, T)

        xi = jnp.swapaxes(signal, -1, -2)  # (*batch, T, d)
        return xi, {}, key

    def _sample_designs_random_sampler(
        self,
        key: Key,
        batch_shape: Shape,
        candidate_samplers: Sequence[str] | None = None,
        sampler_kwargs_by_type: dict[str, dict] | None = None,
        sampler_weights_by_type: dict[str, float] = SAMPLER_WEIGHTS,
    ) -> tuple[Array, dict, Key]:
        canonical_sampler_map = {
            "uniform": self._sample_designs_uniform,
            "random_at_bounds": self._sample_designs_random_at_bounds,
            "pseudo_random": self._sample_designs_pseudo_random,
            "temp_corr": self._sample_designs_temp_corr,
            "tanh_normal": self._sample_designs_tanh_normal,
            "gaussian": self._sample_designs_gaussian,
            "fourier": self._sample_designs_fourier,
        }
        sampler_kwargs_by_type = (
            {} if sampler_kwargs_by_type is None else dict(sampler_kwargs_by_type)
        )
        if candidate_samplers is None:
            candidate_samplers = list(canonical_sampler_map.keys())

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
        # selected, so running them just wastes work and trips their
        # policy_parameterisation guards for samplers we don't actually use; this
        # confines validity checking to the samplers in use.
        active_samplers = [
            name
            for name, weight in zip(normalized_candidates, candidate_weights)
            if weight > 0.0
        ]
        active_weights = [weight for weight in candidate_weights if weight > 0.0]

        candidate_probs = jnp.array(active_weights, dtype=self.dtype) / jnp.array(
            weight_sum, dtype=self.dtype
        )

        key, choice_key, sample_key = jax.random.split(key, 3)
        # One sampler index per batch element so each sample can come from a
        # different candidate sampler.
        sampler_idx = jax.random.choice(
            choice_key,
            a=len(active_samplers),
            shape=batch_shape,
            p=candidate_probs,
        )

        # Run every active candidate sampler over the full batch, then gather per sample.
        sampler_keys = jax.random.split(sample_key, len(active_samplers))
        candidate_designs = []
        for i, sampler_name in enumerate(active_samplers):
            selected_sampler_kwargs = dict(sampler_kwargs_by_type.get(sampler_name, {}))
            sampler_fn = canonical_sampler_map[sampler_name]
            xi_i, _, _ = sampler_fn(
                sampler_keys[i],
                batch_shape,
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
        """Sample designs with specified sampler type and kwargs.

        Args:
            key: Random key
            batch_shape: Batch shape for sampling
            sampler_type: Type of sampler to use ("uniform", "random_at_bounds", "pseudo_random",
                "temp_corr", "tanh_normal", "random_sampler").
                Defaults to "uniform" if None.
            sampler_kwargs: Keyword arguments to pass to the sampler.
                For "random_at_bounds": {"jitter_scale": float}
                For "pseudo_random": {"switch_rate_min": float, "switch_rate_max": float, "jitter_scale": float}
                For "random_sampler": optional keys like "candidate_samplers", "sampler_kwargs_by_type",
                and "sampler_weights_by_type".
                Defaults to empty dict if None.
        """
        del theta  # standard samplers do not condition on the latent
        if sampler_type is None:
            sampler_type = "uniform"
        if sampler_kwargs is None:
            sampler_kwargs = {}

        sampler_type = sampler_type.lower()

        if sampler_type == "uniform":
            xi, aux, key = self._sample_designs_uniform(key, batch_shape)
        elif sampler_type == "random_at_bounds":
            xi, aux, key = self._sample_designs_random_at_bounds(
                key, batch_shape, **sampler_kwargs
            )
        elif sampler_type == "pseudo_random":
            xi, aux, key = self._sample_designs_pseudo_random(
                key,
                batch_shape,
                **sampler_kwargs,
            )
        elif sampler_type == "temp_corr":
            xi, aux, key = self._sample_designs_temp_corr(
                key,
                batch_shape,
                **sampler_kwargs,
            )
        elif sampler_type == "tanh_normal":
            xi, aux, key = self._sample_designs_tanh_normal(
                key, batch_shape, **sampler_kwargs
            )
        elif sampler_type == "gaussian":
            xi, aux, key = self._sample_designs_gaussian(
                key, batch_shape, **sampler_kwargs
            )
        elif sampler_type == "fourier":
            xi, aux, key = self._sample_designs_fourier(
                key, batch_shape, **sampler_kwargs
            )
        elif sampler_type == "random_sampler":
            xi, aux, key = self._sample_designs_random_sampler(
                key,
                batch_shape,
                **sampler_kwargs,
            )
        else:
            raise ValueError(
                "sampler_type must be one of "
                "{'uniform', 'random_at_bounds', 'pseudo_random', 'temp_corr', "
                "'tanh_normal', 'gaussian', 'fourier', 'random_sampler'}"
            )

        # The underlying samplers produce xi in constrained (x) space. Convert to
        # network/policy input space via design_bijector_inv so the returned xi is
        # in the space the score network expects: identity in "x" mode, arctanh in
        # "u" mode. sample_data then reapplies design_bijector to recover x-space
        # designs for the likelihood.
        xi = self.design_bijector_inv(xi)
        return xi, aux, key

    def sample_random_acquisition(
        self,
        key: Key,
        theta: Array,
        batch_shape: Shape,
        strategy: str | None = "random",
        lhc_range: float | None = 1.0,  # maps to [-lhc_range, lhc_range]
    ) -> tuple[Array, Array, dict, Key]:
        if strategy == "random":  # uniform or beta if generalised
            # Random means random uniform here (but mapped to R in keeping with score stuff)
            xi, _, key = self.sample_designs(
                key, batch_shape
            )  # [*batch_shape, self.T, self.d]
            y, aux_data, key = self.sample_data(key, xi, theta)
            return xi, y, aux_data, key
        else:
            raise ValueError(f"Unknown strategy {strategy}")

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
        # sample theta
        if theta is None:
            theta, key = self.sample_prior(
                key, batch_shape
            )  # [*batch_shape, self.T, self.theta_dim]
        else:
            assert theta.shape == (*batch_shape, self.T, self.theta_dim)

        # sample designs
        if xi is None:
            xi, _, key = self.sample_designs(
                key,
                batch_shape,
                theta=theta,
                sampler_type=sampler_type,
                sampler_kwargs=sampler_kwargs,
            )  # [*batch_shape, self.T, self.d]
        else:
            assert xi.shape == (*batch_shape, self.T, self.d)

        # sample y
        y, aux_data, key = self.sample_data(
            key, xi, theta
        )  # [*batch_shape, self.T, self.d_y]

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
            sampler_type=sampler_type,
            sampler_kwargs=sampler_kwargs,
        )
        return fast_sampler(
            key,
            batch_shape,
            self.sample_prior,
            self.sample_data,
            design_sampler,
        )

    # Reparametrised sampling functions
    @check_shapes(
        "theta: [*b, T, 2]",
        "epsilon: [*b, Tplus1, 2]",
        "xi: [*b, T, 1]",
        "return[0]: [*b, T, 2]",
    )
    def data_reparam(
        self, epsilon: Array, xi: Array, theta: Array
    ) -> tuple[Array, dict]:
        @check_shapes(
            "carry_state: [b, 2]",
            "new_state[0]: [b, 1]",
            "new_state[1]: [b, 2]",
            "new_state[2]: [b, 2]",
            "return[1]: [b, 2]",
        )
        def EM_scan_fn(carry_state: Array, new_state: tuple[Array, Array, Array]):
            y = carry_state
            xi, theta, epsilon = new_state
            y_next = self.mean_dynamics_fn(y, xi, theta) + jnp.sqrt(
                self.dt
            ) * jnp.matmul(epsilon, self.diffusion_mat)
            return y_next, y_next

        if xi.ndim == 2:  # If it's unbatched then just add a batch dimension
            xi = jnp.expand_dims(xi, axis=0)
            epsilon = jnp.expand_dims(epsilon, axis=0)
            theta = jnp.expand_dims(theta, axis=0)
            unbatched = True
        elif xi.ndim == 3:
            unbatched = False
        else:
            raise ValueError("xi must be 2 or 3 dimensional")

        xi = self.design_bijector(xi)  # [b, T, 1] map back to [-1, 1]
        xs = (
            xi.transpose((1, 0, 2)),
            theta.transpose((1, 0, 2)),
            epsilon.transpose((1, 0, 2))[1:],  # First epsilon is used to sample y0
        )
        y_init = jnp.zeros_like(epsilon[:, 0, :])  # TODO again zeros or sample?
        _, y = jax.lax.scan(EM_scan_fn, y_init, xs=xs)

        y = y.transpose((1, 0, 2))  # [b, T, 2]
        if unbatched:
            y = jnp.squeeze(y, axis=0)
            y_init = jnp.squeeze(y_init, axis=0)
        return y, {"y_init": y_init}  # [b, T, 2], y_init [b, 2]

    @check_shapes(
        "theta: [*b, 2]",
        "epsilon: [*b, 2]",
        "xi: [*b, 1]",
        "y_prev: [*b, 2]",
        "return: [*b, 2]",
    )
    def data_reparam_singleT(
        self, epsilon: Array, xi: Array, theta: Array, y_prev: Array
    ) -> Array:
        xi = self.design_bijector(xi)
        y_next = self.mean_dynamics_fn(y_prev, xi, theta) + jnp.sqrt(
            self.dt
        ) * jnp.matmul(epsilon, self.diffusion_mat)
        return y_next

    @check_shapes(
        "theta: [*b, T, 2]",
        "epsilon: [*b, Tplus1, 2]",
        "return[0]: [*b, T, 2]",
        "return[1]: [*b, T, 1]",
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
            factory_name="dynamical",
            sampler_factory=lambda net: get_reparam_sampler_dynamical_system(
                net,
                self.data_reparam_singleT,
                self.data_reparam,
                self.T,
                self.theta_shape,
            ),
        )

        return reparam_sampler(params, theta, epsilon, policy_epsilon)

    # log likelihood functions
    @check_shapes("theta: [b, T, 2]", "return: [b]")
    def prior_log_lik(self, theta: Array) -> Array:
        theta = theta[:, 0, :]  # [b, T, 2] -> [b, 2]
        log_theta = jnp.log(theta)
        return jnp.array(
            jax.scipy.stats.multivariate_normal.logpdf(
                log_theta,
                mean=jnp.zeros(log_theta.shape[-1]),
                cov=0.01 * jnp.eye(log_theta.shape[-1]),
            )
        )

    @check_shapes(
        "yT: [*batch_shape, T, 2]",
        "xT: [*batch_shape, T, 1]",
        "theta: [*batch_shape, T, 2]",
        "return: [*batch_shape, T]",
    )
    def per_step_data_log_lik(
        self,
        yT: Array,
        xT: Array,
        theta: Array,
        aux_data: dict | None = None,
        noise_mult: Array | None = None,
    ) -> Array:
        xT = self.design_bijector(xT)
        assert aux_data is not None
        y_init = aux_data["y_init"]
        y_tminus1 = jnp.concatenate(
            [jnp.expand_dims(y_init, -2), yT[..., :-1, :]], axis=-2
        )
        means = self.mean_dynamics_fn(y_tminus1, xT, theta)
        base_cov = self.dt * jnp.matmul(
            self.diffusion_mat, jnp.transpose(self.diffusion_mat)
        )
        if noise_mult is None:
            cov = base_cov
        else:
            # Scale the per-trajectory covariance by lambda**2 (NCSN noise
            # annealing); the leading [*batch, 1, d_y, d_y] cov broadcasts over T.
            nm = jnp.asarray(noise_mult, dtype=yT.dtype)
            cov = jnp.reshape(nm**2, (*nm.shape, 1, 1, 1)) * base_cov
        return jax.scipy.stats.multivariate_normal.logpdf(  # type: ignore
            yT,
            mean=means,
            cov=cov,
        )  # [*batch, T]

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

    def log_lik(
        self,
        yT: Array,
        xT: Array,
        theta: Array,
        aux_data: dict | None = None,
        noise_mult: Array | None = None,
    ) -> Array:
        return self.data_log_lik(yT, xT, theta, aux_data, noise_mult=noise_mult)

    @check_shapes("xi: [*batch, T, 1]", "return: [*batch]")
    def marginal_design_log_lik(self, xi: Array) -> Array:
        """Log-pdf of Uniform[-1, 1] design prior (constant log-density)."""
        xi = jnp.squeeze(xi, axis=-1)
        return jnp.sum(jnp.full_like(xi, jnp.log(1.0 / (2.0 * 1.0))), axis=-1)

    def marginal_design_score(self, xT: Array) -> Array:
        logpdf, logpdf_grad_fn = jax.vjp(self.marginal_design_log_lik, xT)
        return logpdf_grad_fn(jnp.ones_like(logpdf))[0]
