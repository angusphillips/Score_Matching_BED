from collections.abc import Callable, Sequence
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import scipy
from check_shapes import check_shapes
from einops import rearrange
from jaxtyping import Array, PyTree
from jaxtyping import PRNGKeyArray as Key

from policy_learning.bed_models.base import (
    BEDModel,
    fast_sampler,
    get_reparam_sampler,
)

Shape = Sequence[int]

LOCATION_FINDING_PROJECTION_BASES = (
    "line",
    "polynomial",
    "sinusoids",
    "fourier",
    "spiral",
    "gp",
)

# Each basis defines how projection rank is selected.
# - mode="fixed": deterministic rank (clamped to [1, T-1])
# - mode="range": sample rank uniformly in [min_rank, min(max_rank, T-1)]
LOCATION_FINDING_PROJECTION_RANK_POLICY = {
    "line": {"mode": "fixed", "rank": 2},
    "polynomial": {"mode": "range", "min_rank": 2, "max_rank": 5},
    "sinusoids": {"mode": "range", "min_rank": 2, "max_rank": 8},
    "fourier": {"mode": "range", "min_rank": 2, "max_rank": 8},
    "spiral": {"mode": "range", "min_rank": 2, "max_rank": 6},
    "gp": {"mode": "range", "min_rank": 2, "max_rank": 6},
}

SAMPLER_WEIGHTS = {
    "isotropic_unit_gaussian": 0.0,
    "time_corr_unit_gaussian": 0.0,
    "isotropic_random_scale_gaussian": 0.0,
    "time_corr_random_scale_gaussian": 1.0,
    "random_projection": 1.0,
    "adaptive": 1.0
}

DEFAULT_SAMPLER_KWARGS = {
    "isotropic_unit_gaussian": {"scale": 1.0, "time_corr_rho": 0.0},
    "time_corr_unit_gaussian": {"scale": 1.0},
    "isotropic_random_scale_gaussian": {"time_corr_rho": 0.0},
    "time_corr_random_scale_gaussian": {},
    "random_projection": {},
    "adaptive": {},
    "random_sampler": {},
}


def make_time_grid(T):
    return jnp.linspace(0.0, 1.0, T)


def rbf_kernel(t, lengthscale=0.2):
    dists = (t[:, None] - t[None, :]) ** 2
    return jnp.exp(-0.5 * dists / (lengthscale**2))


@check_shapes("x: [*batch, d]", "theta: [*batch, k, d]", "return: [*batch, 1]")
def intensity_function(
    x: Array,
    theta: Array,
    alpha: float = 1.0,
    max_signal: float = 1e-4,
    base_intensity: float = 0.1,
) -> Array:
    @check_shapes("x: [*batch, d]", "theta: [*batch, d]", "return: [*batch]")
    def individual_source_intensity(x, theta):
        return alpha / (max_signal + jnp.sum((x - theta) ** 2, axis=-1))

    out_intensity = base_intensity + jnp.sum(
        jax.vmap(
            lambda theta: individual_source_intensity(x, theta), in_axes=-2, out_axes=-1
        )(theta),
        axis=-1,
    )
    return out_intensity[..., None]


class LocationFinding(BEDModel):
    """Samples the joint marginal distribution p*(y_{1:T}, xi_{1:T}) which we will approximate the score of.
    Also samples via the reparametrisation trick and computes log likelihoods.

    Params
    ------
    T:  Experiment time horizon
    p:  Dimension of search space
    k:  Number of hidden sources
    sigma: Observation noise variance
    """

    def __init__(
        self,
        T: int,
        d: int,
        k: int,
        init_seed: int = 0,
        d_y: int = 1,
        sigma: float = 0.5,
        dtype: jnp.dtype = jnp.float32,
        alpha: float = 1.0,
        max_signal: float = 1e-4,
        base_intensity: float = 0.1,
        standardise: bool = True,
        constant_additive_noise: bool = True,
        **kwargs,
    ):
        self.key = jax.random.key(init_seed)
        self.init_seed = init_seed
        dim = T * (d + 1)
        self.T = T
        self.d = d
        self.k = k
        self.d_y = d_y
        self.theta_dim = self.d * self.k
        self.theta_shape = (self.k, self.d)
        self.sigma = sigma
        self.alpha = alpha
        self.max_signal = max_signal
        self.base_intensity = base_intensity
        self.dtype = dtype
        super().__init__(dim)
        self.constant_additive_noise = constant_additive_noise
        self.log_intensity_fn = lambda xi, theta: jnp.log(
            intensity_function(
                x=xi,
                theta=theta,
                alpha=alpha,
                max_signal=max_signal,
                base_intensity=base_intensity,
            )
        )

        self.design_bijector = lambda x: x
        self.design_bijector_inv = lambda x: x

        self.theta_transform = lambda x: x
        self.theta_transform_inv = lambda x: x
        self._fast_design_sampler_cache: dict[
            tuple[str, tuple[tuple[str, str], ...]], Callable
        ] = {}

        if standardise:
            self.compute_standardisation(10000)

    def _freeze_sampler_kwargs(
        self, sampler_kwargs: dict
    ) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((str(k), repr(v)) for k, v in sampler_kwargs.items()))

    def _get_cached_fast_design_sampler(
        self,
        sampler_type: str,
        sampler_kwargs: dict,
    ):
        cache_key = (sampler_type, self._freeze_sampler_kwargs(sampler_kwargs))
        design_sampler = self._fast_design_sampler_cache.get(cache_key)
        if design_sampler is None:
            design_sampler = partial(
                self.sample_designs,
                sampler_type=sampler_type,
                **sampler_kwargs,
            )
            self._fast_design_sampler_cache[cache_key] = design_sampler
        return design_sampler

    def get_obvs_sigma(self):
        return self.sigma

    # Ordinary sampling functions

    def sample_prior(self, key: Key, batch_shape: Shape) -> tuple[Array, Key]:
        key, subkey = jax.random.split(key)
        theta = jax.random.normal(
            subkey, (*batch_shape, 1, self.k, self.d), dtype=self.dtype
        )
        theta = jnp.broadcast_to(theta, (*batch_shape, self.T, self.k, self.d))
        return theta, key

    def sample_epsilon(self, key: Key, batch_shape: Shape) -> tuple[Array, Key]:
        key, subkey = jax.random.split(key)
        epsilon = jax.random.normal(subkey, (*batch_shape, self.T, 1), dtype=self.dtype)
        return epsilon, key

    @check_shapes("xi: [*batch, d]", "theta: [*batch, k, d]", "return[0]: [*batch, 1]")
    def sample_data(self, key: Key, xi: Array, theta: Array) -> tuple[Array, dict, Key]:
        """Sample the marginal p(y_t|theta, xi_t) on log scale.
        Agnostic to batch dimension as long as last dimensions match and batch dimensions are consistent
        """
        assert xi.shape[-1] == theta.shape[-1]
        mean = self.log_intensity_fn(xi=xi, theta=theta)
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, mean.shape, dtype=self.dtype)
        return mean + self.sigma * z, {}, key

    def _td_dim(self) -> int:
        return self.T * self.d

    def _time_cholesky(self, rho: float | Array, jitter: float = 1e-6) -> Array:
        idx_t = jnp.arange(self.T)
        corr_t = rho ** jnp.abs(idx_t[:, None] - idx_t[None, :])
        corr_t = corr_t + jitter * jnp.eye(self.T, dtype=self.dtype)
        return jnp.linalg.cholesky(corr_t)

    def _reshape_flat_to_td(self, x_flat: Array, batch_shape: Shape) -> Array:
        return x_flat.reshape((*batch_shape, self.T, self.d))

    def _sample_designs_gaussian(
        self,
        key: Key,
        batch_shape: Shape,
        theta: Array | None = None,  # accepted for a uniform sampler signature; ignored
        scale: float | Array | None = None,
        scale_min: float = 0.2,
        scale_max: float = 5.0,
        time_corr_rho: float | Array | None = None,
        time_corr_rho_min: float = 0.7,
        time_corr_rho_max: float = 1.0,
        jitter: float = 1e-6,
    ) -> tuple[Array, dict, Key]:
        if scale is None:
            key, scale_key = jax.random.split(key)
            scale = jax.random.uniform(
                scale_key,
                (*batch_shape, 1, 1),
                minval=scale_min,
                maxval=scale_max,
                dtype=self.dtype,
            )
        if time_corr_rho is None:
            key, rho_key = jax.random.split(key)
            time_corr_rho = jax.random.uniform(
                rho_key,
                shape=(),
                minval=time_corr_rho_min,
                maxval=time_corr_rho_max,
                dtype=self.dtype,
            )

        chol_t = self._time_cholesky(time_corr_rho, jitter=jitter)

        key, z_key = jax.random.split(key)
        z = jax.random.normal(z_key, (*batch_shape, self.T, self.d), dtype=self.dtype)
        xi_base = jnp.einsum("st,...td->...sd", chol_t, z)
        xi = scale * xi_base

        return (
            xi,
            {"random_scale": scale, "time_corr_rho": time_corr_rho},
            key,
        )

    def _sample_designs_adaptive(
        self,
        key: Key,
        batch_shape: Shape,
        theta: Array | None = None,
        scale_min: float = 0.1,
        scale_max: float = 0.5,
    ) -> tuple[Array, dict, Key]:
        """Theta-dependent design sampler: a mixture of narrow, random-scale
        Gaussians centred on the latent source locations.

        For each (trajectory, time-step) we pick one of the ``k`` sources uniformly
        and draw a design from an isotropic Gaussian centred on that source with a
        per-design random standard deviation in ``[scale_min, scale_max]``. Across
        the ``T`` designs of a trajectory this realises a mixture over all sources,
        concentrating designs where the signal is informative.

        ``theta`` has shape ``[*batch, T, k, d]`` (LocationFinding broadcasts the
        sources across time, so the source set is shared over ``t``).
        """
        if theta is None:
            raise ValueError("adaptive sampler requires theta but received None")
        theta = jnp.asarray(theta, dtype=self.dtype)

        key, idx_key, scale_key, z_key = jax.random.split(key, 4)
        # One source index per (trajectory, time-step) -> (*batch, T).
        source_idx = jax.random.randint(
            idx_key, (*batch_shape, self.T), minval=0, maxval=self.k
        )
        # Gather the chosen source location for each design: (*batch, T, d).
        centres = jnp.take_along_axis(
            theta, source_idx[..., None, None], axis=-2
        )[..., 0, :]
        scale = jax.random.uniform(
            scale_key,
            (*batch_shape, self.T, 1),
            minval=scale_min,
            maxval=scale_max,
            dtype=self.dtype,
        )
        z = jax.random.normal(z_key, (*batch_shape, self.T, self.d), dtype=self.dtype)
        xi = centres + scale * z
        return xi, {"random_scale": scale, "source_idx": source_idx}, key

    def _sample_designs_random_projection(
        self,
        key: Key,
        batch_shape: Shape,
        theta: Array | None = None,  # accepted for a uniform sampler signature; ignored
        min_scale: float = 0.5,
        max_scale: float = 2.0,
        time_projection_basis: str | Sequence[str] = LOCATION_FINDING_PROJECTION_BASES,
        time_projection_basis_kwargs: dict[str, dict] | None = None,
    ) -> tuple[Array, dict, Key]:
        key, basis_choice_key, rank_key, basis_key, scale_key, z_key = jax.random.split(
            key, 6
        )

        if isinstance(time_projection_basis, str):
            basis_options = (time_projection_basis,)
        else:
            basis_options = tuple(time_projection_basis)  # type: ignore
            if len(basis_options) == 0:
                raise ValueError("time_projection_basis sequence must be non-empty")

        for basis_name in basis_options:
            if basis_name not in LOCATION_FINDING_PROJECTION_BASES:
                valid = ", ".join(LOCATION_FINDING_PROJECTION_BASES)
                raise ValueError(
                    f"Unknown time_projection_basis '{basis_name}'. Valid options: {valid}"
                )

        def _basis_config(basis_name: str) -> tuple[str, int, int]:
            rank_policy = LOCATION_FINDING_PROJECTION_RANK_POLICY[basis_name]
            rank_mode: str = rank_policy["mode"]  # type: ignore
            if rank_mode == "fixed":
                min_rank: int = int(rank_policy["rank"])  # type: ignore
                max_rank: int = int(rank_policy["rank"])  # type: ignore
                return rank_mode, min_rank, max_rank
            if rank_mode == "range":
                min_rank = int(rank_policy["min_rank"])  # type: ignore
                max_rank = int(rank_policy["max_rank"])  # type: ignore
                return rank_mode, min_rank, max_rank

            raise ValueError(
                f"Unknown rank policy mode '{rank_mode}' for basis '{basis_name}'"
            )

        max_valid_rank = max(1, self.T - 1)
        branch_max_ranks = []
        for basis_name in basis_options:
            _, _, branch_max_rank = _basis_config(basis_name)
            branch_max_ranks.append(min(max_valid_rank, branch_max_rank))
        max_projection_rank = max(branch_max_ranks)

        basis_kwargs_by_name = (
            {} if time_projection_basis_kwargs is None else time_projection_basis_kwargs
        )

        def _mask_columns(matrix: Array, active_rank: Array | int) -> Array:
            column_mask = (
                jnp.arange(max_projection_rank, dtype=self.dtype) < active_rank
            ).astype(self.dtype)
            return matrix * column_mask[None, :]

        def _build_line_basis() -> Array:
            t = make_time_grid(self.T)
            basis = jnp.stack([jnp.ones_like(t), t], axis=1)
            if max_projection_rank > 2:
                basis = jnp.pad(basis, ((0, 0), (0, max_projection_rank - 2)))
            return basis.astype(self.dtype)

        def _build_polynomial_basis() -> Array:
            t = make_time_grid(self.T)
            powers = jnp.arange(max_projection_rank)
            return (t[:, None] ** powers[None, :]).astype(self.dtype)

        def _build_sinusoids_basis(basis_key: Key, freq_range=(1.0, 5.0)) -> Array:
            t = make_time_grid(self.T)
            num_freqs = max_projection_rank // 2
            key1, key2 = jax.random.split(basis_key)
            freqs = jax.random.uniform(
                key1, (num_freqs,), minval=freq_range[0], maxval=freq_range[1]
            )
            phases = jax.random.uniform(
                key2, (num_freqs,), minval=0.0, maxval=2 * jnp.pi
            )

            phi_list = []
            for i in range(num_freqs):
                w = freqs[i]
                p = phases[i]
                phi_list.append(jnp.cos(2 * jnp.pi * w * t + p))
                phi_list.append(jnp.sin(2 * jnp.pi * w * t + p))

            basis = jnp.stack(phi_list, axis=1)
            if max_projection_rank % 2 == 1:
                basis = jnp.concatenate(
                    [basis, jnp.cos(2 * jnp.pi * freqs[0] * t)[:, None]], axis=1
                )
            return basis[:, :max_projection_rank].astype(self.dtype)

        def _build_fourier_basis() -> Array:
            t = make_time_grid(self.T)
            phi_list: list[Array] = []
            k = 1
            while len(phi_list) < max_projection_rank:
                phi_list.append(jnp.cos(2 * jnp.pi * k * t))
                if len(phi_list) < max_projection_rank:
                    phi_list.append(jnp.sin(2 * jnp.pi * k * t))
                k += 1
            return jnp.stack(phi_list[:max_projection_rank], axis=1).astype(self.dtype)

        def _build_spiral_basis(basis_key: Key, decay=None) -> Array:
            t = make_time_grid(self.T)
            num_pairs = max_projection_rank // 2
            freqs = jax.random.uniform(basis_key, (num_pairs,), minval=1.0, maxval=4.0)

            phi_list = []
            for i in range(num_pairs):
                w = freqs[i]
                scale = t if decay is None else jnp.exp(-decay * t)
                phi_list.append(scale * jnp.cos(2 * jnp.pi * w * t))
                phi_list.append(scale * jnp.sin(2 * jnp.pi * w * t))

            basis = jnp.stack(phi_list, axis=1)
            if max_projection_rank % 2 == 1:
                basis = jnp.concatenate([basis, t[:, None]], axis=1)
            return basis[:, :max_projection_rank].astype(self.dtype)

        def _build_gp_basis(lengthscale=0.2) -> Array:
            t = make_time_grid(self.T)
            K = rbf_kernel(t, lengthscale)
            eigvals, eigvecs = jnp.linalg.eigh(K)
            idx = jnp.argsort(eigvals)[::-1][:max_projection_rank]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            return (eigvecs * jnp.sqrt(eigvals)).astype(self.dtype)

        def _build_basis_matrix(basis_name: str, basis_key: Key) -> Array:
            selected_kwargs = dict(basis_kwargs_by_name.get(basis_name, {}))
            if basis_name == "line":
                return _build_line_basis()
            if basis_name == "polynomial":
                return _build_polynomial_basis()
            if basis_name == "sinusoids":
                return _build_sinusoids_basis(basis_key, **selected_kwargs)
            if basis_name == "fourier":
                return _build_fourier_basis()
            if basis_name == "spiral":
                return _build_spiral_basis(basis_key, **selected_kwargs)
            if basis_name == "gp":
                return _build_gp_basis(**selected_kwargs)
            raise ValueError(f"Unknown basis '{basis_name}'")

        branches = []
        for basis_name in basis_options:
            rank_mode, min_rank, max_rank = _basis_config(basis_name)
            branch_max_rank = min(max_valid_rank, max_rank)

            def branch_fn(
                current_key: Key,
                _basis_name=basis_name,
                _rank_mode=rank_mode,
                _min_rank=min_rank,
                _max_rank=branch_max_rank,
            ) -> tuple[Array, Key]:
                basis_matrix = _build_basis_matrix(_basis_name, basis_key)
                if _rank_mode == "fixed":
                    active_rank = jnp.array(
                        min(max_valid_rank, _max_rank), dtype=jnp.int32
                    )
                else:
                    low = max(1, _min_rank)
                    high = max(low, min(max_valid_rank, _max_rank))
                    active_rank = jax.random.randint(
                        rank_key,
                        shape=(),
                        minval=low,
                        maxval=high + 1,
                    )

                basis_matrix = _mask_columns(basis_matrix, active_rank)
                scale = jax.random.uniform(
                    scale_key, (*batch_shape, 1, 1), minval=min_scale, maxval=max_scale
                )
                z = scale * jax.random.normal(
                    z_key,
                    (*batch_shape, max_projection_rank, self.d),
                    dtype=self.dtype,
                )
                z = z * (
                    jnp.arange(max_projection_rank, dtype=self.dtype) < active_rank
                )[None, :, None].astype(self.dtype)
                xi = jnp.einsum("ts,...sd->...td", basis_matrix, z)
                return xi, current_key

            branches.append(branch_fn)

        basis_idx = (
            0
            if len(basis_options) == 1
            else jax.random.randint(
                basis_choice_key,
                shape=(),
                minval=0,
                maxval=len(basis_options),
            )
        )

        xi, key = jax.lax.switch(basis_idx, branches, key)
        return xi, {}, key

    def _sample_designs_random_sampler(
        self,
        key: Key,
        batch_shape: Shape,
        theta: Array | None = None,
        candidate_samplers: Sequence[str] | None = None,
        sampler_kwargs_by_type: dict[str, dict] = DEFAULT_SAMPLER_KWARGS,
        sampler_weights_by_type: dict[str, float] = SAMPLER_WEIGHTS,
    ) -> tuple[Array, dict, Key]:
        canonical_sampler_map = {
            "isotropic_unit_gaussian": self._sample_designs_gaussian,
            "time_corr_unit_gaussian": self._sample_designs_gaussian,
            "isotropic_random_scale_gaussian": self._sample_designs_gaussian,
            "time_corr_random_scale_gaussian": self._sample_designs_gaussian,
            "random_projection": self._sample_designs_random_projection,
            "adaptive": self._sample_designs_adaptive,
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
        # selected, so running them just wastes work and would trip their guards
        # (e.g. the adaptive theta requirement) for samplers we do not actually use.
        active_samplers = [
            name
            for name, weight in zip(normalized_candidates, candidate_weights, strict=True)
            if weight > 0.0
        ]
        active_weights = [weight for weight in candidate_weights if weight > 0.0]

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
            # Every sampler takes theta as its leading argument (theta-independent ones
            # ignore it); a theta-dependent candidate raises if it needs it but gets None.
            xi_i, _, _ = sampler_fn(
                sampler_keys[i],
                batch_shape,
                theta,
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
        sampler_map: dict[str, Callable] = {
            "isotropic_unit_gaussian": self._sample_designs_gaussian,
            "time_corr_unit_gaussian": self._sample_designs_gaussian,
            "isotropic_random_scale_gaussian": self._sample_designs_gaussian,
            "time_corr_random_scale_gaussian": self._sample_designs_gaussian,
            "random_projection": self._sample_designs_random_projection,
            "adaptive": self._sample_designs_adaptive,
            "random_sampler": self._sample_designs_random_sampler,
        }
        if sampler_type is None:
            sampler_type = "time_corr_random_scale_gaussian"

        if sampler_type not in sampler_map:
            valid = ", ".join(sorted(sampler_map))
            raise ValueError(
                f"Unknown sampler_type '{sampler_type}'. Valid options: {valid}"
            )

        if sampler_kwargs is None:
            sampler_kwargs = {}
        default_sampler_kwargs = DEFAULT_SAMPLER_KWARGS[sampler_type]
        merged_sampler_kwargs = {**default_sampler_kwargs, **sampler_kwargs}

        # Every sampler shares a uniform signature taking ``theta`` as its leading
        # argument: theta-independent samplers accept and ignore it, while
        # theta-dependent ones ("adaptive", or a random_sampler mixture containing it)
        # use it and raise if it is genuinely needed but None.
        xi, _, key = sampler_map[sampler_type](
            key, batch_shape, theta, **merged_sampler_kwargs
        )
        # Design-sampler metadata is intentionally discarded to keep aux_data simple
        # and fully batch-compatible downstream.
        return xi, {}, key

    def sample_policy_epsilon(self, key: Key, batch_shape: Shape) -> tuple[Array, Key]:
        key, subkey = jax.random.split(key)
        policy_epsilon = jax.random.normal(
            subkey, (*batch_shape, self.T, self.d), dtype=self.dtype
        )
        return policy_epsilon, key

    @check_shapes(
        "theta: [*batch, T, k, d]",
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
        if strategy == "random":
            xi, _, key = self.sample_designs(
                key, batch_shape
            )  # [*batch_shape, self.T, self.d]

            y, aux_data, key = self.sample_data(key, xi, theta)
            return xi, y, aux_data, key
        elif strategy == "lhc":
            assert len(batch_shape) <= 1
            assert lhc_range is not None
            if len(batch_shape) == 0:
                flat_shape = self.T
                xi = scipy.stats.qmc.LatinHypercube(d=self.d).random(n=flat_shape)
                xi = (xi - 0.5) * lhc_range * 2
                xi = jnp.array(xi)
                y, aux_data, key = self.sample_data(key, xi, theta)
                return xi, y, aux_data, key
            else:
                flat_shape = int(np.prod(batch_shape) * self.T)
                xi = scipy.stats.qmc.LatinHypercube(d=self.d).random(n=flat_shape)
                xi = (xi - 0.5) * lhc_range * 2
                xi = rearrange(xi, "(b T) d -> b T d", T=self.T)
                xi = jnp.array(xi)
                y, aux_data, key = self.sample_data(key, xi, theta)
            return xi, y, aux_data, key
        else:
            raise ValueError(f"Unknown strategy {strategy}")

    @check_shapes(
        "theta: [*batch, T, k, d]",
        "return[0]: [*batch, T, 1]",
        "return[1]: [*batch, T, d]",
        "return[2]: [*batch, T, k, d]",
    )
    def sample(
        self,
        key: Key,
        batch_shape: Shape,
        theta: Array | None = None,
        xi: Array | None = None,
        design_aux_data: dict | None = None,
        sampler_type: str | None = None,
        sampler_kwargs: dict | None = None,
    ) -> tuple[Array, Array, Array, dict, Key]:  # SAMPLES Y ON LOG SCALE
        if sampler_type == "policy_rollout":
            return self._policy_rollout_sample(key, batch_shape)
        sampler_kwargs = {} if sampler_kwargs is None else sampler_kwargs
        # sample theta
        if theta is None:
            theta, key = self.sample_prior(key, batch_shape)
        else:
            assert theta.shape == (*batch_shape, self.T, self.k, self.d)
        assert theta is not None
        # sample designs
        if xi is None:
            xi, design_aux_data, key = self.sample_designs(
                key,
                batch_shape,
                theta=theta,
                sampler_type=sampler_type,
                sampler_kwargs=sampler_kwargs,
            )  # [*batch_shape, self.T, self.d]
        else:
            assert xi.shape == (*batch_shape, self.T, self.d)
        assert xi is not None
        # sample y
        y, aux_data, key = self.sample_data(key, xi, theta)  # [*batch_shape, self.T, 1]

        if design_aux_data is not None:
            aux_data = {**design_aux_data, **aux_data}
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
        if sampler_type is None:
            sampler_type = "isotropic_unit_gaussian"
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
        "epsilon: [*batch, 1]",
        "xi: [*batch, d]",
        "theta: [*batch, k, d]",
        "return[0]: [*batch, 1]",
    )
    def data_reparam(
        self, epsilon: Array, xi: Array, theta: Array
    ) -> tuple[Array, dict]:  # tested
        """Sample the marginal p(y_t|theta, xi_t) using the reparam variable epsilon ON LOG SCALE.
        Agnostic to batch dimension as long as last dimensions match
        """
        assert xi.shape[-1] == theta.shape[-1]
        mean = self.log_intensity_fn(xi, theta)
        assert mean.shape == epsilon.shape
        return mean + self.sigma * epsilon, {}

    @check_shapes(
        "theta: [*b, T, k, d]",
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

    # Log likelihood functions
    @check_shapes("theta: [*batch, T, k, d]", "return: [*batch]")
    def prior_log_lik(self, theta: Array) -> Array:
        assert theta.shape[-2:] == (self.k, self.d)
        assert len(theta.shape) > 2
        theta = theta[..., 0, :, :]
        theta = jnp.reshape(theta, (*theta.shape[:-2], -1))  # [*batch_shape, k*d]
        return jnp.array(
            jax.scipy.stats.multivariate_normal.logpdf(
                theta, mean=jnp.zeros(theta.shape[-1]), cov=jnp.eye(theta.shape[-1])
            )
        )

    @check_shapes(
        "yT: [*batch, T, 1]",
        "xT: [*batch, T, d]",
        "theta: [*batch, T, k, d]",
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
        # cov = sigma^2 * I_T so observations are conditionally independent.
        mean = self.log_intensity_fn(xi=xT, theta=theta)  # [..., T, 1]
        if noise_mult is None:
            scale = self.sigma
        else:
            # NCSN noise annealing: scale the per-trajectory std by lambda. nm is
            # [*batch]; the trailing axis broadcasts over the T observations.
            scale = self.sigma * jnp.asarray(noise_mult, dtype=yT.dtype)[..., None]
        return jax.scipy.stats.norm.logpdf(
            yT.squeeze(axis=-1), mean.squeeze(axis=-1), scale
        )  # [..., T]

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
        "theta: [*batch, T, k, d]",
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
        return (
            self.data_log_lik(yT, xT, theta, aux_data, noise_mult=noise_mult)
            # + self.prior_log_lik(theta)
        )

    @check_shapes("xi: [*batch, T, d]", "return: [*batch]")
    def marginal_design_log_lik(self, xi: Array, design_sigma: float = 1.0) -> Array:
        # assert xi.shape[-2:] == (self.T, self.d)
        if xi.ndim > 2:
            xi = rearrange(xi, "... T d -> ... (T d)")
        else:
            xi = rearrange(xi, "T d -> (T d)")
        return jnp.array(
            jax.scipy.stats.multivariate_normal.logpdf(
                xi,
                mean=jnp.zeros(xi.shape[-1]),
                cov=design_sigma**2 * jnp.eye(xi.shape[-1]),
            )
        )

    def marginal_design_score(self, xT: Array, design_sigma: float = 1.0) -> Array:
        return -xT / design_sigma**2
