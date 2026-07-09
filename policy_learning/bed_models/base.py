import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
from check_shapes import check_shapes
from jaxtyping import Array, PyTree
from jaxtyping import PRNGKeyArray as Key

from policy_learning.policies import IqbalGRUNet, StaticDesignNet

Shape = Sequence[int]


@partial(
    jax.jit,
    static_argnames=(
        "batch_shape",
        "prior_sampler",
        "data_sampler",
        "design_sampler",
    ),
)
def fast_sampler(
    key: Key,
    batch_shape: Shape,
    prior_sampler: Callable,
    data_sampler: Callable,
    design_sampler: Callable,
) -> tuple[Array, Array, Array, dict, Key]:
    # sample theta
    theta, key = prior_sampler(key, batch_shape)

    # sample designs. theta is forwarded so theta-dependent ("adaptive") design
    # samplers can place designs relative to the latent; standard samplers accept
    # and ignore it (see BEDModel.sample_designs).
    xi, design_aux_data, key = design_sampler(key, batch_shape, theta=theta)

    y, aux_data, key = data_sampler(key, xi, theta)

    aux_data = {**design_aux_data, **aux_data}
    return y, xi, theta, aux_data, key


def get_reparam_sampler(
    policy_net: PyTree,
    data_reparam_fn: Callable,
    T: int,
    theta_shape: Shape,
):
    theta_trailing_dims = len(theta_shape)
    if isinstance(policy_net, StaticDesignNet):

        @check_shapes(
            "theta: [*batch, T, *theta_shape]",
            "epsilon: [*batch, T, d_y]",
            "policy_epsilon: [*batch, T, d_xi]",
        )
        def reparam_sampler_static(
            params: PyTree, theta: Array, epsilon: Array, policy_epsilon: Array
        ):
            batch_shape = epsilon.shape[:-2]
            xT = policy_net.apply({"params": params})
            og_shape = xT.shape
            xT = xT[(None,) * len(batch_shape) + (slice(None),) * len(og_shape)]
            xT = jnp.broadcast_to(xT, (*batch_shape, *og_shape))
            yT, aux_data = data_reparam_fn(epsilon, xT, theta)
            return yT, xT, aux_data

        return reparam_sampler_static

    elif policy_net.concat:
        if isinstance(policy_net, IqbalGRUNet):

            @check_shapes(
                "theta: [*batch, T, *theta_shape]",
                "epsilon: [*batch, T, d_y]",
                "policy_epsilon: [*batch, T, d_xi]",
            )
            def reparam_sampler_concat(
                params: PyTree,
                theta: Array,
                epsilon: Array,
                policy_epsilon: Array,
            ):
                batch_shape = epsilon.shape[:-2]
                if batch_shape == ():
                    theta = jnp.expand_dims(theta, axis=0)
                    epsilon = jnp.expand_dims(epsilon, axis=0)
                    batch_shape = (1,)
                if policy_net.deterministic:
                    policy_epsilon = jnp.zeros_like(policy_epsilon)

                def scan_fn(carry, inputs):
                    gru_state, t = carry
                    epsilon_t, policy_epsilon_t = inputs

                    h_top = gru_state[-1]
                    xt = policy_net.apply(
                        {"params": params},
                        h=h_top,
                        z=policy_epsilon_t,
                        method="emitter_from_state",
                    )

                    yt, aux_data = data_reparam_fn(
                        epsilon_t,
                        xt,
                        theta[..., 0, *[slice(None)] * theta_trailing_dims],
                    )

                    new_encoding = policy_net.apply(
                        {"params": params}, y=yt, x=xt, method="encoder"
                    )
                    gru_state, _ = policy_net.apply(
                        {"params": params},
                        state=gru_state,
                        r_t=new_encoding,
                        method="gru_step",
                    )
                    new_t = t + 1
                    return (gru_state, new_t), (xt, yt, aux_data)

                init_state = policy_net.apply(
                    {"params": params},
                    batch_shape=batch_shape,
                    method="init_gru_state",
                )
                t0 = jnp.array(0, dtype=jnp.int32)
                _, (xT, yT, aux_data) = jax.lax.scan(
                    scan_fn,
                    init=(init_state, t0),
                    xs=(
                        jnp.moveaxis(epsilon, -2, 0),
                        jnp.moveaxis(policy_epsilon, -2, 0),
                    ),
                )

                yT = jnp.moveaxis(yT, 0, -2)
                xT = jnp.moveaxis(xT, 0, -2)  # type: ignore
                aux_data = jax.tree_util.tree_map(
                    lambda data: jnp.moveaxis(data, 0, -2), aux_data
                )

                if batch_shape == (1,):
                    return (
                        yT[0],
                        xT[0],
                        jax.tree_util.tree_map(lambda data: data[0], aux_data),
                    )
                else:
                    return yT, xT, aux_data

            return reparam_sampler_concat

        @check_shapes(
            "theta: [*batch, T, *theta_shape]",
            "epsilon: [*batch, T, d_y]",
            "policy_epsilon: [*batch, T, d_xi]",
        )
        def reparam_sampler_concat(
            params: PyTree,
            theta: Array,
            epsilon: Array,
            policy_epsilon: Array,
        ):
            batch_shape = epsilon.shape[:-2]
            if batch_shape == ():
                theta = jnp.expand_dims(theta, axis=0)
                epsilon = jnp.expand_dims(epsilon, axis=0)
                batch_shape = (1,)

            def scan_fn(carry, inputs):
                enc_buffer, t = carry
                epsilon, policy_epsilon = inputs

                xt = policy_net.apply(
                    {"params": params},
                    enc=enc_buffer,
                    t=t * jnp.ones(batch_shape, dtype=jnp.int32),
                    z=policy_epsilon,
                    method="emitter",
                )

                yt, aux_data = data_reparam_fn(
                    epsilon, xt, theta[..., 0, *[slice(None)] * theta_trailing_dims]
                )

                new_encoding = policy_net.apply(
                    {"params": params}, y=yt, x=xt, method="encoder"
                )
                new_t = t + 1
                enc_buffer = enc_buffer.at[..., new_t, :].set(new_encoding)
                return (enc_buffer, new_t), (xt, yt, aux_data)

            enc_buffer_init = (
                jnp.ones((*batch_shape, T, policy_net.encoding_dim))
                * policy_net.empty_value
            )
            t0 = jnp.array(0, dtype=jnp.int32)

            _, (xT, yT, aux_data) = jax.lax.scan(
                scan_fn,
                init=(enc_buffer_init, t0),
                xs=(
                    jnp.moveaxis(epsilon, -2, 0),
                    jnp.moveaxis(policy_epsilon, -2, 0),
                ),  # scan over time
            )

            yT = jnp.moveaxis(yT, 0, -2)
            xT = jnp.moveaxis(xT, 0, -2)
            aux_data = jax.tree_util.tree_map(
                lambda data: jnp.moveaxis(data, 0, -2), aux_data
            )

            if batch_shape == (1,):
                return (
                    yT[0],
                    xT[0],
                    jax.tree_util.tree_map(lambda data: data[0], aux_data),
                )
            else:
                return yT, xT, aux_data

        return reparam_sampler_concat
    else:

        @check_shapes(
            "theta: [*batch, T, *theta_shape]",
            "epsilon: [*batch, T, d_y]",
            "policy_epsilon: [*batch, T, d_xi]",
        )
        def reparam_sampler(
            params: PyTree, theta: Array, epsilon: Array, policy_epsilon: Array
        ):
            batch_shape = epsilon.shape[:-2]
            if batch_shape == ():
                theta = jnp.expand_dims(theta, axis=0)
                epsilon = jnp.expand_dims(epsilon, axis=0)
                batch_shape = (1,)

            def scan_fun(carry_state, new_state):
                encoding = carry_state
                epsilon, policy_epsilon = new_state

                xt = policy_net.apply(
                    {"params": params},
                    enc=encoding,
                    z=policy_epsilon,
                    method="emitter",
                )

                yt, aux_data = data_reparam_fn(
                    epsilon, xt, theta[..., 0, *[slice(None)] * theta_trailing_dims]
                )

                encoding += policy_net.apply(
                    {"params": params}, y=yt, x=xt, method="encoder"
                )
                return encoding, (xt, yt, aux_data)

            first_encoding = (
                jnp.ones((*batch_shape, policy_net.encoding_dim))
                * policy_net.empty_value
            )

            _, (xT, yT, aux_data) = jax.lax.scan(
                scan_fun,
                init=(first_encoding),
                xs=(
                    jnp.moveaxis(epsilon, -2, 0),
                    jnp.moveaxis(policy_epsilon, -2, 0),
                ),
            )

            yT = jnp.moveaxis(yT, 0, -2)
            xT = jnp.moveaxis(xT, 0, -2)
            aux_data = jax.tree_util.tree_map(
                lambda data: jnp.moveaxis(data, 0, -2), aux_data
            )

            if batch_shape == (1,):
                return (
                    yT[0],
                    xT[0],
                    jax.tree_util.tree_map(lambda data: data[0], aux_data),
                )
            else:
                return yT, xT, aux_data

        return reparam_sampler


def get_reparam_sampler_dynamical_system(
    policy_net: PyTree,
    data_reparam_fn_singleT: Callable,
    data_reparam_fn: Callable,
    T: int,
    theta_shape: Sequence[int],
):
    theta_trailing_dims = len(theta_shape)
    if isinstance(policy_net, StaticDesignNet):

        @check_shapes(
            "theta: [*batch, T, *theta_shape]",
            "epsilon: [*batch, Tplus1, d_y]",
            "policy_epsilon: [*batch, T, d_xi]",
        )
        def reparam_sampler_static(
            params: PyTree, theta: Array, epsilon: Array, policy_epsilon: Array
        ) -> tuple[Array, Array, dict]:
            batch_shape = epsilon.shape[:-2]
            xT = policy_net.apply({"params": params})
            og_shape = xT.shape
            xT = xT[(None,) * len(batch_shape) + (slice(None),) * len(og_shape)]
            xT = jnp.broadcast_to(xT, (*batch_shape, *og_shape))
            yT, aux_data = data_reparam_fn(epsilon, xT, theta)
            return yT, xT, aux_data

        return reparam_sampler_static

    elif policy_net.concat:
        if isinstance(policy_net, IqbalGRUNet):

            @check_shapes(
                "theta: [*batch, T, *theta_shape]",
                "epsilon: [*batch, Tplus1, d_y]",
                "policy_epsilon: [*batch, T, d_xi]",
            )
            def reparam_sampler_concat(
                params: PyTree,
                theta: Array,
                epsilon: Array,
                policy_epsilon: Array,
            ) -> tuple[Array, Array, dict]:
                batch_shape = epsilon.shape[:-2]
                if batch_shape == ():
                    theta = jnp.expand_dims(theta, axis=0)
                    epsilon = jnp.expand_dims(epsilon, axis=0)
                    batch_shape = (1,)
                if policy_net.deterministic:
                    policy_epsilon = jnp.zeros_like(policy_epsilon)

                def scan_fn(carry, inputs):
                    prev_y, gru_state, t = carry
                    epsilon_t, policy_epsilon_t = inputs

                    h_top = gru_state[-1]
                    xt = policy_net.apply(
                        {"params": params},
                        h=h_top,
                        z=policy_epsilon_t,
                        method="emitter_from_state",
                    )

                    yt = data_reparam_fn_singleT(
                        epsilon_t,
                        xt,
                        theta[..., 0, *[slice(None)] * theta_trailing_dims],
                        prev_y,
                    )

                    new_encoding = policy_net.apply(
                        {"params": params}, y=yt, x=xt, method="encoder"
                    )
                    gru_state, _ = policy_net.apply(
                        {"params": params},
                        state=gru_state,
                        r_t=new_encoding,
                        method="gru_step",
                    )
                    new_t = t + 1
                    return (yt, gru_state, new_t), (xt, yt)

                y_init = jnp.zeros_like(epsilon[..., 0, :])
                init_state = policy_net.apply(
                    {"params": params},
                    batch_shape=batch_shape,
                    method="init_gru_state",
                )
                t0 = jnp.array(0, dtype=jnp.int32)
                _, (xT, yT) = jax.lax.scan(
                    scan_fn,
                    init=(y_init, init_state, t0),
                    xs=(
                        jnp.moveaxis(epsilon, -2, 0)[1:],
                        jnp.moveaxis(policy_epsilon, -2, 0),
                    ),
                )

                yT = jnp.moveaxis(yT, 0, -2)
                xT = jnp.moveaxis(xT, 0, -2)  # type: ignore
                aux_data = {"y_init": y_init}

                if batch_shape == (1,):
                    return (
                        yT[0],
                        xT[0],
                        jax.tree_util.tree_map(lambda data: data[0], aux_data),
                    )
                else:
                    return yT, xT, aux_data

            return reparam_sampler_concat

        @check_shapes(
            "theta: [*batch, T, *theta_shape]",
            "epsilon: [*batch, Tplus1, d_y]",
            "policy_epsilon: [*batch, T, d_xi]",
        )
        def reparam_sampler_concat(
            params: PyTree,
            theta: Array,
            epsilon: Array,
            policy_epsilon: Array,
        ) -> tuple[Array, Array, dict]:
            batch_shape = epsilon.shape[:-2]
            if batch_shape == ():
                theta = jnp.expand_dims(theta, axis=0)
                epsilon = jnp.expand_dims(epsilon, axis=0)
                batch_shape = (1,)

            def scan_fn(carry, inputs):
                prev_y, enc_buffer, t = carry
                epsilon, policy_epsilon = inputs

                xt = policy_net.apply(
                    {"params": params},
                    enc=enc_buffer,
                    t=t * jnp.ones(batch_shape, dtype=jnp.int32),
                    z=policy_epsilon,
                    method="emitter",
                )

                yt = data_reparam_fn_singleT(
                    epsilon,
                    xt,
                    theta[..., 0, *[slice(None)] * theta_trailing_dims],
                    prev_y,
                )

                new_encoding = policy_net.apply(
                    {"params": params}, y=yt, x=xt, method="encoder"
                )
                new_t = t + 1
                enc_buffer = enc_buffer.at[..., new_t, :].set(new_encoding)
                return (yt, enc_buffer, new_t), (xt, yt)

            y_init = jnp.zeros_like(
                epsilon[..., 0, :]
            )  # Initial y is the first epsilon
            enc_buffer_init = jnp.zeros((*batch_shape, T, policy_net.encoding_dim))
            t0 = jnp.array(0, dtype=jnp.int32)
            _, (xT, yT) = jax.lax.scan(
                scan_fn,
                init=(y_init, enc_buffer_init, t0),
                xs=(
                    jnp.moveaxis(epsilon, -2, 0)[1:],
                    jnp.moveaxis(policy_epsilon, -2, 0),
                ),  # scan over time
            )

            yT = jnp.moveaxis(yT, 0, -2)
            xT = jnp.moveaxis(xT, 0, -2)
            aux_data = {"y_init": y_init}

            if batch_shape == (1,):
                return (
                    yT[0],
                    xT[0],
                    jax.tree_util.tree_map(lambda data: data[0], aux_data),
                )
            else:
                return yT, xT, aux_data

        return reparam_sampler_concat
    else:

        @check_shapes(
            "theta: [*batch, T, *theta_shape]",
            "epsilon: [*batch, Tplus1, d_y]",
            "policy_epsilon: [*batch, T, d_xi]",
        )
        def reparam_sampler(
            params: PyTree, theta: Array, epsilon: Array, policy_epsilon: Array
        ) -> tuple[Array, Array, dict]:
            batch_shape = epsilon.shape[:-2]
            if batch_shape == ():
                theta = jnp.expand_dims(theta, axis=0)
                epsilon = jnp.expand_dims(epsilon, axis=0)
                batch_shape = (1,)

            def scan_fun(carry_state, new_state):
                y_prev, encoding = carry_state
                epsilon, policy_epsilon = new_state

                xt = policy_net.apply(
                    {"params": params},
                    enc=encoding,
                    z=policy_epsilon,
                    method="emitter",
                )

                yt = data_reparam_fn_singleT(
                    epsilon,
                    xt,
                    theta[..., 0, *[slice(None)] * theta_trailing_dims],
                    y_prev,
                )

                encoding += policy_net.apply(
                    {"params": params}, y=yt, x=xt, method="encoder"
                )
                return (yt, encoding), (xt, yt)

            first_encoding = (
                jnp.ones((*batch_shape, policy_net.encoding_dim))
                * policy_net.empty_value
            )
            y_init = jnp.zeros_like(
                epsilon[..., 0, :]
            )  # Initial y is the first epsilon
            _, (xT, yT) = jax.lax.scan(
                scan_fun,
                init=(y_init, first_encoding),
                xs=(
                    jnp.moveaxis(epsilon, -2, 0)[1:],
                    jnp.moveaxis(policy_epsilon, -2, 0),
                ),
            )

            yT = jnp.moveaxis(yT, 0, -2)
            xT = jnp.moveaxis(xT, 0, -2)
            aux_data = {"y_init": y_init}

            if batch_shape == (1,):
                return (
                    yT[0],
                    xT[0],
                    jax.tree_util.tree_map(lambda data: data[0], aux_data),
                )
            else:
                return yT, xT, aux_data

        return reparam_sampler


class BEDModel(ABC):
    T: int
    d: int
    k: int
    prior: str
    theta_dim: int
    d_y: int
    theta_shape: Sequence[int]
    sigma: float
    y_mean: Array
    y_std: Array
    xi_mean: Array
    xi_std: Array
    design_bijector: Callable
    theta_transform: Callable
    theta_transform_inv: Callable

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.constant_additive_noise: bool = False
        self._reparam_sampler_cache: dict[tuple[str, int, bool], Callable] = {}
        self.theta_transform_inv = lambda x: x
        # Fixed per-model policy-parameterisation flag: "no_param" (policy emits
        # unbounded / linearly-scaled designs, bijector is identity; e.g. location
        # finding, gravimetry) or "x" (policy emits box-constrained designs via a
        # tanh output activation; the dynamical-systems models set this in __init__).
        self._policy_parameterisation: str = "no_param"

        self.design_bijector = lambda x: x
        self.design_bijector_inv = lambda x: x

    @property
    def latent_prior_entropy(self) -> float:
        """Differential entropy H[p(z)] of the prior over the latent.

        For every model in this codebase the latent ``z = theta_transform_inv(theta)``
        is a priori standard normal of dimension ``theta_dim`` (LocationFinding
        samples theta ~ N(0, I) directly; the dynamical systems sample
        theta = exp(sqrt(0.01) z) with z ~ N(0, I)). The mutual information
        I(theta; y) = I(z; y) is invariant under this bijection, so the BA bound
        can be evaluated entirely in z-space with this prior entropy.
        """
        # Pure-Python constant (no jax ops) so it stays concrete under jit.
        return float(self.theta_dim) * (0.5 * math.log(2.0 * math.pi) + 0.5)

    def _get_cached_reparam_sampler(
        self,
        policy_net: nn.Module,
        factory_name: str,
        sampler_factory: Callable[[nn.Module], Callable],
        jit_sampler: bool = True,
    ) -> Callable:
        """Lazily build and cache reparam samplers per policy-network instance."""
        cache_key = (factory_name, id(policy_net), jit_sampler)
        sampler = self._reparam_sampler_cache.get(cache_key)
        if sampler is None:
            sampler = sampler_factory(policy_net)
            if jit_sampler:
                sampler = jax.jit(sampler)
            self._reparam_sampler_cache[cache_key] = sampler  # type: ignore
        return sampler  # type: ignore

    def clear_reparam_sampler_cache(self) -> None:
        """Clear cached samplers, e.g. after swapping policy architectures."""
        self._reparam_sampler_cache.clear()

    def set_policy_rollout_context(self, policy_net: nn.Module, params: PyTree) -> None:
        # For when the sampling distribution on designs comes from a policy, we store the policy parameters here
        """Attach the (policy_net, params) used by the policy_rollout sampler."""
        self._policy_rollout_ctx: tuple[nn.Module, PyTree] = (policy_net, params)

    def _policy_rollout_sample(
        self, key: Key, batch_shape: Shape
    ) -> tuple[Array, Array, Array, dict, Key]:
        """Full ``(y, xi, theta, aux_data, key)`` from a fixed-policy rollout."""
        ctx = getattr(self, "_policy_rollout_ctx", None)
        if ctx is None:
            raise RuntimeError(
                "sampler_type='policy_rollout' requires "
                "set_policy_rollout_context(policy_net, params) first."
            )
        policy_net, params = ctx
        theta, key = self.sample_prior(key, batch_shape)
        epsilon, key = self.sample_epsilon(key, batch_shape)
        policy_epsilon, key = self.sample_policy_epsilon(key, batch_shape)
        y, xi, aux_data = self.reparam_sample(
            params, policy_net, theta, epsilon, policy_epsilon
        )
        return y, xi, theta, aux_data, key

    def get_policy_output_activation(self, activation_type: str) -> Callable:
        """Return a callable to apply as the policy emitter's final activation.

        The default implementation only supports "identity" (for unbounded design
        spaces such as LocationFinding). Bounded models override this method to
        support additional activation types such as "tanh" and "sigmoid".

        Args:
            activation_type: One of "identity", "tanh", "sigmoid" (model-dependent).

        Returns:
            A callable ``f: Array -> Array`` applied element-wise after the
            policy emitter's linear output layer.
        """
        if activation_type == "identity":
            return lambda x: x
        raise NotImplementedError(
            f"Activation type '{activation_type}' is not supported by this BEDModel. "
            "Override get_policy_output_activation in the subclass."
        )

    def compute_standardisation(self, n: int = 10000):
        y, xi, _, _, self.key = self.sample(self.key, (n,))
        self.y_mean = jnp.mean(y, axis=0)
        self.y_std = jnp.std(y, axis=0)
        self.xi_mean = jnp.mean(xi, axis=0)
        self.xi_std = jnp.std(xi, axis=0)
        self.standardised = True

    def get_std_stats_dict(self):
        if self.standardised:
            return {
                "y_mean": self.y_mean,
                "y_std": self.y_std,
                "xi_mean": self.xi_mean,
                "xi_std": self.xi_std,
            }
        else:
            raise ValueError("Standarisation statistics not yet computed")

    @abstractmethod
    def get_obvs_sigma(self):
        pass

    # Normal sampling functions
    @abstractmethod
    def sample_prior(self, key: Key, batch_shape: Shape) -> tuple[Array, Key]:
        pass

    @abstractmethod
    def sample_epsilon(self, key: Key, batch_shape: Shape) -> tuple[Array, Key]:
        pass

    @abstractmethod
    def sample_data(
        self,
        key: Key,
        xi: Array,
        theta: Array,
        noise_mult: Array | None = None,
    ) -> tuple[Array, dict, Key]:
        """Sample data. ``noise_mult`` (per-trajectory observation-noise multiplier
        lambda, broadcastable to ``[*batch]``) scales the observation noise for
        NCSN-style noise annealing; None means the model's fixed noise."""
        pass

    @abstractmethod
    def sample_designs(
        self, key: Key, batch_shape: Shape, theta: Array | None = None
    ) -> tuple[Array, dict, Key]:
        """Sample designs. ``theta`` (the latent, shape ``[*batch, T, ...]``) is
        provided so theta-dependent design samplers can condition on it; samplers
        that do not need it must accept and ignore it."""
        pass

    @abstractmethod
    def sample_policy_epsilon(self, key: Key, batch_shape: Shape) -> tuple[Array, Key]:
        pass

    @abstractmethod
    def sample_random_acquisition(
        self,
        key: Key,
        theta: Array,
        batch_shape: Shape,
        strategy: str | None = "random",
        lhc_range: float | None = 2.0,
    ) -> tuple[Array, Array, dict, Key]:
        pass

    @abstractmethod
    def sample(
        self,
        key: Key,
        batch_shape: Shape,
        theta: Array | None = None,
        xi: Array | None = None,
        sampler_type: str | None = None,
        sampler_kwargs: dict | None = None,
    ) -> tuple[Array, Array, Array, dict, Key]:
        pass

    @abstractmethod
    def fast_sample(
        self,
        key: Key,
        batch_shape: Shape,
        sampler_type: str | None = None,
        sampler_kwargs: dict | None = None,
    ) -> tuple[Array, Array, Array, dict, Key]:
        pass

    # Reparametrised sampling functions
    @abstractmethod
    def data_reparam(
        self, epsilon: Array, xi: Array, theta: Array
    ) -> tuple[Array, dict]:
        pass

    @abstractmethod
    def reparam_sample(
        self,
        params: PyTree,
        policy_net: nn.Module,
        theta: Array,
        epsilon: Array,
        policy_epsilon: Array,
    ) -> tuple[Array, Array, dict]:
        pass

    # log likelihood functions
    @abstractmethod
    def prior_log_lik(self, theta: Array) -> Array:
        pass

    @abstractmethod
    def data_log_lik(
        self,
        yT: Array,
        xT: Array,
        theta: Array,
        aux_data: dict | None = None,
        noise_mult: Array | None = None,
    ) -> Array:
        pass

    @abstractmethod
    def log_lik(
        self,
        yT: Array,
        xT: Array,
        theta: Array,
        aux_data: dict | None = None,
        noise_mult: Array | None = None,
    ) -> Array:
        pass

    @abstractmethod
    def marginal_design_log_lik(self, xT: Array) -> Array:
        pass

    @abstractmethod
    def marginal_design_score(self, xT: Array) -> Array:
        pass

    @abstractmethod
    def per_step_data_log_lik(
        self,
        yT: Array,
        xT: Array,
        theta: Array,
        aux_data: dict | None = None,
        noise_mult: Array | None = None,
    ) -> Array:
        """Per-step log-likelihoods summing to data_log_lik. Shape: [*batch, T].

        ``noise_mult`` scales the observation-noise std for NCSN-style annealing
        (cov is scaled by ``noise_mult**2``); None means the model's fixed noise.
        """
        pass
