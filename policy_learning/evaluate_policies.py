from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
from check_shapes import check_shapes
from jax.errors import JaxRuntimeError
from jaxtyping import Array, PyTree
from jaxtyping import PRNGKeyArray as Key

from policy_learning.bed_models import BEDModel

# Maps (id(target_dist), id(policy_net), reuse_inner_samples, strategy, lhc_range)
# -> (target_dist, policy_net, jit_fn).
# Strong refs to target_dist and policy_net prevent id-reuse bugs across GC cycles.
_eig_fn_cache: dict[tuple, tuple] = {}


def clear_eig_fn_cache() -> None:
    """Clear cached JIT-compiled EIG bound functions.

    Call between training restarts so each new policy_net gets a fresh compiled
    function rather than a stale closure over an old network instance.
    """
    _eig_fn_cache.clear()


def eval_policy_bounds(
    key: Key,
    target_dist: BEDModel,
    policy_net: nn.Module | None,
    policy_params: PyTree | None,
    L: int = int(1e5),
    outer_samples: int = 2048,
    outer_batch_size: int = 128,
    reuse_inner_samples: bool = False,
    strategy: str = "random",
    lhc_range: float = 2.0,
):
    """Evaluate the EIG bounds for a policy network or random policy."""

    # Build (or retrieve) the JIT-compiled inner function once per
    # (target_dist, policy_net, reuse_inner_samples, strategy, lhc_range) combination.
    # Without this cache every call creates a new closure, forcing XLA to recompile
    # on every callback invocation and blocking the training step dispatch queue.
    _cache_key = (id(target_dist), id(policy_net), reuse_inner_samples, strategy, lhc_range)
    _cached = _eig_fn_cache.get(_cache_key)
    if _cached is None or _cached[0] is not target_dist or _cached[1] is not policy_net:
        if not reuse_inner_samples:

            @partial(jax.jit, static_argnums=(5))
            @check_shapes("return[0]: []")
            def _jit_fn(
                key: Key,
                policy_params: PyTree,
                outer_theta: Array,
                epsilon: Array,
                policy_epsilon: Array,
                L: int,
            ) -> tuple[Array, Array, Array]:
                true_theta = outer_theta
                if policy_net is None and policy_params is None:
                    xT, yT, aux_data, key = target_dist.sample_random_acquisition(
                        key,
                        true_theta,
                        batch_shape=(),
                        strategy=strategy,
                        lhc_range=lhc_range,
                    )
                elif policy_net is not None and policy_params is not None:
                    yT, xT, aux_data = target_dist.reparam_sample(
                        policy_params,
                        policy_net,
                        theta=true_theta,
                        epsilon=epsilon,
                        policy_epsilon=policy_epsilon,
                    )
                else:
                    raise ValueError(
                        "Please pass either None or valid values for `policy_net` and `policy_params`."
                    )
                inner_thetas, key = target_dist.sample_prior(key, (L,))
                inner_thetas = jnp.concatenate(
                    [jnp.expand_dims(true_theta, axis=0), inner_thetas], axis=0
                )
                log_lik0 = target_dist.data_log_lik(
                    yT=yT, xT=xT, theta=true_theta, aux_data=aux_data
                )
                contrastive_log_liks = jax.vmap(
                    lambda theta: target_dist.data_log_lik(
                        yT=yT, xT=xT, theta=theta, aux_data=aux_data
                    )
                )(inner_thetas)
                contrastive_log_lik_lb = jax.scipy.special.logsumexp(
                    contrastive_log_liks, axis=0
                )
                contrastive_log_lik_ub = jax.scipy.special.logsumexp(
                    contrastive_log_liks[1:], axis=0
                )
                lb = log_lik0 + jnp.log(L + 1) - contrastive_log_lik_lb
                ub = log_lik0 + jnp.log(L) - contrastive_log_lik_ub
                return lb, ub, key

        else:

            @jax.jit
            @check_shapes("return[0]: []")
            def _jit_fn(
                key: Key,
                policy_params: PyTree,
                outer_theta: Array,
                epsilon: Array,
                policy_epsilon: Array,
                inner_thetas: Array,
            ) -> tuple[Array, Array, Array]:
                true_theta = outer_theta
                if policy_net is None and policy_params is None:
                    xT, yT, aux_data, key = target_dist.sample_random_acquisition(
                        key,
                        true_theta,
                        batch_shape=(),
                        strategy=strategy,
                        lhc_range=lhc_range,
                    )
                elif policy_net is not None and policy_params is not None:
                    yT, xT, aux_data = target_dist.reparam_sample(
                        policy_params,
                        policy_net,
                        theta=true_theta,
                        epsilon=epsilon,
                        policy_epsilon=policy_epsilon,
                    )
                else:
                    raise ValueError(
                        "Please pass either None or valid values for `policy_net` and `policy_params`."
                    )
                inner_thetas = jnp.concatenate(
                    [jnp.expand_dims(true_theta, axis=0), inner_thetas], axis=0
                )
                log_lik0 = target_dist.data_log_lik(
                    yT=yT, xT=xT, theta=true_theta, aux_data=aux_data
                )
                contrastive_log_liks = jax.vmap(
                    lambda theta: target_dist.data_log_lik(
                        yT=yT, xT=xT, theta=theta, aux_data=aux_data
                    )
                )(inner_thetas)
                contrastive_log_lik_lb = jax.scipy.special.logsumexp(
                    contrastive_log_liks, axis=0
                )
                contrastive_log_lik_ub = jax.scipy.special.logsumexp(
                    contrastive_log_liks[1:], axis=0
                )
                lb = log_lik0 + jnp.log(L + 1) - contrastive_log_lik_lb
                ub = log_lik0 + jnp.log(L) - contrastive_log_lik_ub
                return lb, ub, key

        _eig_fn_cache[_cache_key] = (target_dist, policy_net, _jit_fn)
    else:
        _, _, _jit_fn = _cached

    keys = jax.random.split(key, outer_samples + 1)
    outer_thetas, key = target_dist.sample_prior(key, (outer_samples,))
    epsilons, key = target_dist.sample_epsilon(key, (outer_samples,))
    policy_epsilons, key = target_dist.sample_policy_epsilon(key, (outer_samples,))
    if reuse_inner_samples:
        inner_thetas, key = target_dist.sample_prior(key, (L,))
    else:
        inner_thetas = None
    try:
        if reuse_inner_samples:
            (lbs, ubs, _) = jax.lax.map(
                lambda x: _jit_fn(
                    x[0], policy_params, x[1], x[2], x[3], inner_thetas
                ),
                xs=(keys[1:], outer_thetas, epsilons, policy_epsilons),
                batch_size=outer_batch_size,
            )
        else:
            (lbs, ubs, _) = jax.lax.map(
                lambda x: _jit_fn(x[0], policy_params, x[1], x[2], x[3], int(L)),
                xs=(keys[1:], outer_thetas, epsilons, policy_epsilons),
                batch_size=outer_batch_size,
            )
    except JaxRuntimeError as err:
        print("Outer batch size too large:", outer_batch_size)
        success = False
        while not success:
            outer_batch_size //= 2
            if outer_batch_size == 0:
                raise ValueError("Cannot reduce outer batch size any further") from err
            try:
                print("Trying with outer batch size:", outer_batch_size)
                if reuse_inner_samples:
                    (lbs, ubs, _) = jax.lax.map(
                        lambda x: _jit_fn(
                            x[0], policy_params, x[1], x[2], x[3], inner_thetas
                        ),
                        xs=(keys[1:], outer_thetas, epsilons, policy_epsilons),
                        batch_size=outer_batch_size,
                    )
                else:
                    (lbs, ubs, _) = jax.lax.map(
                        lambda x: _jit_fn(
                            x[0], policy_params, x[1], x[2], x[3], int(L)
                        ),
                        xs=(keys[1:], outer_thetas, epsilons, policy_epsilons),
                        batch_size=outer_batch_size,
                    )
                success = True
            except JaxRuntimeError:
                success = False

    key = keys[0]

    lb = jnp.mean(lbs)
    ub = jnp.mean(ubs)
    se_lb = jnp.std(lbs, ddof=1) / jnp.sqrt(outer_samples)
    se_ub = jnp.std(ubs, ddof=1) / jnp.sqrt(outer_samples)

    return lb, ub, se_lb, se_ub, key, outer_batch_size
