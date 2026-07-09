import inspect
from collections.abc import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from jaxtyping import PRNGKeyArray as Key
from jaxtyping import PyTree

from policy_learning.bed_models.base import BEDModel


def noop(*args, **kwargs):
    return None


def _score_fn_accepts_key(score_fn: Callable) -> bool:
    """Heuristic: a score fn taking >=4 positional args also accepts a PRNG key
    (y, xi, aux_data, key). Used to call learned-net and MC-posterior score fns
    uniformly regardless of whether they consume randomness."""
    try:
        n_positional = sum(
            1
            for p in inspect.signature(score_fn).parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        )
    except (TypeError, ValueError):
        return False
    return n_positional >= 4


def get_grad_estimate_fn(
    policy_net: nn.Module,
    bed_model: BEDModel,
    data_score_fn: Callable,
    design_score_fn: Callable,
    N: int,  # Number of outer samples
    outer_batch_size: int = 100,  # adjust depending on M and total memory available (increase for smaller M)
    estimate_type: str = "score",
    M: int = 1000,  # Number of inner samples
    M0: int = 1,  # Number of inner samples at level 0 in MLMC estimate
    tau: float = 1.1,  # Decay rate in MLMC estimate
) -> Callable[[Key, PyTree], tuple[PyTree, Key]]:
    constant_additive_noise: bool = bed_model.constant_additive_noise

    def _reparam_sample(params, theta, epsilon, policy_epsilon):
        return bed_model.reparam_sample(params, policy_net, theta, epsilon, policy_epsilon)

    def logp(epsilon, outer_theta, params, inner_theta, policy_epsilon):
        r"""log p_m(y, | \xi, \theta)"""
        yT, xT, aux_data = _reparam_sample(params, outer_theta, epsilon, policy_epsilon)
        return bed_model.data_log_lik(yT, xT, inner_theta, aux_data)

    def grad_logp(epsilon, outer_theta, phi, inner_theta, policy_epsilon):
        r"""Full gradient of log p_m(y, | \xi, \theta) wrt \phi"""
        return jax.grad(
            lambda params: logp(
                epsilon, outer_theta, params, inner_theta, policy_epsilon
            )
        )(phi)

    def zero_likelihood_term(params: PyTree) -> PyTree:
        """Returns a pytree of zeros with the same structure as params.
        Used when constant_additive_noise=True, since the likelihood term
        grad_logp is analytically zero in that regime."""
        return jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params)

    def logp_stopygrad(epsilon, outer_theta, params, inner_theta, policy_epsilon):
        yT, xT, aux_data = _reparam_sample(params, outer_theta, epsilon, policy_epsilon)
        return bed_model.data_log_lik(
            jax.lax.stop_gradient(yT), xT, inner_theta, aux_data
        )

    def logp_stopxigrad(epsilon, outer_theta, params, inner_theta, policy_epsilon):
        yT, xT, aux_data = _reparam_sample(params, outer_theta, epsilon, policy_epsilon)
        return bed_model.data_log_lik(
            yT, jax.lax.stop_gradient(xT), inner_theta, aux_data
        )

    data_score_accepts_key = _score_fn_accepts_key(data_score_fn)
    design_score_accepts_key = _score_fn_accepts_key(design_score_fn)

    def _eval_data_score(y, xi, aux_data, key):
        if data_score_accepts_key:
            return data_score_fn(y, xi, aux_data, key)
        return data_score_fn(y, xi, aux_data)

    def _eval_design_score(y, xi, aux_data, key):
        if design_score_accepts_key:
            return design_score_fn(y, xi, aux_data, key)
        return design_score_fn(y, xi, aux_data)

    def _reparam_sample_with_vjp(theta, epsilon, params, policy_epsilon):
        (yT, xT, aux_data), reparam_vjp_fun = jax.vjp(
            lambda phi: _reparam_sample(phi, theta, epsilon, policy_epsilon),
            params,
        )
        return yT, xT, aux_data, reparam_vjp_fun

    if estimate_type == "score_det":

        @jax.jit
        def unbatched_marginal_est(params, theta, epsilon, policy_epsilon, score_key):
            yT, xT, aux_data, reparam_vjp_fun = _reparam_sample_with_vjp(
                theta, epsilon, params, policy_epsilon
            )
            batched_aux_data = jax.tree_util.tree_map(
                lambda data: data[None], aux_data
            )

            data_score = _eval_data_score(
                yT[None],
                xT[None],
                batched_aux_data,
                score_key,
            )
            xT_cotangent = jnp.zeros_like(xT)
            aux_data_cotangent = jax.tree_util.tree_map(jnp.zeros_like, aux_data)

            return reparam_vjp_fun((data_score, xT_cotangent, aux_data_cotangent))[0]

        @jax.jit
        def eig_grad_score_det(key: Key, params: PyTree):
            epsilon, key = bed_model.sample_epsilon(key, (N,))
            outer_theta, key = bed_model.sample_prior(key, (N,))
            policy_epsilon, key = bed_model.sample_policy_epsilon(key, (N,))
            score_keys = jax.random.split(key, N + 1)
            key = score_keys[0]
            score_keys = score_keys[1:]

            if constant_additive_noise:
                likelihood_term = jax.vmap(lambda theta: zero_likelihood_term(params))(
                    outer_theta
                )
            else:
                likelihood_term = jax.vmap(
                    lambda epsilon, theta, policy_epsilon: grad_logp(
                        epsilon, theta, params, theta, policy_epsilon
                    )
                )(epsilon, outer_theta, policy_epsilon)

            marginal_term = jax.vmap(
                lambda epsilon,
                theta,
                policy_epsilon,
                score_key: unbatched_marginal_est(
                    params, theta, epsilon, policy_epsilon, score_key
                )
            )(epsilon, outer_theta, policy_epsilon, score_keys)

            return (
                jax.tree_util.tree_map(
                    lambda l, m: jnp.mean(l - m, axis=0), likelihood_term, marginal_term
                ),
                key,
            )

        return eig_grad_score_det

    elif estimate_type == "score":

        @jax.jit
        def unbatched_marginal_est(epsilon, theta, params, policy_epsilon, score_key):
            yT, xT, aux_data, reparam_vjp_fun = _reparam_sample_with_vjp(
                theta, epsilon, params, policy_epsilon
            )
            batched_aux_data = jax.tree_util.tree_map(
                lambda data: data[None], aux_data
            )

            data_score_key, design_score_key = jax.random.split(score_key)

            data_score = _eval_data_score(
                yT[None],
                xT[None],
                batched_aux_data,
                data_score_key,
            )
            design_score = _eval_design_score(
                yT[None],
                xT[None],
                batched_aux_data,
                design_score_key,
            )
            aux_data_cotangent = jax.tree_util.tree_map(jnp.zeros_like, aux_data)

            return reparam_vjp_fun((data_score, design_score, aux_data_cotangent))[0]

        @jax.jit
        def eig_grad_score(key: Key, params: PyTree):
            epsilon, key = bed_model.sample_epsilon(key, (N,))
            outer_theta, key = bed_model.sample_prior(key, (N,))
            policy_epsilon, key = bed_model.sample_policy_epsilon(key, (N,))
            score_keys = jax.random.split(key, N + 1)
            key = score_keys[0]
            score_keys = score_keys[1:]

            if constant_additive_noise:
                likelihood_term = jax.vmap(lambda theta: zero_likelihood_term(params))(
                    outer_theta
                )
            else:
                likelihood_term = jax.vmap(
                    lambda epsilon, theta, policy_epsilon: grad_logp(
                        epsilon, theta, params, theta, policy_epsilon
                    )
                )(epsilon, outer_theta, policy_epsilon)

            marginal_term = jax.vmap(
                lambda epsilon,
                theta,
                policy_epsilon,
                score_key: unbatched_marginal_est(
                    epsilon, theta, params, policy_epsilon, score_key
                )
            )(epsilon, outer_theta, policy_epsilon, score_keys)

            return (
                jax.tree_util.tree_map(
                    lambda l, m: jnp.mean(l - m, axis=0), likelihood_term, marginal_term
                ),
                key,
            )

        return eig_grad_score

    elif estimate_type == "score_chunked":
        # Identical to `score` except the per-sample marginal estimate is mapped
        # via jax.lax.map with batch_size=outer_batch_size rather than jax.vmap.
        # This bounds the effective parallelism passed to the score function,
        # so chunked / memory-aware score implementations (e.g. MonteCarloPosteriorScore)
        # are not asked to evaluate all N samples in one shot.

        @jax.jit
        def unbatched_marginal_est(epsilon, theta, params, policy_epsilon, score_key):
            yT, xT, aux_data, reparam_vjp_fun = _reparam_sample_with_vjp(
                theta, epsilon, params, policy_epsilon
            )
            batched_aux_data = jax.tree_util.tree_map(
                lambda data: data[None], aux_data
            )

            data_score_key, design_score_key = jax.random.split(score_key)

            data_score = _eval_data_score(
                yT[None],
                xT[None],
                batched_aux_data,
                data_score_key,
            )
            design_score = _eval_design_score(
                yT[None],
                xT[None],
                batched_aux_data,
                design_score_key,
            )
            aux_data_cotangent = jax.tree_util.tree_map(jnp.zeros_like, aux_data)

            return reparam_vjp_fun((data_score, design_score, aux_data_cotangent))[0]

        @jax.jit
        def eig_grad_score_chunked(key: Key, params: PyTree):
            epsilon, key = bed_model.sample_epsilon(key, (N,))
            outer_theta, key = bed_model.sample_prior(key, (N,))
            policy_epsilon, key = bed_model.sample_policy_epsilon(key, (N,))
            score_keys = jax.random.split(key, N + 1)
            key = score_keys[0]
            score_keys = score_keys[1:]

            if constant_additive_noise:
                likelihood_term = jax.vmap(lambda theta: zero_likelihood_term(params))(
                    outer_theta
                )
            else:
                likelihood_term = jax.vmap(
                    lambda epsilon, theta, policy_epsilon: grad_logp(
                        epsilon, theta, params, theta, policy_epsilon
                    )
                )(epsilon, outer_theta, policy_epsilon)

            marginal_term = jax.lax.map(
                lambda x: unbatched_marginal_est(x[0], x[1], params, x[2], x[3]),
                xs=(epsilon, outer_theta, policy_epsilon, score_keys),
                batch_size=outer_batch_size,
            )

            return (
                jax.tree_util.tree_map(
                    lambda l, m: jnp.mean(l - m, axis=0), likelihood_term, marginal_term
                ),
                key,
            )

        return eig_grad_score_chunked

    elif estimate_type == "variational_marg":

        @jax.jit
        def unbatched_marginal_est(epsilon, theta, params, policy_epsilon, score_key):
            yT, xT, aux_data, reparam_vjp_fun = _reparam_sample_with_vjp(
                theta, epsilon, params, policy_epsilon
            )
            batched_aux_data = jax.tree_util.tree_map(
                lambda data: data[None], aux_data
            )

            data_score_key, design_score_key = jax.random.split(score_key)

            data_score = _eval_data_score(
                yT[None],
                xT[None],
                batched_aux_data,
                data_score_key,
            )
            design_score = _eval_design_score(
                yT[None],
                xT[None],
                batched_aux_data,
                design_score_key,
            )
            aux_data_cotangent = jax.tree_util.tree_map(jnp.zeros_like, aux_data)

            return reparam_vjp_fun((data_score, design_score, aux_data_cotangent))[0]

        @jax.jit
        def eig_grad_variational(key: Key, params: PyTree):
            epsilon, key = bed_model.sample_epsilon(key, (N,))
            outer_theta, key = bed_model.sample_prior(key, (N,))
            policy_epsilon, key = bed_model.sample_policy_epsilon(key, (N,))
            score_keys = jax.random.split(key, N + 1)
            key = score_keys[0]
            score_keys = score_keys[1:]

            if constant_additive_noise:
                likelihood_term = jax.vmap(lambda theta: zero_likelihood_term(params))(
                    outer_theta
                )
            else:
                likelihood_term = jax.vmap(
                    lambda epsilon, theta, policy_epsilon: grad_logp(
                        epsilon, theta, params, theta, policy_epsilon
                    )
                )(epsilon, outer_theta, policy_epsilon)

            marginal_term = jax.vmap(
                lambda epsilon,
                theta,
                policy_epsilon,
                score_key: unbatched_marginal_est(
                    epsilon, theta, params, policy_epsilon, score_key
                )
            )(epsilon, outer_theta, policy_epsilon, score_keys)

            return (
                jax.tree_util.tree_map(
                    lambda l, m: jnp.mean(l - m, axis=0), likelihood_term, marginal_term
                ),
                key,
            )

        return eig_grad_variational

    elif estimate_type == "nmc":

        @jax.jit
        def grad_marginal_term(key, epsilon, outer_theta, phi, policy_epsilon):
            r"""\nabla_\phi \log 1/M \sum_{m=1}^M p_m(y, | \xi, \theta_{nm})"""
            inner_thetas, key = bed_model.sample_prior(key, (M,))

            def logp_mc_est(epsilon, outer_theta, phi, inner_thetas, policy_epsilon):
                logp_vals = jax.vmap(
                    lambda inner_theta: logp(
                        epsilon, outer_theta, phi, inner_theta, policy_epsilon
                    )
                )(inner_thetas)
                logp_est = logsumexp(logp_vals, axis=0) - jnp.log(M)
                return logp_est

            grad_marginal = jax.grad(
                lambda params: logp_mc_est(
                    epsilon, outer_theta, params, inner_thetas, policy_epsilon
                )
            )(phi)
            return grad_marginal

        @jax.jit
        def eig_grad_nmc(key: Key, params: PyTree):
            # params.shape = [T, d]
            epsilon, key = bed_model.sample_epsilon(key, (N,))
            outer_theta, key = bed_model.sample_prior(key, (N,))
            policy_epsilon, key = bed_model.sample_policy_epsilon(key, (N,))

            if constant_additive_noise:
                likelihood_term = jax.vmap(lambda theta: zero_likelihood_term(params))(
                    outer_theta
                )
            else:
                likelihood_term = jax.vmap(
                    lambda epsilon, theta, policy_epsilon: grad_logp(
                        epsilon, theta, params, theta, policy_epsilon
                    )
                )(epsilon, outer_theta, policy_epsilon)

            keys = jax.random.split(key, N + 1)
            marginal_term = jax.lax.map(
                lambda x: grad_marginal_term(x[0], x[1], x[2], params, x[3]),
                xs=(keys[1:], epsilon, outer_theta, policy_epsilon),
                batch_size=outer_batch_size,
            )
            key = keys[0]

            return (
                jax.tree_util.tree_map(
                    lambda l, m: jnp.mean(l - m, axis=0), likelihood_term, marginal_term
                ),
                key,
            )

        return eig_grad_nmc

    elif estimate_type == "spce":

        @jax.jit
        def grad_marginal_term(key, epsilon, outer_theta, phi, policy_epsilon):
            r"""\nabla_\phi \log 1/M \sum_{m=1}^M p_m(y, | \xi, \theta_{nm})"""
            inner_thetas, key = bed_model.sample_prior(key, (M,))
            inner_thetas = jnp.concatenate(
                [inner_thetas, jnp.expand_dims(outer_theta, axis=0)], axis=0
            )

            def logp_mc_est(epsilon, outer_theta, phi, inner_thetas, policy_epsilon):
                logp_vals = jax.vmap(
                    lambda inner_theta: logp(
                        epsilon, outer_theta, phi, inner_theta, policy_epsilon
                    )
                )(inner_thetas)
                logp_est = logsumexp(logp_vals, axis=0) - jnp.log(M)
                return logp_est

            grad_marginal = jax.grad(
                lambda params: logp_mc_est(
                    epsilon, outer_theta, params, inner_thetas, policy_epsilon
                )
            )(phi)
            return grad_marginal

        @jax.jit
        def eig_grad_spce(key: Key, params: PyTree):
            # params.shape = [T, d]
            epsilon, key = bed_model.sample_epsilon(key, (N,))
            outer_theta, key = bed_model.sample_prior(key, (N,))
            policy_epsilon, key = bed_model.sample_policy_epsilon(key, (N,))

            if constant_additive_noise:
                likelihood_term = jax.vmap(lambda theta: zero_likelihood_term(params))(
                    outer_theta
                )
            else:
                likelihood_term = jax.vmap(
                    lambda epsilon, theta, policy_epsilon: grad_logp(
                        epsilon, theta, params, theta, policy_epsilon
                    )
                )(epsilon, outer_theta, policy_epsilon)

            keys = jax.random.split(key, N + 1)
            marginal_term = jax.lax.map(
                lambda x: grad_marginal_term(x[0], x[1], x[2], params, x[3]),
                xs=(keys[1:], epsilon, outer_theta, policy_epsilon),
                batch_size=outer_batch_size,
            )
            key = keys[0]

            return (
                jax.tree_util.tree_map(
                    lambda l, m: jnp.mean(l - m, axis=0), likelihood_term, marginal_term
                ),
                key,
            )

        return eig_grad_spce

    elif estimate_type == "mlmc":
        """ """

        def tree_zeros_like(target):
            return jax.tree_util.tree_map(lambda a: jnp.zeros_like(a), target)

        def sample_level_mlmc(key, tau=1.1):
            p0 = 1 - 2**-tau
            p = 2**-tau
            x = jax.random.uniform(key)

            def cond_fn(state):
                l, S = state
                return x >= S

            def body_fn(state):
                l, S = state
                l += 1
                S += p0 * p**l
                return l, S

            l_init = 0
            S_init = p0
            l_final, _ = jax.lax.while_loop(cond_fn, body_fn, (l_init, S_init))
            return l_final

        def inner_loop_mlmc(
            policy_epsilon, epsilon, outer_theta, phi, inner_thetas, mask
        ):
            def lse_xi_grad(
                policy_epsilon, epsilon, outer_theta, phi, inner_thetas, mask
            ):
                vals = jax.vmap(
                    lambda inner_theta: logp_stopygrad(
                        epsilon, outer_theta, phi, inner_theta, policy_epsilon
                    )
                )(inner_thetas)
                masked_vals = jnp.where(mask, vals, -jnp.inf)
                logsumexpstuff = jax.scipy.special.logsumexp(masked_vals, axis=0)
                return logsumexpstuff

            xi_grad_part = jax.grad(
                lambda params: lse_xi_grad(
                    policy_epsilon, epsilon, outer_theta, params, inner_thetas, mask
                )
            )(phi)

            def lse_y_grad(
                policy_epsilon, epsilon, outer_theta, phi, inner_thetas, mask
            ):
                vals = jax.vmap(
                    lambda inner_theta: logp_stopxigrad(
                        epsilon, outer_theta, phi, inner_theta, policy_epsilon
                    )
                )(inner_thetas)
                masked_vals = jnp.where(mask, vals, -jnp.inf)
                logsumexpstuff = jax.scipy.special.logsumexp(masked_vals, axis=0)
                return logsumexpstuff

            y_grad_part = jax.grad(
                lambda params: lse_y_grad(
                    policy_epsilon, epsilon, outer_theta, params, inner_thetas, mask
                )
            )(phi)

            return jax.tree_util.tree_map(lambda x, y: x + y, xi_grad_part, y_grad_part)

        def eig_grad_mlmc(key: Key, params: PyTree):
            w0 = 1 - 2**-tau
            w_ratio = 2**-tau

            keys = jax.random.split(key, N + 1)
            key = keys[0]
            ls = jax.vmap(lambda k: sample_level_mlmc(k, tau))(keys[1:])
            wls = w0 * w_ratio**ls
            l_cap = int(
                -(np.log(0.0001) / (tau * np.log(2)))
            )  # Cap on the level to allow for jitting
            n_thetas_max = M0 * 2**l_cap
            ls = jnp.clip(
                ls, 0, l_cap
            )  # Might bias the statistic but we're working far in the tails

            def create_masks(n_thetas, n_thetas_max):
                mask_full = jnp.zeros(n_thetas_max)
                mask_half_a = jnp.zeros(n_thetas_max)
                mask_half_b = jnp.zeros(n_thetas_max)

                half_n_thetas = n_thetas // 2

                def loop_body(x):
                    i, mask = x
                    mask = mask.at[i].set(1)
                    i += 1
                    return (i, mask)

                def cond(i, n):
                    return i < n

                mask_full = jax.lax.while_loop(
                    lambda x: cond(x[0], n_thetas), loop_body, (0, mask_full)
                )[1]
                mask_half_a = jax.lax.while_loop(
                    lambda x: cond(x[0], half_n_thetas),
                    loop_body,
                    (0, mask_half_a),
                )[1]
                mask_half_b = jax.lax.while_loop(
                    lambda x: cond(x[0], n_thetas),
                    loop_body,
                    (half_n_thetas, mask_half_b),
                )[1]

                return mask_full, mask_half_a, mask_half_b

            def process_l0(i, key, marginal_term):
                epsilon, key = bed_model.sample_epsilon(key, ())
                outer_theta, key = bed_model.sample_prior(key, ())
                policy_epsilon, key = bed_model.sample_policy_epsilon(key, ())
                n_thetas = M0
                full_thetas, key = bed_model.sample_prior(key, (n_thetas,))
                phi = inner_loop_mlmc(
                    policy_epsilon,
                    epsilon,
                    outer_theta,
                    params,
                    full_thetas,
                    jnp.ones(n_thetas),
                )
                marginal_term = jax.tree_util.tree_map(
                    lambda a, b: a + b / (wls[i] * N),  # pyright: ignore[reportIndexIssue]
                    marginal_term,
                    phi,
                )
                return key, marginal_term

            def process_l_nonzero(i, key, marginal_term):
                epsilon, key = bed_model.sample_epsilon(key, ())
                outer_theta, key = bed_model.sample_prior(key, ())
                policy_epsilon, key = bed_model.sample_policy_epsilon(key, ())
                n_thetas = M0 * 2 ** ls[i]
                full_thetas, key = bed_model.sample_prior(key, (n_thetas_max,))
                mask_full, mask_half_a, mask_half_b = create_masks(
                    n_thetas, n_thetas_max
                )
                phi = inner_loop_mlmc(
                    policy_epsilon, epsilon, outer_theta, params, full_thetas, mask_full
                )
                phi_a = inner_loop_mlmc(
                    policy_epsilon,
                    epsilon,
                    outer_theta,
                    params,
                    full_thetas,
                    mask_half_a,
                )
                phi_b = inner_loop_mlmc(
                    policy_epsilon,
                    epsilon,
                    outer_theta,
                    params,
                    full_thetas,
                    mask_half_b,
                )
                correction_rv = jax.tree_util.tree_map(
                    lambda a, b, c: a - 0.5 * (b + c), phi, phi_a, phi_b
                )
                marginal_term = jax.tree_util.tree_map(
                    lambda a, b: a + b / (wls[i] * N),  # pyright: ignore[reportIndexIssue]
                    marginal_term,
                    correction_rv,
                )
                return key, marginal_term

            def loop_body(i, carry):
                key, marginal_term = carry
                key, marginal_term = jax.lax.cond(
                    ls[i] == 0,
                    lambda _: process_l0(i, key, marginal_term),
                    lambda _: process_l_nonzero(i, key, marginal_term),
                    operand=None,
                )
                return key, marginal_term

            marginal_term = tree_zeros_like(params)
            key, subkey = jax.random.split(key)
            carry = (subkey, marginal_term)

            key, marginal_term = jax.lax.fori_loop(0, N, loop_body, carry)

            epsilon, key = bed_model.sample_epsilon(key, (N,))
            outer_theta, key = bed_model.sample_prior(key, (N,))
            policy_epsilon, key = bed_model.sample_policy_epsilon(key, (N,))

            if constant_additive_noise:
                likelihood_term = jax.vmap(lambda theta: zero_likelihood_term(params))(
                    outer_theta
                )
            else:
                likelihood_term = jax.vmap(
                    lambda epsilon, theta, policy_epsilon: grad_logp(
                        epsilon, theta, params, theta, policy_epsilon
                    )
                )(epsilon, outer_theta, policy_epsilon)

            return (
                jax.tree_util.tree_map(
                    lambda l, m: jnp.mean(l - m, axis=0), likelihood_term, marginal_term
                ),
                key,
            )

        return eig_grad_mlmc

    else:
        raise ValueError(f"grad estimate type {estimate_type} not recognised")
