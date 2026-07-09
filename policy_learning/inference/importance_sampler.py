"""
Build a general importance sampler for the posterior

There should be prior data prior_y and prior_xi
There should be a likelihood function that computes log p(prior_y|prior_x, theta) + log p(theta) as a function of theta
There should be a grad log likelihood function that computes the gradient of the above, which will have to be defined as an
unbatched version

So we can just ask for an unbatched likelihood function which we can jax.grad and jax.vmap to get the two functions above which we need

Parameters are:
Number of proposal samples
MALA step number and size


"""

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array
from jaxtyping import PRNGKeyArray as Key

Shape = Sequence[int]


def _effective_sample_size_from_normalized_weights(weights: Array) -> Array:
    """Compute standard importance-sampling ESS from normalized weights."""
    return 1.0 / jnp.sum(jnp.square(weights))


def mala_step(key, theta, step_size, grad_log_lik, unbatched_log_lik):
    """Single MALA step for one particle.
    theta: [k, d]
    returns: (updated theta [k, d], key)
    """
    key, noise_key, accept_key = jax.random.split(key, 3)

    # Current gradient and log prob
    grad_current = grad_log_lik(theta)
    log_p_current = unbatched_log_lik(theta)

    # Propose: theta' = theta + (step_size/2) * grad + sqrt(step_size) * z
    noise = jax.random.normal(noise_key, theta.shape, dtype=theta.dtype)
    theta_proposed = (
        theta + 0.5 * step_size * grad_current + jnp.sqrt(step_size) * noise
    )

    # Proposed gradient and log prob
    grad_proposed = grad_log_lik(theta_proposed)
    log_p_proposed = unbatched_log_lik(theta_proposed)

    # Log proposal densities for MH correction
    # log q(theta' | theta) and log q(theta | theta')
    mean_forward = theta + 0.5 * step_size * grad_current
    mean_backward = theta_proposed + 0.5 * step_size * grad_proposed

    log_q_forward = -0.5 * jnp.sum((theta_proposed - mean_forward) ** 2) / step_size
    log_q_backward = -0.5 * jnp.sum((theta - mean_backward) ** 2) / step_size

    # Log acceptance ratio
    log_alpha = log_p_proposed - log_p_current + log_q_backward - log_q_forward

    # Accept/reject
    u = jax.random.uniform(accept_key)
    accept = jnp.log(u) < log_alpha
    theta_new = jnp.where(accept, theta_proposed, theta)

    return theta_new, key


def mala_move_batch(
    key,
    samples,
    grad_log_lik,
    unbatched_log_lik,
    step_size=0.1,
    num_steps=5,
    mala_step=mala_step,
    return_diagnostics: bool = False,
) -> tuple[Array, Key, Array | None]:
    """Apply multiple MALA steps to a batch of samples.
    samples: [num_samples, k, d]
    returns: [num_samples, k, d]
    """

    def body_fn(carry):
        samples, key = carry
        keys = jax.random.split(key, samples.shape[0] + 1)
        key, subkeys = keys[0], keys[1:]
        updated_samples = jax.vmap(
            lambda th, k: mala_step(k, th, step_size, grad_log_lik, unbatched_log_lik)[
                0
            ]
        )(samples, subkeys)

        return updated_samples, key

    if not return_diagnostics:

        def loop_body(_, carry):
            return body_fn(carry)

        samples, key = jax.lax.fori_loop(0, num_steps, loop_body, (samples, key))
        return samples, key, None

    def diag_loop_body(_, carry):
        samples, key, accepted_total = carry
        updated_samples, key = body_fn((samples, key))
        accepted = jax.vmap(lambda new_th, old_th: jnp.any(new_th != old_th))(
            updated_samples, samples
        )
        accepted_total = accepted_total + jnp.sum(accepted.astype(samples.dtype))
        return updated_samples, key, accepted_total

    accepted_total_init = jnp.array(0.0, dtype=samples.dtype)
    samples, key, accepted_total = jax.lax.fori_loop(
        0,
        num_steps,
        diag_loop_body,
        (samples, key, accepted_total_init),
    )

    total_proposals = jnp.asarray(num_steps * samples.shape[0], dtype=samples.dtype)
    acceptance_rate: Array = jnp.where(  # type: ignore
        total_proposals > 0,
        accepted_total / total_proposals,
        jnp.array(0.0, dtype=samples.dtype),
    )

    return samples, key, acceptance_rate


def sample_posterior_is(
    key: Key,
    unbatched_log_lik: Callable[[Array], Array],
    prior_sampler: Callable[[Key, Shape], tuple[Array, Key]],
    theta_shape: Shape,
    batch_shape: Shape,
    mala_steps: int = 5,
    mala_step_size: float = 0.1,
    return_ess: bool = False,
) -> tuple[Array, Key, dict]:
    """
    An importance sampler + MALA steps implementation. The unbatched_log_lik function should take a single theta sample and evaluate
    the posterior log likelihood (i.e. p(y|x, theta)p(theta)).
    """

    grad_log_lik = jax.grad(unbatched_log_lik)

    log_lik_batch = jax.vmap(unbatched_log_lik)

    num_samples = int(np.prod(batch_shape)) if batch_shape else 1

    num_proposal = num_samples

    # Step 1: Sample from the prior (proposal distribution)
    proposal_samples, key = prior_sampler(key, (num_proposal,))

    # Step 2: Compute log importance weights
    # log_weights = log p(y | theta, xi) since prior cancels with proposal
    log_weights = log_lik_batch(proposal_samples)  # [num_proposals]

    # Step 3: Normalize weights in log space for numerical stability
    log_weights_normalized = log_weights - jax.scipy.special.logsumexp(log_weights)
    weights = jnp.exp(log_weights_normalized)

    # Step 4: Resample according to weights (multinomial resampling)
    key, subkey = jax.random.split(key)
    indices = jax.random.choice(
        subkey, num_proposal, shape=(num_samples,), replace=True, p=weights
    )

    samples = proposal_samples[indices]  # [num_samples, *theta_shape]
    initial_samples = samples

    # Step 5: MALA move steps to rejuvenate particles
    # jax.lax.fori_loop in mala_move_batch naturally supports num_steps == 0.
    if return_ess:
        samples, key, mala_acceptance_rate = mala_move_batch(
            key,
            samples,
            grad_log_lik,
            unbatched_log_lik,
            step_size=mala_step_size,
            num_steps=mala_steps,
            return_diagnostics=True,
        )
    else:
        samples, key, _ = mala_move_batch(
            key,
            samples,
            grad_log_lik,
            unbatched_log_lik,
            step_size=mala_step_size,
            num_steps=mala_steps,
        )

    # Reshape to batch_shape, don't broadcast T dimension, has to be done manually afterwards
    samples = jnp.reshape(samples, (*batch_shape, *theta_shape))

    if return_ess:
        is_ess = _effective_sample_size_from_normalized_weights(weights)
        ess_metrics = {
            "is_ess": is_ess,
            "mala_acceptance_rate": mala_acceptance_rate,
        }
        return samples, key, ess_metrics

    return samples, key, {}


def _find_map(
    neg_log_post_fn: Callable[[Array], Array],
    flat_dim: int,
    n_steps: int = 500,
    lr: float = 0.01,
) -> Array:
    """Find MAP in flat unconstrained z-space via Adam gradient descent.

    neg_log_post_fn: (z_flat [flat_dim],) -> scalar
    Returns z_map of shape (flat_dim,).
    """
    return _find_map_from(neg_log_post_fn, jnp.zeros(flat_dim), n_steps, lr)


def _find_map_from(
    neg_log_post_fn: Callable[[Array], Array],
    z0: Array,
    n_steps: int = 500,
    lr: float = 0.01,
) -> Array:
    """MAP in flat z-space via Adam, starting from an explicit initial point.

    Identical to ``_find_map`` but lets the caller choose ``z0`` (used by the
    multi-start mixture sampler to seed independent restarts).
    """
    opt = optax.adam(lr)
    opt_state = opt.init(z0)

    def step(carry, _):
        z, state = carry
        grad = jax.grad(neg_log_post_fn)(z)
        updates, new_state = opt.update(grad, state)
        new_z = optax.apply_updates(z, updates)
        return (new_z, new_state), None

    (z_map, _), _ = jax.lax.scan(step, (z0, opt_state), None, length=n_steps)
    return z_map


def sample_posterior_laplace(
    key: Key,
    unbatched_log_lik: Callable[[Array], Array],
    theta_shape: Shape,
    theta_transform: Callable[[Array], Array],
    batch_shape: Shape,
    mala_steps: int = 5,
    mala_step_size: float = 0.1,
    return_ess: bool = False,
    laplace_reg: float = 1e-4,
    n_map_steps: int = 500,
    map_lr: float = 0.01,
) -> tuple[Array, Key, dict]:
    """IS+MALA posterior sampling using a Laplace approximation as the IS proposal.

    Finds the MAP in unconstrained z-space (where z ~ N(0,I) and
    theta = theta_transform(z)), builds a Gaussian proposal centred there,
    then applies corrected IS weights and MALA rejuvenation. Drop-in replacement
    for sample_posterior_is that dramatically improves ESS for concentrated
    posteriors.

    Args:
        unbatched_log_lik: theta -> scalar log-likelihood (may include prior).
        theta_shape: shape of a single theta sample.
        theta_transform: maps unconstrained z (same shape as theta) to theta.
        batch_shape: number of posterior samples to return, e.g. (n_post,).
        laplace_reg: regularisation added to the Hessian diagonal before Cholesky.
        n_map_steps: Adam steps for MAP optimisation.
        map_lr: Adam learning rate for MAP optimisation.
    """
    flat_dim = int(np.prod(theta_shape))
    n_samples = int(np.prod(batch_shape)) if batch_shape else 1

    def neg_log_post_flat(z_flat):
        theta = theta_transform(z_flat.reshape(theta_shape))
        # + 0.5 ||z||^2 is the negative log N(0,I) prior in z-space
        return -unbatched_log_lik(theta) + 0.5 * jnp.sum(z_flat**2)

    # Step 1: MAP in z-space
    z_map = _find_map(neg_log_post_flat, flat_dim, n_map_steps, map_lr)

    # Step 2: Hessian at MAP — this is the precision matrix of the Laplace approx
    H = jax.hessian(neg_log_post_flat)(z_map)  # [flat_dim, flat_dim]
    H_reg = H + laplace_reg * jnp.eye(flat_dim)
    L = jnp.linalg.cholesky(H_reg)  # lower triangular

    # Step 3: Sample z ~ N(z_map, H_reg^{-1}) via z = z_map + L^{-T} eps
    key, sk = jax.random.split(key)
    eps = jax.random.normal(sk, (n_samples, flat_dim))
    delta = jax.vmap(lambda e: jax.scipy.linalg.solve_triangular(L.T, e, lower=False))(
        eps
    )  # [n_samples, flat_dim]
    z_samples = z_map + delta  # [n_samples, flat_dim]
    theta_samples = jax.vmap(lambda z: theta_transform(z.reshape(theta_shape)))(
        z_samples
    )  # [n_samples, *theta_shape]

    # Step 4: IS weights — log p(y|theta) + log p(z) - log q_laplace(z), up to const
    log_liks = jax.vmap(unbatched_log_lik)(theta_samples)  # [n_samples]
    log_prior_z = -0.5 * jnp.sum(z_samples**2, axis=-1)  # [n_samples]
    log_q = -0.5 * jax.vmap(lambda d: d @ H_reg @ d)(
        delta
    )  # [n_samples] (proportional)
    log_weights = log_liks + log_prior_z - log_q

    # Step 5: Normalise and resample
    log_weights_normalized = log_weights - jax.scipy.special.logsumexp(log_weights)
    weights = jnp.exp(log_weights_normalized)

    key, sk = jax.random.split(key)
    indices = jax.random.choice(
        sk, n_samples, shape=(n_samples,), replace=True, p=weights
    )
    samples = theta_samples[indices]  # [n_samples, *theta_shape]

    # Step 6: MALA rejuvenation (same as sample_posterior_is)
    grad_log_lik = jax.grad(unbatched_log_lik)
    if return_ess:
        samples, key, mala_acceptance_rate = mala_move_batch(
            key,
            samples,
            grad_log_lik,
            unbatched_log_lik,
            step_size=mala_step_size,
            num_steps=mala_steps,
            return_diagnostics=True,
        )
    else:
        samples, key, _ = mala_move_batch(
            key,
            samples,
            grad_log_lik,
            unbatched_log_lik,
            step_size=mala_step_size,
            num_steps=mala_steps,
        )

    samples = jnp.reshape(samples, (*batch_shape, *theta_shape))

    if return_ess:
        is_ess = _effective_sample_size_from_normalized_weights(weights)
        ess_metrics = {
            "is_ess": is_ess,
            "mala_acceptance_rate": mala_acceptance_rate,
        }
        return samples, key, ess_metrics

    return samples, key, {}


def _mvn_logpdf_from_eigh(
    z: Array, mean: Array, eigvals: Array, eigvecs: Array
) -> Array:
    """log N(z; mean, prec^{-1}) where prec = eigvecs diag(eigvals) eigvecs^T.

    Works directly from the (floored) eigendecomposition of the precision so we
    never need a Cholesky of a possibly-indefinite matrix.
    """
    flat_dim = z.shape[-1]
    delta = z - mean
    # u = eigvecs^T delta, quadratic form = sum(eigvals * u^2)
    u = eigvecs.T @ delta
    quad = jnp.sum(eigvals * u**2)
    log_det_prec = jnp.sum(jnp.log(eigvals))
    return 0.5 * (log_det_prec - quad - flat_dim * jnp.log(2.0 * jnp.pi))


def sample_posterior_laplace_mixture(
    key: Key,
    unbatched_log_lik: Callable[[Array], Array],
    theta_shape: Shape,
    theta_transform: Callable[[Array], Array],
    batch_shape: Shape,
    mala_steps: int = 5,
    mala_step_size: float = 0.1,
    return_ess: bool = False,
    n_restarts: int = 8,
    laplace_reg: float = 1e-4,
    eig_floor: float = 1e-2,
    defensive_prior_weight: float = 0.1,
    defensive_prior_scale: float = 1.0,
    n_map_steps: int = 500,
    map_lr: float = 0.01,
    **_ignored,
) -> tuple[Array, Key, dict]:
    """Multi-start (mixture-of-Laplace) IS + MALA posterior sampler.

    Runs ``n_restarts`` independent MAP optimisations from prior-sampled starts,
    builds a robust Gaussian around each mode using an *eigenvalue-floored*
    precision (so an indefinite Hessian at a saddle/non-min can never produce a
    NaN Cholesky), and proposes from the equal-weight mixture of those Gaussians
    plus a broad defensive prior component. Proper mixture-IS weights are applied
    and MALA rejuvenates the resampled particles.

    This targets the same z-space distribution as ``sample_posterior_laplace``
    (``log pi(z) = unbatched_log_lik(theta_transform(z)) - 0.5||z||^2``), so it is
    a drop-in replacement that additionally copes with the multimodal /
    non-convex posteriors (e.g. LocationFinding) that break a single Laplace fit.
    """
    flat_dim = int(np.prod(theta_shape))
    n_samples = int(np.prod(batch_shape)) if batch_shape else 1

    def log_lik_z(z_flat):
        return unbatched_log_lik(theta_transform(z_flat.reshape(theta_shape)))

    def neg_log_post_flat(z_flat):
        return -log_lik_z(z_flat) + 0.5 * jnp.sum(z_flat**2)

    def log_post_flat(z_flat):
        return log_lik_z(z_flat) - 0.5 * jnp.sum(z_flat**2)

    # Step 1: independent MAP restarts from prior-sampled inits.
    key, init_key = jax.random.split(key)
    z_inits = jax.random.normal(init_key, (n_restarts, flat_dim))
    z_maps = jax.vmap(
        lambda z0: _find_map_from(neg_log_post_flat, z0, n_map_steps, map_lr)
    )(z_inits)  # [R, flat_dim]

    # Step 2: robust local Gaussians via eigenvalue-floored precision.
    def _floored_eigh(z_map):
        H = jax.hessian(neg_log_post_flat)(z_map) + laplace_reg * jnp.eye(flat_dim)
        evals, evecs = jnp.linalg.eigh(H)  # H symmetric
        evals = jnp.maximum(evals, eig_floor)  # precision eigenvalues > 0
        return evals, evecs

    comp_eigvals, comp_eigvecs = jax.vmap(_floored_eigh)(z_maps)  # [R,d], [R,d,d]

    # Mixture log-weights: (1 - dpw) split over R Laplace components, dpw on prior.
    use_defensive = defensive_prior_weight > 0.0
    laplace_logw = jnp.log((1.0 - defensive_prior_weight) / n_restarts) * jnp.ones(
        n_restarts
    )

    def log_q_components(z_flat):
        """Per-component log density (Laplace comps, then optional prior comp)."""
        lap = jax.vmap(lambda m, ev, evec: _mvn_logpdf_from_eigh(z_flat, m, ev, evec))(
            z_maps, comp_eigvals, comp_eigvecs
        )  # [R]
        comp_logp = laplace_logw + lap
        if use_defensive:
            prior_logp = jnp.log(defensive_prior_weight) + (
                -0.5 * jnp.sum(z_flat**2) / defensive_prior_scale**2
                - flat_dim * jnp.log(defensive_prior_scale)
                - 0.5 * flat_dim * jnp.log(2.0 * jnp.pi)
            )
            comp_logp = jnp.concatenate([comp_logp, prior_logp[None]])
        return comp_logp

    def log_q(z_flat):
        return jax.scipy.special.logsumexp(log_q_components(z_flat))

    # Step 3: sample component indices, then draw from the chosen component.
    key, comp_key, eps_key = jax.random.split(key, 3)
    if use_defensive:
        mix_probs = jnp.concatenate(
            [
                (1.0 - defensive_prior_weight) / n_restarts * jnp.ones(n_restarts),
                jnp.array([defensive_prior_weight]),
            ]
        )
    else:
        mix_probs = jnp.ones(n_restarts) / n_restarts
    comp_idx = jax.random.choice(
        comp_key, mix_probs.shape[0], shape=(n_samples,), p=mix_probs
    )
    eps = jax.random.normal(eps_key, (n_samples, flat_dim))

    def _draw(idx, e):
        # Laplace component: z = mean + evecs diag(1/sqrt(eigvals)) e
        def laplace_draw(_):
            i = jnp.minimum(idx, n_restarts - 1)
            mean = z_maps[i]
            ev = comp_eigvals[i]
            evec = comp_eigvecs[i]
            return mean + evec @ (e / jnp.sqrt(ev))

        def prior_draw(_):
            return defensive_prior_scale * e

        if use_defensive:
            return jax.lax.cond(
                idx < n_restarts, laplace_draw, prior_draw, operand=None
            )
        return laplace_draw(None)

    z_samples = jax.vmap(_draw)(comp_idx, eps)  # [n_samples, flat_dim]

    # Step 4: mixture-IS weights and resample.
    log_target = jax.vmap(log_post_flat)(z_samples)
    log_proposal = jax.vmap(log_q)(z_samples)
    log_weights = log_target - log_proposal
    log_weights_norm = log_weights - jax.scipy.special.logsumexp(log_weights)
    weights = jnp.exp(log_weights_norm)

    key, sk = jax.random.split(key)
    indices = jax.random.choice(
        sk, n_samples, shape=(n_samples,), replace=True, p=weights
    )
    z_resampled = z_samples[indices]

    # Step 5: MALA rejuvenation against the full z-space target.
    grad_log_post = jax.grad(log_post_flat)
    if return_ess:
        z_final, key, mala_acceptance_rate = mala_move_batch(
            key,
            z_resampled,
            grad_log_post,
            log_post_flat,
            step_size=mala_step_size,
            num_steps=mala_steps,
            return_diagnostics=True,
        )
    else:
        z_final, key, _ = mala_move_batch(
            key,
            z_resampled,
            grad_log_post,
            log_post_flat,
            step_size=mala_step_size,
            num_steps=mala_steps,
        )

    theta_samples = jax.vmap(lambda z: theta_transform(z.reshape(theta_shape)))(z_final)
    samples = jnp.reshape(theta_samples, (*batch_shape, *theta_shape))

    if return_ess:
        is_ess = _effective_sample_size_from_normalized_weights(weights)
        ess_metrics = {
            "is_ess": is_ess,
            "mala_acceptance_rate": mala_acceptance_rate,
        }
        return samples, key, ess_metrics

    return samples, key, {}


def sample_posterior_smc(
    key: Key,
    unbatched_log_lik: Callable[[Array], Array],
    theta_shape: Shape,
    theta_transform: Callable[[Array], Array],
    batch_shape: Shape,
    mala_steps: int = 10,
    mala_step_size: float = 0.1,
    return_ess: bool = False,
    n_temperatures: int = 64,
    temp_power: float = 2.0,
    adapt_step_size: bool = True,
    target_accept: float = 0.574,
    step_adapt_rate: float = 0.4,
    final_mala_steps: int = 25,
    **_ignored,
) -> tuple[Array, Key, dict]:
    """Tempered SMC (annealed IS) posterior sampler in z-space.

    Particles start as exact draws from the N(0, I) prior (beta = 0) and are
    annealed to the posterior (beta = 1) along the geometric path

        log pi_beta(z) = -0.5||z||^2 + beta * unbatched_log_lik(theta_transform(z)),

    which interpolates the prior and the same target as
    ``sample_posterior_laplace`` / ``sample_posterior_laplace_mixture``. At each
    temperature we reweight by the incremental likelihood, multinomially
    resample, and apply MALA moves (with optional acceptance-targeting step-size
    adaptation) at the current temperature. There is no Hessian/Cholesky, so the
    method is robust to the non-convex, multimodal posteriors that break a single
    Laplace fit, and it transitions smoothly between the broad (random-design)
    and concentrated (optimal-policy) regimes.

    The beta ladder is ``(i / n_temperatures) ** temp_power`` for
    i = 0..n_temperatures (temp_power > 1 places more steps at low beta, where the
    weight degeneracy of a concentrated posterior is worst).

    ``final_mala_steps`` (> 0) appends extra MALA moves against the full beta=1
    target after annealing, using the step size adapted during the run. This is
    a cheap way to remove residual equilibration bias on tightly-concentrated
    posteriors (e.g. the optimal-policy regime) without lengthening the ladder.
    """
    flat_dim = int(np.prod(theta_shape))
    n_samples = int(np.prod(batch_shape)) if batch_shape else 1

    def log_lik_z(z_flat):
        return unbatched_log_lik(theta_transform(z_flat.reshape(theta_shape)))

    batched_log_lik_z = jax.vmap(log_lik_z)

    grid = jnp.arange(n_temperatures + 1, dtype=jnp.float32) / n_temperatures
    betas = grid**temp_power  # betas[0] = 0, betas[-1] = 1
    beta_pairs = jnp.stack([betas[:-1], betas[1:]], axis=1)  # [n_temperatures, 2]

    key, init_key = jax.random.split(key)
    z = jax.random.normal(init_key, (n_samples, flat_dim))  # exact draw at beta = 0

    def temp_step(carry, beta_pair):
        z, key, step_size, log_evidence, min_ess, last_accept = carry
        beta_prev, beta = beta_pair[0], beta_pair[1]
        dbeta = beta - beta_prev

        # Incremental importance weights for beta_prev -> beta.
        llz = batched_log_lik_z(z)  # [n_samples]
        log_w = dbeta * llz
        log_w_sum = jax.scipy.special.logsumexp(log_w)
        log_w_norm = log_w - log_w_sum
        weights = jnp.exp(log_w_norm)
        ess = 1.0 / jnp.sum(weights**2)
        min_ess = jnp.minimum(min_ess, ess)
        log_evidence = log_evidence + log_w_sum - jnp.log(n_samples)

        # Multinomial resample (resample-move SMC).
        key, rk = jax.random.split(key)
        idx = jax.random.choice(rk, n_samples, shape=(n_samples,), p=weights)
        z = z[idx]

        # MALA move at temperature beta.
        def log_post_beta(z_flat):
            return -0.5 * jnp.sum(z_flat**2) + beta * log_lik_z(z_flat)

        grad_log_post_beta = jax.grad(log_post_beta)
        z, key, accept = mala_move_batch(
            key,
            z,
            grad_log_post_beta,
            log_post_beta,
            step_size=step_size,
            num_steps=mala_steps,
            return_diagnostics=True,
        )

        # Acceptance-targeting multiplicative step-size adaptation.
        if adapt_step_size:
            step_size = step_size * jnp.exp(step_adapt_rate * (accept - target_accept))
            step_size = jnp.clip(step_size, 1e-7, 10.0)

        return (z, key, step_size, log_evidence, min_ess, accept), None

    init_carry = (
        z,
        key,
        jnp.asarray(mala_step_size, dtype=z.dtype),
        jnp.asarray(0.0, dtype=z.dtype),
        jnp.asarray(float(n_samples), dtype=z.dtype),
        jnp.asarray(0.0, dtype=z.dtype),
    )
    (z, key, step_size, log_evidence, min_ess, last_accept), _ = jax.lax.scan(
        temp_step, init_carry, beta_pairs
    )

    # Optional extra polish at the full target (beta = 1) to remove residual
    # equilibration bias on tightly-concentrated posteriors.
    if final_mala_steps > 0:

        def log_post_final(z_flat):
            return -0.5 * jnp.sum(z_flat**2) + log_lik_z(z_flat)

        z, key, last_accept = mala_move_batch(
            key,
            z,
            jax.grad(log_post_final),
            log_post_final,
            step_size=step_size,
            num_steps=final_mala_steps,
            return_diagnostics=True,
        )

    theta_samples = jax.vmap(lambda zf: theta_transform(zf.reshape(theta_shape)))(z)
    samples = jnp.reshape(theta_samples, (*batch_shape, *theta_shape))

    if return_ess:
        ess_metrics = {
            # Worst per-temperature ESS: the SMC bottleneck. Reported as "is_ess"
            # so existing diagnostics (which divide by n_post) keep working.
            "is_ess": min_ess,
            "mala_acceptance_rate": last_accept,
            "smc_log_evidence": log_evidence,
            "smc_final_step_size": step_size,
        }
        return samples, key, ess_metrics

    return samples, key, {}
