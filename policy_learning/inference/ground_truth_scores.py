"""Ground truth score estimation using Monte Carlo posterior sampling.

This module provides functions for computing high-quality score estimates
using posterior-based Monte Carlo methods. These are useful for evaluating
learned score networks against ground truth.
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from tqdm import trange

from policy_learning.bed_models import BEDModel
from policy_learning.inference.importance_sampler import (
    sample_posterior_is,
    sample_posterior_laplace,
    sample_posterior_laplace_mixture,
    sample_posterior_smc,
)

# Posterior samplers selectable via ``posterior_sampler`` (drop-in, jit+vmap).
# All except "prior_is" propose in z-space via theta_transform.
POSTERIOR_SAMPLERS = ("prior_is", "laplace", "laplace_mixture", "smc")


class MonteCarloPosteriorScore:
    """Stateful MC posterior score estimator compatible with score-network APIs.

    This class wraps posterior-based Monte Carlo score estimation and supports two
    call patterns:
    1. Keyless calls that internally advance a stored PRNG key.
    2. Explicit-key calls `(y, xi, aux_data, key)` for pure functional use.
    """

    def __init__(
        self,
        target_dist: BEDModel,
        key,
        n_post_samples: int = 10000,
        batch_size: int | None = None,
        mala_steps: int = 5,
        mala_step_size: float = 0.1,
        laplace_reg: float = 1e-4,
        n_map_steps: int = 500,
        posterior_sampler: str | None = None,
        sampler_kwargs: dict | None = None,
    ):
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be a positive integer or None")
        self.target_dist = target_dist
        self.key = key
        self.n_post_samples = n_post_samples
        self.batch_size = batch_size
        self.mala_steps = mala_steps
        self.mala_step_size = mala_step_size
        self._mc_score_fn = create_ground_truth_score_function(
            target_dist,
            laplace_reg=laplace_reg,
            n_map_steps=n_map_steps,
            posterior_sampler=posterior_sampler,
            sampler_kwargs=sampler_kwargs,
        )
        # in_axes for (key, y, xi, aux_data, n_post_samples, mala_steps,
        #              mala_step_size, return_variance, return_all, return_ess,
        #              noise_mult). Trailing seven are static / shared across the
        #              batch (noise_mult is one shared scalar per call).
        self._batched_score_fn = jax.vmap(
            self._score_single,
            in_axes=(0, 0, 0, 0, None, None, None, None, None, None, None),
        )

    @classmethod
    def from_config(
        cls,
        mc_cfg: dict | None,
        target_dist: BEDModel,
        key,
        *,
        n_post_samples: int | None = None,
        batch_size: int | None = None,
    ) -> "MonteCarloPosteriorScore":
        """Build from an ``mc_posterior`` config block (plain dict from the
        ``mc_posterior`` config group).

        The block holds the posterior-sampler settings shared by every MC
        consumer (mala_*, laplace_*, posterior_sampler, sampler_kwargs). The
        sample count is supplied per use case via the ``n_post_samples`` override
        (``train_n_post_samples`` / ``eval_n_post_samples`` in the run scripts);
        ``batch_size`` likewise overrides for purpose-specific estimators — e.g.
        ground-truth test sets pass ``batch_size=1`` to chunk per test point.
        """
        mc_cfg = dict(mc_cfg or {})
        raw_kwargs = mc_cfg.get("sampler_kwargs", None)
        return cls(
            target_dist=target_dist,
            key=key,
            n_post_samples=int(
                n_post_samples
                if n_post_samples is not None
                else mc_cfg.get("n_post_samples", 10000)
            ),
            batch_size=int(
                batch_size if batch_size is not None else mc_cfg.get("batch_size", 16)
            ),
            mala_steps=int(mc_cfg.get("mala_steps", 5)),
            mala_step_size=float(mc_cfg.get("mala_step_size", 0.1)),
            laplace_reg=float(mc_cfg.get("laplace_reg", 1e-4)),
            n_map_steps=int(mc_cfg.get("n_map_steps", 500)),
            posterior_sampler=mc_cfg.get("posterior_sampler", None),
            sampler_kwargs=(None if raw_kwargs is None else dict(raw_kwargs)),
        )

    def set_key(self, key):
        self.key = key

    def _next_key(self):
        next_key, self.key = jax.random.split(self.key)
        return next_key

    def _score_single(
        self,
        key,
        y,
        xi,
        aux_data: dict | None,
        n_post_samples: int,
        mala_steps: int,
        mala_step_size: float,
        return_variance: bool,
        return_all: bool,
        return_ess: bool,
        noise_mult=None,
    ):
        return self._mc_score_fn(
            key,
            y,
            xi,
            aux_data or {},
            n_post_samples=n_post_samples,
            mala_steps=mala_steps,
            mala_step_size=mala_step_size,
            return_variance=return_variance,
            return_all=return_all,
            return_ess=return_ess,
            noise_mult=noise_mult,
        )

    def __call__(
        self,
        y,
        xi,
        aux_data: dict | None = None,
        key=None,
        n_post_samples: int | None = None,
        mala_steps: int | None = None,
        mala_step_size: float | None = None,
        return_variance: bool = False,
        return_all: bool = False,
        return_ess: bool = False,
        noise_mult=None,
        desc: str | None = None,
        show_progress: bool = True,
    ):
        n_post_samples = (
            self.n_post_samples if n_post_samples is None else n_post_samples
        )
        mala_steps = self.mala_steps if mala_steps is None else mala_steps
        mala_step_size = (
            self.mala_step_size if mala_step_size is None else mala_step_size
        )

        if y.ndim == 2:
            score_key = self._next_key() if key is None else key
            return self._score_single(
                score_key,
                y,
                xi,
                aux_data,
                n_post_samples,
                mala_steps,
                mala_step_size,
                return_variance,
                return_all,
                return_ess,
                noise_mult,
            )

        batch_size = y.shape[0]

        if key is None:
            call_keys = jax.random.split(self.key, batch_size + 1)
            self.key = call_keys[0]
            score_keys = call_keys[1:]
        else:
            score_keys = jax.random.split(key, batch_size)

        aux_data = aux_data or {}

        if self.batch_size is None or batch_size <= self.batch_size:
            return self._batched_score_fn(
                score_keys,
                y,
                xi,
                aux_data,
                n_post_samples,
                mala_steps,
                mala_step_size,
                return_variance,
                return_all,
                return_ess,
                noise_mult,
            )

        chunk_outputs = []
        for start in trange(
            0, batch_size, self.batch_size, desc=desc, disable=not show_progress
        ):
            stop = min(start + self.batch_size, batch_size)
            y_chunk = y[start:stop]
            xi_chunk = xi[start:stop]
            aux_chunk = jax.tree_util.tree_map(
                lambda a, start=start, stop=stop: a[start:stop], aux_data
            )
            key_chunk = score_keys[start:stop]
            chunk_outputs.append(
                self._batched_score_fn(
                    key_chunk,
                    y_chunk,
                    xi_chunk,
                    aux_chunk,
                    n_post_samples,
                    mala_steps,
                    mala_step_size,
                    return_variance,
                    return_all,
                    return_ess,
                    noise_mult,
                )
            )

        return jax.tree_util.tree_map(
            lambda *parts: jnp.concatenate(parts, axis=0), *chunk_outputs
        )

    def get_score_fn(self):
        return lambda y, xi, aux_data=None, key=None: self(
            y, xi, aux_data=aux_data, key=key
        )

    def get_data_score_fn(self):
        return lambda y, xi, aux_data=None, key=None: self(
            y, xi, aux_data=aux_data, key=key
        )[..., : self.target_dist.d_y]

    def get_design_score_fn(self):
        return lambda y, xi, aux_data=None, key=None: self(
            y, xi, aux_data=aux_data, key=key
        )[..., self.target_dist.d_y :]

    def get_unbatched_score_fns(self):
        def data_score_fn(y, xi, aux_data=None, key=None):
            return self(y, xi, aux_data=aux_data, key=key)[0, :, : self.target_dist.d_y]

        def design_score_fn(y, xi, aux_data=None, key=None):
            return self(y, xi, aux_data=aux_data, key=key)[0, :, self.target_dist.d_y :]

        return data_score_fn, design_score_fn


def create_ground_truth_score_function(
    target_dist: BEDModel,
    prior_sampler_gt: Callable | None = None,
    laplace_reg: float = 1e-4,
    n_map_steps: int = 500,
    posterior_sampler: str | None = None,
    sampler_kwargs: dict | None = None,
):
    """
    Create Monte Carlo posterior-based score estimation function.

    Provides high-quality score estimates for evaluating learned score networks.
    Uses extensive sampling to approximate:
    ∇_(y,xi) log p(y|xi) = E_{p(θ|y,xi)}[∇_(y,xi) log p(y|xi,θ)]

    Args:
        target_dist: The BED model defining the target distribution
        aux_data: Optional auxiliary data for the model

        posterior_sampler: which posterior routine to use. One of
            ``"prior_is"`` (prior importance sampling + MALA), ``"laplace"``
            (single Laplace-approx proposal), ``"laplace_mixture"`` (multi-start
            mixture-of-Laplace, robust to multimodal/non-convex posteriors), or
            ``"smc"`` (tempered SMC / annealed IS). Defaults to ``"prior_is"``
            when ``None``.
        sampler_kwargs: extra keyword arguments forwarded to the chosen sampler
            (e.g. ``n_restarts`` for laplace_mixture, ``n_temperatures`` /
            ``temp_power`` for smc).

    Returns:
        mc_score_fn: Function for posterior-based MC score estimation.
            Signature: (y, xi, key, n_post_samples, return_variance) -> scores or (scores, variance)
    """
    if posterior_sampler is None:
        posterior_sampler = "prior_is"
    if posterior_sampler not in POSTERIOR_SAMPLERS:
        raise ValueError(
            f"Unknown posterior_sampler {posterior_sampler!r}. "
            f"Valid options: {', '.join(POSTERIOR_SAMPLERS)}"
        )
    sampler_kwargs = {} if sampler_kwargs is None else dict(sampler_kwargs)

    if prior_sampler_gt is None:

        def _prior_sampler_gt(key, batch_shape):
            key, subkey = jax.random.split(key)
            return target_dist.theta_transform(
                jax.random.normal(subkey, (*batch_shape, *target_dist.theta_shape))
            ), key

        prior_sampler_gt = _prior_sampler_gt

    def grad_log_lik(y, xi, theta, aux_data, noise_mult=None):
        # noise_mult (None at base noise) is a trace-time Python branch: when None
        # we call data_log_lik with the original signature so models that don't
        # implement noise annealing are unaffected; otherwise we pass the
        # lambda^2-scaled covariance through for NCSN ground-truth at noise lambda.
        def ll(yy, xx):
            if noise_mult is None:
                return target_dist.data_log_lik(yy, xx, theta, aux_data)
            return target_dist.data_log_lik(
                yy, xx, theta, aux_data, noise_mult=noise_mult
            )

        y_grad = jax.grad(lambda y_: ll(y_, xi))(y)
        xi_grad = jax.grad(lambda xi_: ll(y, xi_))(xi)
        return jnp.concatenate([y_grad, xi_grad], axis=-1)

    @partial(
        jax.jit,
        static_argnames=(
            "n_post_samples",
            "mala_steps",
            "mala_step_size",
            "return_variance",
            "return_ess",
            "return_all",
        ),
    )
    def mc_score_fn(
        key,
        y,
        xi,
        aux_data,
        n_post_samples=10000,
        mala_steps: int = 5,
        mala_step_size: float = 0.1,
        return_variance=False,
        return_all: bool = False,
        return_ess: bool = False,
        noise_mult=None,
    ):
        """Monte Carlo score estimation using posterior samples.

        Estimates ∇_(y,xi) log p(y|xi) by:
        1. Sampling θ ~ p(θ|y,xi) using importance sampling
        2. Computing E_{p(θ|y,xi)}[∇_(y,xi) log p(y|xi,θ)]

        Args:
            y: Observation data [T, d_y]
            xi: Design [T, d]
            key: JAX random key
            n_post_samples: Number of posterior samples to use
            return_variance: If True, also return variance of the estimate

        Returns:
            If return_variance=False: Score estimate [T, d_y + d]
            If return_variance=True: (score_estimate, variance) both [T, d_y + d]
        """

        @jax.jit
        def unbatched_log_lik(theta, y_cond=y, xi_cond=xi, aux_data_cond=aux_data):
            theta_broadcast = jnp.broadcast_to(
                jnp.expand_dims(theta, 0),
                (target_dist.T, *target_dist.theta_shape),
            )
            # Same trace-time branch as grad_log_lik so the posterior is sampled at
            # noise lambda when annealing, and is byte-identical at base noise.
            if noise_mult is None:
                return target_dist.log_lik(
                    y_cond, xi_cond, theta_broadcast, aux_data_cond or {}
                )
            return target_dist.log_lik(
                y_cond,
                xi_cond,
                theta_broadcast,
                aux_data_cond or {},
                noise_mult=noise_mult,
            )

        if posterior_sampler == "prior_is":
            posterior_samples, key, ess_metrics = sample_posterior_is(
                key,
                unbatched_log_lik,
                prior_sampler_gt,
                target_dist.theta_shape,
                (n_post_samples,),
                mala_steps=mala_steps,
                mala_step_size=mala_step_size,
                return_ess=return_ess,
            )
        elif posterior_sampler == "laplace":
            posterior_samples, key, ess_metrics = sample_posterior_laplace(
                key,
                unbatched_log_lik,
                theta_shape=target_dist.theta_shape,
                theta_transform=target_dist.theta_transform,
                batch_shape=(n_post_samples,),
                mala_steps=mala_steps,
                mala_step_size=mala_step_size,
                return_ess=return_ess,
                laplace_reg=laplace_reg,
                n_map_steps=n_map_steps,
                **sampler_kwargs,
            )
        elif posterior_sampler == "laplace_mixture":
            posterior_samples, key, ess_metrics = sample_posterior_laplace_mixture(
                key,
                unbatched_log_lik,
                theta_shape=target_dist.theta_shape,
                theta_transform=target_dist.theta_transform,
                batch_shape=(n_post_samples,),
                mala_steps=mala_steps,
                mala_step_size=mala_step_size,
                return_ess=return_ess,
                laplace_reg=laplace_reg,
                n_map_steps=n_map_steps,
                **sampler_kwargs,
            )
        else:  # "smc"
            posterior_samples, key, ess_metrics = sample_posterior_smc(
                key,
                unbatched_log_lik,
                theta_shape=target_dist.theta_shape,
                theta_transform=target_dist.theta_transform,
                batch_shape=(n_post_samples,),
                mala_steps=mala_steps,
                mala_step_size=mala_step_size,
                return_ess=return_ess,
                **sampler_kwargs,
            )

        # Explicitly insert T dimension at position 1
        posterior_samples = jnp.expand_dims(posterior_samples, axis=1)
        posterior_samples = jnp.broadcast_to(
            posterior_samples,
            (n_post_samples, target_dist.T, *posterior_samples.shape[2:]),
        )

        mc_samples = jax.vmap(
            lambda post_theta: grad_log_lik(
                y, xi, post_theta, aux_data or {}, noise_mult=noise_mult
            )
        )(posterior_samples)

        score_mean = jnp.mean(mc_samples, axis=0)

        if return_variance:
            flat_scores = jnp.reshape(mc_samples, (n_post_samples, -1))

            score_cov = jnp.cov(flat_scores, rowvar=False)
            if return_ess:
                return score_mean, score_cov, ess_metrics
            return score_mean, score_cov
        if return_all:
            if return_ess:
                return score_mean, posterior_samples, mc_samples, ess_metrics
            return score_mean, posterior_samples, mc_samples
        else:
            if return_ess:
                return score_mean, ess_metrics
            return score_mean

    return mc_score_fn


def generate_test_set_from_prior(
    key,
    target_dist: BEDModel,
    n_test: int = 100,
    gt_n_samples: int = 100000,
    n_proposal: int = 1000000,
    mala_steps: int = 5,
    mala_step_size: float = 0.1,
    laplace_reg: float = 1e-4,
    n_map_steps: int = 500,
    posterior_sampler: str | None = None,
    sampler_kwargs: dict | None = None,
):
    """
    Generate test set with ground truth score estimates using prior design sampling.

    This uses the standard prior-based design sampler (not a policy) and computes
    high-quality ground truth scores using extensive MC posterior sampling.

    Args:
        key: JAX random key
        target_dist: BED model
        n_test: Number of test points to generate
        gt_n_samples: Number of samples to use for ground truth score estimation
        mala_steps: Number of MALA refinement steps per posterior sample.
        mala_step_size: MALA step size for the posterior refinement.
        posterior_sampler: which posterior routine to use (see
            ``create_ground_truth_score_function``); defaults to ``"prior_is"``.
        laplace_reg: Diagonal regulariser added to the Laplace Hessian.
        n_map_steps: Optimiser steps to find the MAP used as the Laplace mean.

    Returns:
        y_test: Test observations [n_test, T, d_y]
        xi_test: Test designs [n_test, T, d]
        gt_scores: Ground truth scores via MC [n_test, T, d_y + d]
        gt_score_variance: Variance of MC score estimates [n_test, T, d_y + d]
        mean_variance_per_dim: Mean variance per dimension (quality metric)
        key: Updated random key
    """
    # Sample test data using prior
    y_test, xi_test, true_theta_test, aux_data_test, key = target_dist.fast_sample(
        key, (n_test,)
    )

    # Create ground truth score function. Threading laplace / mala-related kwargs
    # through ensures the gt-test-set MC matches whatever the policy training
    # uses (cf. cfg["mc_posterior"] in the run scripts).
    mc_score_fn = create_ground_truth_score_function(
        target_dist,
        laplace_reg=laplace_reg,
        n_map_steps=n_map_steps,
        posterior_sampler=posterior_sampler,
        sampler_kwargs=sampler_kwargs,
    )

    # Compute ground truth scores with variance
    keys = jax.random.split(key, n_test + 1)
    key = keys[0]

    # Loop over test points to avoid memory issues
    gt_scores_list = []
    gt_cov_list = []
    ess_metrics_list = []
    for i in trange(n_test):
        score, cov, ess_metrics = mc_score_fn(
            keys[i + 1],
            y_test[i],
            xi_test[i],
            {k: v[i] for k, v in aux_data_test.items()},
            n_post_samples=gt_n_samples,
            mala_steps=mala_steps,
            mala_step_size=mala_step_size,
            return_variance=True,
            return_ess=True,
        )
        gt_scores_list.append(score)
        gt_cov_list.append(cov)
        ess_metrics_list.append(ess_metrics)

    gt_scores = jnp.stack(gt_scores_list, axis=0)
    gt_covs = jnp.stack(gt_cov_list, axis=0)

    return (
        y_test,
        xi_test,
        aux_data_test,
        true_theta_test,
        gt_scores,
        gt_covs,
        ess_metrics_list,
        key,
    )
