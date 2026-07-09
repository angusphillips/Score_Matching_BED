"""Pluggable objectives for variational posterior (var_post) training of the
policy and conditional posterior."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array, PyTree
from jaxtyping import PRNGKeyArray as Key

from policy_learning.bed_models import BEDModel


def BA_bound(
    policy_params: PyTree,
    posterior_params: PyTree,
    posterior_static_vars: PyTree,
    policy_net: nn.Module,
    posterior_net: nn.Module,
    bed_model: BEDModel,
    key: Key,
    batch_cfg: dict[str, Any],
) -> tuple[Array, dict[str, Array]]:
    """Barber-Agakov lower bound on the mutual information I(theta; y).

    Samples a rollout from the current policy, encodes it in the posterior, and
    maximises E_{p(theta)p(y|theta)}[log q(z | y, x)] + H[p(z)], where
    z = theta_transform_inv(theta) is the (a priori standard-normal) latent. The
    bound is evaluated in z-space: I(theta; y) = I(z; y) because z <-> theta is a
    deterministic bijection, so no Jacobian correction is required and the
    Gaussian-mixture posterior models a Gaussian-friendly target.

    Args:
        policy_params: Policy network parameters
        posterior_params: Posterior network parameters
        policy_net: Policy network module
        posterior_net: Posterior network module
        bed_model: BED model with target distribution
        key: Random key
        batch_cfg: Batch configuration dict (e.g., outer_batch_size)

    Returns:
        (loss_scalar, metrics_dict) where loss is negative likelihood (for minimization)
    """
    batch_size = batch_cfg.get("batch_size", 256)

    posterior_only_random_policy = bool(
        batch_cfg.get("posterior_only_random_policy", False)
    )
    random_policy_std = float(batch_cfg.get("posterior_only_random_policy_std", 1.0))
    # Posterior-only training design distribution: if a design sampler is named, draw
    # designs from it (e.g. the benchmark's configurable training distribution); else
    # fall back to the legacy isotropic-Gaussian design process.
    design_sampler_type = batch_cfg.get("design_sampler_type", None)
    design_sampler_kwargs = batch_cfg.get("design_sampler_kwargs", None) or {}

    # Sample rollout from policy or from a fixed (policy-free) design process
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)

    theta_outer, subkey1 = bed_model.sample_prior(subkey1, (batch_size,))
    if posterior_only_random_policy:
        if design_sampler_type is not None:
            x, _, subkey3 = bed_model.sample_designs(
                subkey3, (batch_size,), theta_outer,
                sampler_type=design_sampler_type, sampler_kwargs=design_sampler_kwargs,
            )
        else:
            policy_epsilon, subkey3 = bed_model.sample_policy_epsilon(
                subkey3, (batch_size,)
            )
            x = random_policy_std * policy_epsilon
        y, _, _ = bed_model.sample_data(subkey2, x, theta_outer)
    else:
        epsilon, subkey2 = bed_model.sample_epsilon(subkey2, (batch_size,))
        policy_epsilon, subkey3 = bed_model.sample_policy_epsilon(subkey3, (batch_size,))

        # Generate rollout
        y, x, _ = bed_model.reparam_sample(
            params=policy_params,
            policy_net=policy_net,
            theta=theta_outer,
            epsilon=epsilon,
            policy_epsilon=policy_epsilon,
        )

    # Bind posterior model with params for method calls
    posterior_bound = posterior_net.bind(
        {
            "params": posterior_params,
            **posterior_static_vars,
        }
    )

    # Work in the unconstrained latent space z = theta_transform_inv(theta).
    # For LocationFinding this is the identity; for the dynamical systems it
    # maps the lognormal theta back to its standard-normal latent.
    z_outer = bed_model.theta_transform_inv(theta_outer)

    # Evaluate E_{p(z)p(y|z)}[log q(z | y, x)]
    posterior_log_prob = posterior_bound.log_prob(z_outer, y, x)

    # H[p(z)] for the standard-normal latent, scaled by theta_dim.
    prior_entropy = bed_model.latent_prior_entropy

    loss = jnp.mean(posterior_log_prob) + prior_entropy

    metrics = {
        "ba_bound": loss,
    }

    return loss, metrics


def get_objective_fn(
    objective_name: str,
) -> Callable:
    """Get objective function by name.

    Args:
        objective_name: Name of objective ("dummy_posterior_likelihood", ...)

    Returns:
        Objective function with standard signature
    """
    if objective_name == "ba_bound":
        return BA_bound
    else:
        raise ValueError(
            f"Unknown objective: {objective_name}. "
            f"Supported: ['dummy_posterior_likelihood']"
        )


def make_var_post_value_and_grad_fn(
    objective_fn: Callable,
) -> Callable:
    """Create a function that computes objective, metrics, and both gradients.

    Args:
        objective_fn: Objective function with standard signature

    Returns:
        Function: (policy_params, posterior_params, policy_net, posterior_net, bed_model, key, batch_cfg)
                  -> ((loss, metrics), (grad_policy, grad_posterior), new_key)
    """

    def value_and_grad(
        policy_params: PyTree,
        posterior_params: PyTree,
        posterior_static_vars: PyTree,
        policy_net: nn.Module,
        posterior_net: nn.Module,
        bed_model: BEDModel,
        key: Key,
        batch_cfg: dict[str, Any],
    ) -> tuple[tuple[Array, dict], tuple[PyTree, PyTree], Key]:
        """Compute objective, metrics, and gradients."""
        # Split key for this objective call
        key, subkey = jax.random.split(key)

        def objective_fn_wrapped(policy_p, posterior_p):
            loss, metrics = objective_fn(
                policy_p,
                posterior_p,
                posterior_static_vars,
                policy_net,
                posterior_net,
                bed_model,
                subkey,
                batch_cfg,
            )
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(
            objective_fn_wrapped, argnums=(0, 1), has_aux=True
        )(policy_params, posterior_params)

        grad_policy, grad_posterior = grads

        return (loss, metrics), (grad_policy, grad_posterior), key

    return value_and_grad
