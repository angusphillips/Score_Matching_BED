"""Trainer for variational approximation of joint marginal p(y, xi).

All distributions are Equinox-based (via flowjax), providing a unified interface.
Uses the flowjax/paramax pattern for partitioning trainable parameters.
"""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import paramax
from jaxtyping import Array, PyTree
from jaxtyping import PRNGKeyArray as Key

from policy_learning.bed_models import BEDModel
from policy_learning.trainers.callbacks_base import BaseCallback
from policy_learning.variational.distributions import VariationalDistribution

# Type alias for score functions
JointScoreFn = Callable[[Array, Array, dict | None], tuple[Array, Array]]
ScoreFn = Callable[[Array, Array, dict | None], Array]


def get_forward_kl_loss_fn(
    bed_model: BEDModel,
) -> Callable:
    """Create forward KL loss function for variational training.

    Forward KL: -E_{p(y,xi)}[log q(y, xi)]

    This corresponds to maximum likelihood estimation of q.

    Args:
        bed_model: BED model for standardization constants

    Returns:
        Loss function with signature (params, static, batch, key) -> loss
    """

    def forward_kl_loss(
        params: PyTree,
        static: PyTree,
        batch: tuple[Array, Array, Array, Any],
        key: Key,
    ) -> Array:
        """Forward KL: -E_p[log q(y, xi)]"""
        y, xi, theta, aux_data = batch

        # Reconstruct distribution from params and static
        dist = paramax.unwrap(eqx.combine(params, static))

        # Standardize
        y_scaled = (y - bed_model.y_mean) / bed_model.y_std
        xi_scaled = (xi - bed_model.xi_mean) / bed_model.xi_std

        # Compute log prob
        log_q = dist.log_prob(
            y_scaled, xi_scaled
        )  # ignoring the jacobian factor because it doesn't depend on the variational family parameters.
        loss = -jnp.mean(log_q)

        return loss

    return forward_kl_loss


def get_eig_var_marg_loss_fn(
    bed_model: BEDModel,
) -> Callable:
    """
    Get the variational upper bound to the EIG using marginal approximation from variational paper Foster 2019.

    Args:
        bed_model: BED model for standardization constants

    Returns:
        Loss function with signature (params, static, batch, key) -> loss
    """

    def var_loss(
        params: PyTree,
        static: PyTree,
        batch: tuple[Array, Array, Array, Any],
        key: Key,
    ) -> Array:
        """L_marg(phi) = E_p(y, theta | d) p(d) {log p(y|theta, d) / q_phi(y|d) }
        We model q_phi(y|d) = tilde q_phi(y, d) q(d) to avoid the difficulty of designing conditional normalising flows.
        """
        y, xi, theta, aux_data = batch

        # Reconstruct distribution from params and static
        dist = paramax.unwrap(eqx.combine(params, static))

        # Standardize
        y_scaled = (y - bed_model.y_mean) / bed_model.y_std
        xi_scaled = (xi - bed_model.xi_mean) / bed_model.xi_std

        # Compute log prob
        log_q = dist.log_prob(y_scaled, xi_scaled)
        log_p = bed_model.data_log_lik(y, xi, theta, aux_data)
        log_pd = bed_model.marginal_design_log_lik(xi)

        loss = jnp.mean(log_p + log_pd - log_q)

        return loss

    return var_loss


@eqx.filter_jit
def train_step(
    params: PyTree,
    static: PyTree,
    batch: tuple[Array, Array, Array, Any],
    key: Key,
    optimizer: optax.GradientTransformation,
    opt_state: PyTree,
    loss_fn: Callable,
    grad_clip_norm: float | None = None,
) -> tuple[PyTree, PyTree, Array, dict[str, Array]]:
    """Single training step.

    Following the flowjax pattern where params/static are partitioned.

    Args:
        params: Trainable parameters (from eqx.partition)
        static: Static components (from eqx.partition)
        batch: (y, xi, theta, aux_data) tuple
        key: Random key
        optimizer: Optax optimizer
        opt_state: Optimizer state
        loss_fn: Loss function

    Returns:
        Updated params, optimizer state, loss value, and per-step metrics
    """
    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(
        params,
        static,
        batch,
        key,
    )

    grad_norm = optax.global_norm(grads)
    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    update_norm = optax.global_norm(updates)
    params = eqx.apply_updates(params, updates)
    param_norm_after = optax.global_norm(params)

    if grad_clip_norm is None:
        clipping_applied = jnp.asarray(0.0, dtype=grad_norm.dtype)
    else:
        clipping_applied = jnp.asarray(
            grad_norm > jnp.asarray(grad_clip_norm, dtype=grad_norm.dtype),
            dtype=grad_norm.dtype,
        )

    step_metrics = {
        "loss_total": loss_val,
        "grad_norm_preclip": grad_norm,
        "param_norm": param_norm_after,
        "update_norm": update_norm,
        "clip_applied": clipping_applied,
    }

    return params, opt_state, loss_val, step_metrics


class VariationalTrainer:
    """Trainer for variational approximation of joint marginal p(y, xi).

    All distributions use the unified VariationalDistribution interface
    built on flowjax/equinox. Uses the flowjax pattern of partitioning
    params and static for proper gradient computation.

    Args:
        variational_dist: VariationalDistribution (wraps flowjax distributions)
        target_model: BED model providing samples from p(y, xi)
        optimizer: Optax optimizer
        key: JAX random key
        callbacks: List of callbacks for logging, checkpointing, etc.
        loss_type: Type of variational loss ("forward_kl" for MLE)
    """

    def __init__(
        self,
        variational_dist: VariationalDistribution,
        target_model: BEDModel,
        optimizer: optax.GradientTransformation,
        lr_schedule: optax.Schedule | None,
        key: Key,
        callbacks: list[BaseCallback] | None = None,
        loss_type: str = "max_lik",
        grad_clip_norm: float | None = None,
        design_sampler_type: str | None = None,
        design_sampler_kwargs: dict[str, Any] | None = None,
    ):
        self.target = target_model
        self.optimizer = optimizer
        self.callbacks = callbacks or []
        self.key = key
        self.loss_type = loss_type
        self.lr_schedule = lr_schedule
        self.grad_clip_norm = grad_clip_norm
        # Training design distribution (mirrors ScoreTrainer): None -> the model's
        # default fast_sample; otherwise designs are drawn from the named sampler so
        # the variational approximation can be fit on a chosen design distribution.
        self.design_sampler_type = design_sampler_type
        self.design_sampler_kwargs = (
            {} if design_sampler_kwargs is None else dict(design_sampler_kwargs)
        )

        # Store dimensions from the distribution
        self.dim_y = variational_dist.dim_y
        self.dim_xi = variational_dist.dim_xi
        self.T = variational_dist.T
        self.total_dim = variational_dist.total_dim

        # Partition into trainable params and static components
        # Following flowjax pattern: NonTrainable nodes are treated as leaves
        self.params, self.static = eqx.partition(
            variational_dist,
            eqx.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
        )

        # Initialize optimizer state with params (not full dist)
        self.opt_state = self.optimizer.init(self.params)

        # Create loss function
        if loss_type == "max_lik" or loss_type == "forward_kl":
            self.loss_fn = get_forward_kl_loss_fn(self.target)
        elif loss_type == "var_marg":
            self.loss_fn = get_eig_var_marg_loss_fn(self.target)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Training tracking
        self._last_loss = 0.0
        self._cum_loss_since_last_log = 0.0
        self._cum_loss_since_start = 0.0
        self.last_step_metrics: dict[str, Any] = {
            "loss_total": 0.0,
            "grad_norm_preclip": 0.0,
            "param_norm": 0.0,
            "update_norm": 0.0,
            "clip_applied": 0.0,
        }
        self.step = 0

    @property
    def dist(self) -> VariationalDistribution:
        """Reconstruct the full distribution from params and static."""
        return paramax.unwrap(eqx.combine(self.params, self.static))

    @property
    def training_state(self) -> Any:
        """Return a state-like object for callback compatibility."""

        class _TrainingState:
            pass

        state = _TrainingState()
        state.params = self.params  # type: ignore
        state.last_loss = self._last_loss  # type: ignore
        state.cum_loss_since_last_log = self._cum_loss_since_last_log  # type: ignore
        state.cum_loss_since_start = self._cum_loss_since_start  # type: ignore
        return state

    @training_state.setter
    def training_state(self, value: Any):
        """Set training state (for callback compatibility)."""
        if hasattr(value, "cum_loss_since_last_log"):
            self._cum_loss_since_last_log = value.cum_loss_since_last_log

    def _trigger_event(self, event_name: str):
        """Trigger callback events."""
        for cb in self.callbacks:
            func = getattr(cb, event_name, None)
            if func:
                func(self)

    def _sample_target(self, key: Key, batch_shape: tuple[int, ...]):
        """Draw (y, xi, theta, aux, key) under the configured design distribution.

        Mirrors ScoreTrainer._sample_target: with no design_sampler_type the model's
        default fast_sample is used; otherwise designs come from the named sampler so
        the variational fit can target a chosen design distribution.
        """
        if self.design_sampler_type is None:
            return self.target.fast_sample(key, batch_shape)
        try:
            return self.target.fast_sample(
                key,
                batch_shape,
                sampler_type=self.design_sampler_type,
                sampler_kwargs=self.design_sampler_kwargs,
            )
        except TypeError as exc:
            msg = str(exc)
            if "unexpected keyword argument" in msg or "positional argument" in msg:
                raise TypeError(
                    "Target model does not support sampler_type/sampler_kwargs in "
                    "fast_sample"
                ) from exc
            raise

    def train(self, num_iters: int, batch_size: int):
        """Train the variational distribution.

        Args:
            num_iters: Number of training iterations
            batch_size: Batch size for training
        """
        self._num_iters = num_iters
        self._trigger_event("on_train_start")
        self.step = 0

        for _ in range(num_iters):
            self._trigger_event("on_batch_start")

            # Sample from target distribution (under the configured design distribution).
            y, xi, theta, aux_data, self.key = self._sample_target(
                self.key, (batch_size,)
            )
            batch = (y, xi, theta, aux_data)

            # Training step
            self.key, subkey = jax.random.split(self.key)
            self.params, self.opt_state, loss, self.last_step_metrics = train_step(
                self.params,
                self.static,
                batch,
                subkey,
                self.optimizer,
                self.opt_state,
                self.loss_fn,
                self.grad_clip_norm,
            )

            self._last_loss = float(loss)
            self._cum_loss_since_last_log += self._last_loss
            self._cum_loss_since_start += self._last_loss

            self.step += 1
            self._trigger_event("on_batch_end")

        self._trigger_event("on_train_end")

    def get_joint_score_fn(self) -> JointScoreFn:
        """Get the learned score function ∇ log p(y|theta, xi).

        Returns a function that computes the conditional score at given (y, xi) pairs.
        Since the variational distribution q(y, xi) approximates p(y|theta, xi) * p(xi),
        we subtract the marginal design score to get the conditional:
        ∇ log p(y|theta, xi) = ∇ log q(y, xi) - [0, ∇ log p(xi)]
        """
        dist = self.dist  # Use EMA version

        def score_function(
            y: Array, xi: Array, aux_data: dict | None = None
        ) -> tuple[Array, Array]:
            # Standardize inputs
            y_scaled = (y - self.target.y_mean) / self.target.y_std
            xi_scaled = (xi - self.target.xi_mean) / self.target.xi_std

            # Compute score via the distribution's score method
            score_y_scaled, score_xi_scaled = dist.score(y_scaled, xi_scaled)

            # Scale back (chain rule for standardization)
            score_y = score_y_scaled / self.target.y_std
            score_xi = score_xi_scaled / self.target.xi_std

            # Subtract marginal design score to get conditional score
            # ∇ log p(y|theta, xi) = ∇ log q(y, xi) - ∇ log p(xi)
            score_xi = score_xi - self.target.marginal_design_score(xi)

            return score_y, score_xi

        return score_function

    def get_score_fn(self) -> ScoreFn:
        """Get flattened score function [score_y, score_xi]."""
        joint_score_fn = self.get_joint_score_fn()

        def score_fn(y: Array, xi: Array, aux_data: dict | None = None):
            score_y, score_xi = joint_score_fn(y, xi, aux_data)
            return jnp.concatenate([score_y, score_xi], axis=-1)

        return score_fn

    def get_unbatched_score_fns(self) -> tuple[ScoreFn, ScoreFn]:
        """Get separate score functions for y and xi.

        Returns functions matching the API expected by EIG estimators.
        """
        joint_score_fn = self.get_joint_score_fn()

        def data_score_fn(y: Array, xi: Array, aux_data: dict | None = None) -> Array:
            score_y, _ = joint_score_fn(y, xi, aux_data)
            return score_y[0]

        def design_score_fn(y: Array, xi: Array, aux_data: dict | None = None) -> Array:
            _, score_xi = joint_score_fn(y, xi, aux_data)
            return score_xi[0]

        return data_score_fn, design_score_fn

    def get_data_score_fn(self) -> ScoreFn:
        joint_score_fn = self.get_joint_score_fn()

        def data_score_fn(y: Array, xi: Array, aux_data: dict | None = None) -> Array:
            score_y, _ = joint_score_fn(y, xi, aux_data)
            return score_y

        return data_score_fn

    def get_design_score_fn(self) -> ScoreFn:
        joint_score_fn = self.get_joint_score_fn()

        def design_score_fn(y: Array, xi: Array, aux_data: dict | None = None) -> Array:
            _, score_xi = joint_score_fn(y, xi, aux_data)
            return score_xi

        return design_score_fn

    def get_log_prob_fn(self) -> Callable[[Array, Array], Array]:
        """Get log probability function log q(y, xi)."""
        dist = self.dist

        def log_prob_fn(y: Array, xi: Array) -> Array:
            y_scaled = (y - self.target.y_mean) / self.target.y_std
            xi_scaled = (xi - self.target.xi_mean) / self.target.xi_std
            return dist.log_prob(y_scaled, xi_scaled)

        return log_prob_fn

    def sample(self, key: Key, batch_shape: tuple[int, ...]) -> tuple[Array, Array]:
        """Sample from the learned variational distribution.

        Returns samples in the original (unstandardized) space.
        """
        y_scaled, xi_scaled = self.dist.sample(key, batch_shape)

        # Unstandardize
        y = y_scaled * self.target.y_std + self.target.y_mean
        xi = xi_scaled * self.target.xi_std + self.target.xi_mean

        return y, xi
