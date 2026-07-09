import logging
from collections.abc import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from jaxtyping import Array
from jaxtyping import PRNGKeyArray as Key

from policy_learning.bed_models import BEDModel
from policy_learning.trainers.callbacks_base import BaseCallback
from policy_learning.utils.flax import create_train_state, get_train_step_gradfn


class PolicyTrainer:
    def __init__(
        self,
        policy_network: nn.Module,
        target_model: BEDModel,
        grad_fn: Callable,
        optimizer: optax.GradientTransformationExtraArgs,
        lr_schedule: optax.Schedule | None,
        key: Key,
        callbacks: list[BaseCallback],
        grad_clip_norm: float | None = None,
        design_sampler_type: str | None = None,
        design_sampler_kwargs: dict | None = None,
    ):
        self._num_iters = 0
        self.policy_network = policy_network
        self.target = target_model
        self.grad_clip_norm = grad_clip_norm
        # Design sampler used to train the score network. Recorded so that
        # diagnostic callbacks can draw reference batches from the same
        # distribution the score network saw, rather than the model's default
        # (uniform) sampler. See sample_reference_batch.
        self.design_sampler_type = design_sampler_type
        self.design_sampler_kwargs = (
            {} if design_sampler_kwargs is None else dict(design_sampler_kwargs)
        )
        self.train_step = get_train_step_gradfn(
            grad_fn=grad_fn,
            grad_clip_norm=self.grad_clip_norm,
        )
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.callbacks = callbacks
        self.key = key

        # Use the configured design sampler (not the model default "uniform") so
        # initialisation works under every policy_parameterisation: in
        # "soft_constraint" mode the bounded samplers (uniform etc.) hard-fail, so
        # we must draw the init batch from the same sampler the score net trains on.
        y_init, xi_init, _, _, self.key = self.target.sample(
            self.key,
            (10,),
            sampler_type=self.design_sampler_type,
            sampler_kwargs=self.design_sampler_kwargs,
        )
        policy_epsilon, self.key = self.target.sample_policy_epsilon(self.key, (10,))
        self.training_state, self.key = create_train_state(
            self.policy_network,
            self.key,
            self.optimizer,
            y=y_init,
            x=xi_init,
            t=jnp.ones((10,)),
            z=policy_epsilon,
        )

    def _trigger_event(self, event_name: str):
        for cb in self.callbacks:
            func = getattr(cb, event_name, None)
            if func:
                func(self)

    def train(self, num_iters: int, reset_step: bool = True):
        self._num_iters = num_iters
        if reset_step:
            self.step = 0
        self._trigger_event("on_train_start")
        for _ in range(0, num_iters + 1):
            self._trigger_event("on_batch_start")
            self.last_step_training_state = self.training_state
            self.training_state, self.last_step_metrics, self.last_step_grads = (
                self.train_step(self.training_state)
            )
            self._trigger_event("on_batch_end")
            self.step += 1

        self._trigger_event("on_train_end")

    def update_grad_fn(self, grad_fn: Callable) -> None:
        """Replace the gradient function, e.g. after score network finetuning."""
        self.train_step = get_train_step_gradfn(
            grad_fn=grad_fn,
            grad_clip_norm=self.grad_clip_norm,
        )

    def _get_policy_network(self):
        # Return any sort of useful callables relating to the policy network?
        pass

    def load_from_checkpoint(
        self,
        checkpoint_path: str,
        reload_step: int | None = None,
        log: logging.Logger | None = None,
    ) -> None:
        """Load training state from a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint directory (same as used by Checkpointer).
            log: Optional logger for status messages.
        """
        if log is None:
            log = logging.getLogger(__name__)

        log.info(f"Loading policy network from checkpoint: {checkpoint_path}")

        checkpoint_manager = ocp.CheckpointManager(checkpoint_path)
        if reload_step is None:
            reload_step = checkpoint_manager.latest_step()
        if reload_step is None:
            raise ValueError(f"No checkpoints found in {checkpoint_path}")

        log.info(f"Loading checkpoint from step {reload_step}")

        # Restore the training state using current state as template
        restored_state = checkpoint_manager.restore(
            reload_step,
            args=ocp.args.StandardRestore(self.training_state),  # pyright: ignore[reportCallIssue]
        )

        # Convert rng back from key_data format (checkpointer saves as key_data)
        self.training_state = restored_state.replace(  # pyright: ignore[reportAttributeAccessIssue]
            rng=jax.random.wrap_key_data(restored_state.rng)  # pyright: ignore[reportAttributeAccessIssue]
        )

        log.info("Policy network loaded successfully from checkpoint")

    def reset_optimizer(self, reset_step: bool = True) -> None:
        """Reset the optimizer state to its initial values.

        This reinitializes the optimizer state (including learning rate schedule)
        while keeping model parameters and other training state intact.

        Args:
            reset_step: If True (default), also reset the training step counter to 0.
                       Set to False to keep the step counter at its current value.
        """
        # Reinitialize the optimizer state using the current parameters
        new_opt_state = self.optimizer.init(self.training_state.params)

        # Create updated training state with fresh optimizer state
        if reset_step:
            self.training_state = self.training_state.replace(
                opt_state=new_opt_state,
                step=0,
            )
        else:
            self.training_state = self.training_state.replace(
                opt_state=new_opt_state,
            )

    def get_rollout_batch(
        self, key: Key, batch_size: int
    ) -> tuple[Array, Array, dict, Key]:
        theta, key = self.target.sample_prior(key, (batch_size,))
        epsilon, key = self.target.sample_epsilon(key, (batch_size,))
        policy_epsilon, key = self.target.sample_policy_epsilon(key, (batch_size,))

        # Generate roll out
        yT, xT, aux_data = self.target.reparam_sample(
            self.training_state.params,
            self.policy_network,
            theta=theta,
            epsilon=epsilon,
            policy_epsilon=policy_epsilon,
        )

        return yT, xT, aux_data, key

    def sample_reference_batch(
        self, key: Key, batch_shape: tuple[int, ...]
    ) -> tuple[Array, Array, Array, dict, Key]:
        """Draw a reference batch from the score network's training design sampler.

        Mirrors ``ScoreTrainer._sample_target`` (fast-sampler path): uses the
        recorded ``design_sampler_type``/``design_sampler_kwargs`` so diagnostic
        callbacks compare against the same design distribution the score network
        was trained on. Falls back to the model's default sampler when no type was
        recorded. Returns ``(y, xi, theta, aux_data, key)`` like ``fast_sample``.
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
            # Do not mask unrelated runtime TypeErrors (e.g. JAX tracer errors).
            msg = str(exc)
            if "unexpected keyword argument" in msg or "positional argument" in msg:
                raise TypeError(
                    "Target model does not support sampler_type/sampler_kwargs "
                    "in fast_sample"
                ) from exc
            raise
