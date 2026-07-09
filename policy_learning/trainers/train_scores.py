import logging
from collections.abc import Callable
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from jaxtyping import Array, PyTree
from jaxtyping import PRNGKeyArray as Key

from policy_learning.bed_models import BEDModel
from policy_learning.score_losses import get_score_matching_loss
from policy_learning.trainers.callbacks_base import BaseCallback
from policy_learning.utils.flax import create_train_state, get_train_step

ScoreFn = Callable[[Array, Array, dict | None], Array]
ScalarSchedule = float | int | Callable[[int], float]


class ScoreTrainer:
    def __init__(
        self,
        score_network: nn.Module,
        target_model: BEDModel,
        optimizer: optax.GradientTransformationExtraArgs,
        lr_schedule: optax.Schedule | None,
        key: Key,
        callbacks: list[BaseCallback],
        y_score_loss_weight: float = 1.0,
        grad_clip_norm: float | None = None,
        design_sampler_type: str | None = None,
        design_sampler_kwargs: dict[str, Any] | None = None,
    ):
        self._num_iters = 0
        self.score_network = score_network
        self.target = target_model
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.loss_fn = get_score_matching_loss(
            self.score_network, self.target, y_score_loss_weight=y_score_loss_weight
        )
        self.grad_clip_norm = grad_clip_norm
        self.design_sampler_type = design_sampler_type
        self.design_sampler_kwargs = (
            {} if design_sampler_kwargs is None else dict(design_sampler_kwargs)
        )
        self.callbacks = callbacks
        self.key = key

        # initialise model and training state
        y_init, xi_init, _, aux_data, self.key = self._sample_target(
            self.key,
            (4,),
            use_fast_sampler=False,
        )
        # Positive init sigma (lambda=1): the noise-conditioned network takes
        # log(sigma) internally, so 0 would be invalid. Ignored when the network
        # is not sigma-conditioned.
        sigma_init = jnp.ones((4,))

        self.training_state, self.key = create_train_state(
            self.score_network,
            self.key,
            self.optimizer,
            y=y_init,
            xi=xi_init,
            aux_data=aux_data,
            sigma=sigma_init,  # Only needed for ARDAE or noise annealing but the other models can ignore it
            train=False,
            mutable=[
                "params",
                "rff_freqs",
                "batch_stats",
            ],  # Only uses the ones that are actually needed
        )

        self.train_step = get_train_step(
            self.loss_fn,
            grad_clip_norm=self.grad_clip_norm,
        )

    def _trigger_event(self, event_name: str):
        for cb in self.callbacks:
            func = getattr(cb, event_name, None)
            if func:
                func(self)

    def _format_last_step_metrics(
        self,
        training_state,
        step_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        # TODO Optionally log anything from training state as well...
        return step_metrics

    def _score_variables_from_state(self) -> PyTree:
        if self.training_state.batch_stats is not None:
            return {
                "params": self.training_state.params_ema,
                "rff_freqs": self.training_state.rff_freqs,
                "batch_stats": self.training_state.batch_stats,
            }
        if self.training_state.rff_freqs is not None:
            return {
                "params": self.training_state.params_ema,
                "rff_freqs": self.training_state.rff_freqs,
            }
        return {"params": self.training_state.params_ema}

    def _score_from_z_with_variables(
        self,
        score_variables: PyTree,
        z: Array,
        aux_data: dict | None = None,
    ) -> Array:
        if aux_data is None:
            aux_data = {}
        y = z[..., : self.target.d_y]
        xi = z[..., self.target.d_y :]
        joint_score = self.score_network.apply(
            score_variables,
            y=y,
            xi=xi,
            sigma=jnp.ones((*y.shape[:-2],)),
            aux_data=aux_data,
            train=False,
        )
        assert isinstance(joint_score, Array)
        return joint_score

    def _sample_target(
        self,
        key: Key,
        batch_shape: tuple[int, ...],
        use_fast_sampler: bool,
    ):
        sampler_fn = self.target.fast_sample if use_fast_sampler else self.target.sample

        if self.design_sampler_type is None:
            return sampler_fn(key, batch_shape)

        try:
            return sampler_fn(
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
                    "Target model does not support sampler_type/sampler_kwargs in sample/fast_sample"
                ) from exc
            raise

    def train(self, num_iters: int, batch_size: int, start_step: int = 0):
        """Train for ``num_iters`` update steps.

        ``start_step`` (default 0 -> unchanged behaviour) resumes a previously
        checkpointed run: it sets the step counter so the loop runs from there to
        ``num_iters`` instead of restarting at 0. The optimizer state (incl. the
        LR-schedule count) and rng must already be restored into ``training_state``
        via :meth:`load_from_checkpoint`; only the loop bound / step counter is
        adjusted here. A checkpoint saved at orbax key ``K`` holds the state after
        ``K + 1`` updates, so the caller passes ``start_step = K + 1`` to continue
        consistently (matching the Checkpointer's ``step + 1`` save trigger).
        """
        self._num_iters = num_iters
        self._trigger_event("on_train_start")
        self.step = int(start_step)
        for _ in range(int(start_step), num_iters + 1):
            self._trigger_event("on_batch_start")
            y, xi, theta, aux_data, self.key = self._sample_target(
                self.key,
                (batch_size,),
                use_fast_sampler=True,
            )
            batch = (y, xi, theta, aux_data)
            self.training_state, step_metrics = self.train_step(
                self.training_state,
                batch,
            )

            self.last_step_metrics = self._format_last_step_metrics(
                self.training_state,
                step_metrics,
            )
            self._trigger_event("on_batch_end")
            self.step += 1

        self._trigger_event("on_train_end")

    def load_from_checkpoint(
        self,
        checkpoint_path: str,
        step: int | None = None,
        log: logging.Logger | None = None,
    ) -> None:
        """Load training state from a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint directory (same as used by Checkpointer).
            step: Optional checkpoint step to load. If None, loads the latest checkpoint.
            log: Optional logger for status messages.
        """
        if log is None:
            log = logging.getLogger(__name__)

        log.info(f"Loading score network from checkpoint: {checkpoint_path}")

        checkpoint_manager = ocp.CheckpointManager(checkpoint_path)

        if step is None:
            load_step = checkpoint_manager.latest_step()
            if load_step is None:
                raise ValueError(f"No checkpoints found in {checkpoint_path}")
        else:
            load_step = step

        self.step = int(load_step)

        log.info(f"Loading checkpoint from step {load_step}")

        # Restore the training state using current state as template
        restored_state = checkpoint_manager.restore(
            load_step,
            args=ocp.args.StandardRestore(self.training_state),  # pyright: ignore[reportCallIssue]
        )

        # Convert rng back from key_data format (checkpointer saves as key_data)
        self.training_state = restored_state.replace(  # pyright: ignore[reportAttributeAccessIssue]
            rng=jax.random.wrap_key_data(restored_state.rng)  # pyright: ignore[reportAttributeAccessIssue]
        )

        log.info("Score network loaded successfully from checkpoint")

    def get_score_fn(self) -> ScoreFn:
        score_variables = self._score_variables_from_state()

        # @jax.jit
        def score_function_apply(
            y: Array,
            xi: Array,
            aux_data: dict | None = None,
            score_variables: PyTree = score_variables,
        ):
            z = jnp.concatenate([y, xi], axis=-1)
            return self._score_from_z_with_variables(score_variables, z, aux_data)

        return score_function_apply

    def get_data_score_fn(self) -> ScoreFn:
        score_fn: ScoreFn = self.get_score_fn()

        def data_score_fn(y: Array, xi: Array, aux_data: dict | None = None) -> Array:
            return score_fn(y, xi, aux_data)[..., : self.target.d_y]

        return jax.jit(data_score_fn)

    def get_design_score_fn(self) -> ScoreFn:
        score_fn: ScoreFn = self.get_score_fn()

        def design_score_fn(y: Array, xi: Array, aux_data: dict | None = None) -> Array:
            return score_fn(y, xi, aux_data)[..., self.target.d_y :]

        return jax.jit(design_score_fn)

    def get_unbatched_score_fns(self) -> tuple[ScoreFn, ScoreFn]:
        score_fn: ScoreFn = self.get_score_fn()

        def data_score_fn(y: Array, xi: Array, aux_data: dict | None = None) -> Array:
            return score_fn(y, xi, aux_data)[0, :, : self.target.d_y]

        def design_score_fn(y: Array, xi: Array, aux_data: dict | None = None):
            return score_fn(y, xi, aux_data)[0, :, self.target.d_y :]

        return jax.jit(data_score_fn), jax.jit(design_score_fn)
