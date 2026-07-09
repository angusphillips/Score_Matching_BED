import logging
from typing import TYPE_CHECKING, TypeAlias

import jax
import optax
import orbax.checkpoint as ocp
import tqdm
import tqdm.notebook
from wandb.sdk.wandb_run import Run

if TYPE_CHECKING:
    from policy_learning.trainers.train_policies import PolicyTrainer
    from policy_learning.trainers.train_scores import ScoreTrainer

    Trainer: TypeAlias = PolicyTrainer | ScoreTrainer

default_log = logging.getLogger(__name__)


class BaseCallback:
    def __init__(self, log: logging.Logger | None = None):
        self.log = log or default_log
        pass

    def on_train_start(self, trainer: "Trainer"):
        pass

    def on_batch_start(self, trainer: "Trainer"):
        pass

    def on_batch_end(self, trainer: "Trainer"):
        pass

    def on_train_end(self, trainer: "Trainer"):
        pass


class WindowedMetricAggregator:
    """Accumulate raw per-step scalar metrics and aggregate once per window.

    Values are stored as-is (JAX scalars or floats) so appending never forces a
    device sync; the single host transfer happens in ``consume``. Non-scalar
    values are skipped. Wandb key rendering is left to the owning callback so
    each callback's metric-name family stays unchanged.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._values: dict[str, list] = {}
        self._count = 0

    def append(self, step_metrics: dict) -> None:
        if not step_metrics:
            return
        self._count += 1
        for name, value in step_metrics.items():
            try:
                if jax.numpy.ndim(value) != 0:
                    continue
            except TypeError:
                continue
            self._values.setdefault(name, []).append(value)

    def consume(self) -> tuple[dict[str, tuple[float, float]], int]:
        """Return ``({name: (mean, max)}, window_size)`` and reset the window."""
        aggregates: dict[str, tuple[float, float]] = {}
        count = self._count
        for name, values in self._values.items():
            if not values:
                continue
            arr = jax.numpy.asarray(values)
            aggregates[name] = (float(arr.mean()), float(arr.max()))
        self.reset()
        return aggregates, count


class CountParameters(BaseCallback):
    """Count network parameters"""

    def __init__(
        self, name: str, wandb_logger: Run | None, log: logging.Logger | None = None
    ):
        self.wandb_logger = wandb_logger
        self.name = name
        if self.wandb_logger is None:
            self.save_log: list[dict] = []
        super().__init__(log=log)

    def on_train_start(self, trainer: "Trainer"):
        self.param_count = sum(
            x.size for x in jax.tree_util.tree_leaves(trainer.training_state.params)
        )
        metrics = {f"{self.name}_nb_params": self.param_count}
        if self.wandb_logger is not None:
            self.wandb_logger.log(metrics)
        else:
            self.save_log.append(metrics)
        self.log.info("Parameter count: %d", self.param_count)


class Checkpointer(BaseCallback):
    """Checkpoint model during training.

    Args:
        path: Directory path for saving checkpoints.
        frequency: Save every `frequency` iterations. Ignored if `checkpoint_at` is provided.
        checkpoint_at: Optional list of specific iterations at which to checkpoint.
            Useful for non-uniform checkpointing, e.g., [10, 100, 1000, 10000] for
            log-spaced checkpoints to observe convergence.
        max_to_keep: Maximum number of checkpoints to retain.
    """

    def __init__(
        self,
        path: str,
        frequency: int | None = None,
        checkpoint_at: list[int] | None = None,
        max_to_keep: int = 11,
    ):
        super().__init__()
        if frequency is None and checkpoint_at is None:
            raise ValueError("Must provide either `frequency` or `checkpoint_at`.")

        self.path = path
        self.frequency = frequency
        self.checkpoint_at = set(checkpoint_at) if checkpoint_at is not None else None

        # We do Python-level step filtering ourselves, so tell OCP to always save on call.
        self.ocp_options = ocp.CheckpointManagerOptions(
            save_interval_steps=1,  # type: ignore
            max_to_keep=max_to_keep,
            enable_async_checkpointing=False,
        )
        self.checkpoint_manager = ocp.CheckpointManager(
            self.path, options=self.ocp_options
        )

    def on_batch_end(self, trainer: "Trainer"):
        # Determine whether this step should be saved before doing the device sync
        # via key_data().  Previously key_data() was called unconditionally, causing
        # an unnecessary device sync on every non-checkpoint step.
        if self.checkpoint_at is not None:
            if trainer.step + 1 not in self.checkpoint_at:
                return
        elif self.frequency is not None:
            if trainer.step % self.frequency != 0:
                return

        training_state_keydata = trainer.training_state.replace(
            rng=jax.random.key_data(trainer.training_state.rng)
        )
        self.checkpoint_manager.save(
            trainer.step,
            args=ocp.args.StandardSave(training_state_keydata),  # pyright: ignore[reportCallIssue]
        )

    def on_train_end(self, trainer: "Trainer"):
        # Ensure checkpoint resources are flushed and released.
        self.checkpoint_manager.wait_until_finished()
        try:
            self.checkpoint_manager.close()
        except Exception as err:
            self.log.warning("Checkpoint manager close failed: %s", err)


class LearningRateLogger(BaseCallback):
    """Log the learning rate at a fixed frequency.

    Reports the *active* learning rate: it locates the ``ScaleByScheduleState``
    inside the live optimizer state (``trainer.training_state.opt_state`` — the
    state that ``train_step`` actually applies via ``state.tx``) and evaluates
    ``trainer.lr_schedule`` at that optimizer count. This reflects the real
    optimizer step rather than the callback's own counter, so a schedule that was
    reset / swapped (e.g. by ``ScoreTrainer.load_new_optimizer`` during finetune)
    is logged correctly. Falls back to ``lr_schedule(trainer.step)`` when no
    scheduled-scale state is present.
    """

    def __init__(
        self,
        group_name: str,
        wandb_logger: Run | None,
        log: logging.Logger | None = None,
    ):
        super().__init__(log)
        self.group_name = group_name
        self.wandb_logger = wandb_logger
        self.log_frequency = 10
        if self.wandb_logger is None:
            self.save_log: list[dict] = []

    @staticmethod
    def _find_schedule_count(opt_state) -> jax.Array | None:
        """Depth-first search for a ScaleByScheduleState.count in an optax state.

        optax states are NamedTuples (hence tuples), so we recurse into their
        fields to reach the scheduled-scale node wherever it sits in the chain.
        """
        stack = [opt_state]
        while stack:
            node = stack.pop()
            if isinstance(node, optax.ScaleByScheduleState):
                return node.count
            if isinstance(node, (tuple, list)):
                stack.extend(node)
        return None

    def on_batch_end(self, trainer: "Trainer"):
        step = int(trainer.step) + 1
        if step % self.log_frequency != 0:
            return

        lr_schedule = getattr(trainer, "lr_schedule", None)
        if lr_schedule is None:
            return

        # Prefer the live optimizer's schedule count over the callback step, so
        # the logged LR is the one actually being applied.
        opt_state = getattr(
            getattr(trainer, "training_state", None), "opt_state", None
        )
        count = (
            self._find_schedule_count(opt_state) if opt_state is not None else None
        )
        sched_at = count if count is not None else step

        step_key = "score_train_step" if "score" in self.group_name else "policy_train_step"
        metrics = {
            f"{self.group_name}/lr": float(lr_schedule(sched_at)),
            f"{self.group_name}/{step_key}": step,
        }
        if count is not None:
            metrics[f"{self.group_name}/opt_sched_count"] = int(count)

        if self.wandb_logger is not None:
            self.wandb_logger.log(metrics)
        else:
            self.save_log.append(metrics)


class ProgressBar(BaseCallback):
    """Show a tqdm progress bar updated every iteration."""

    def __init__(self, desc: str | None = None, notebook: bool = False):
        super().__init__()
        self.desc = desc
        self.progress_bar = None
        self.notebook = notebook

    def on_train_start(self, trainer: "Trainer"):
        if self.progress_bar is not None:
            try:
                self.progress_bar.close()
            finally:
                self.progress_bar = None

        total = trainer._num_iters
        desc = self.desc or getattr(trainer, "name", "Training")
        if self.notebook:
            self.progress_bar = tqdm.notebook.tqdm(total=total, desc=desc)
        else:
            self.progress_bar = tqdm.tqdm(total=total, desc=desc)

    def on_batch_end(self, trainer: "Trainer"):
        if self.progress_bar is None:
            return
        try:
            self.progress_bar.update(1)
        except Exception:
            pass

    def on_train_end(self, trainer: "Trainer"):
        if self.progress_bar is not None:
            try:
                self.progress_bar.close()
            finally:
                self.progress_bar = None
