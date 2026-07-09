"""Callbacks for variational distribution training."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from wandb.sdk.data_types.image import Image
from wandb.sdk.wandb_run import Run

from policy_learning.trainers.callbacks_base import BaseCallback

if TYPE_CHECKING:
    from policy_learning.variational.train_variational import VariationalTrainer


class EquinoxCheckpointer(BaseCallback):
    """Checkpoint Equinox models during training.

    Uses Equinox's native serialization functions (tree_serialise_leaves and
    tree_deserialise_leaves) for saving and loading PyTrees containing Equinox modules.
    Compatible with the variational trainer's Equinox-based flow models.

    Args:
        path: Directory path to save checkpoints
        frequency: Save checkpoint every `frequency` training iterations.
            Ignored if `checkpoint_at` is provided.
        checkpoint_at: Optional list of specific iterations at which to checkpoint.
            Useful for non-uniform checkpointing, e.g., [10, 100, 1000, 10000].
        max_to_keep: Maximum number of checkpoints to keep (default: 10)

    Example:
        >>> checkpointer = EquinoxCheckpointer(
        ...     path="checkpoints/variational",
        ...     frequency=1000,
        ...     max_to_keep=5
        ... )
        >>> trainer = VariationalTrainer(
        ...     variational_dist=dist,
        ...     target_model=model,
        ...     optimizer=optimizer,
        ...     key=key,
        ...     callbacks=[checkpointer, ...]
        ... )
        >>> trainer.train(num_iters=10000, batch_size=256)
        >>>
        >>> # Later, restore the latest checkpoint
        >>> new_trainer = VariationalTrainer(...)
        >>> checkpointer.restore(new_trainer)  # Restores latest
        >>> # Or restore a specific step
        >>> checkpointer.restore(new_trainer, step=5000)
    """

    def __init__(
        self,
        path: str,
        frequency: int | None = None,
        checkpoint_at: list[int] | None = None,
        max_to_keep: int = 10,
    ):
        super().__init__()
        if frequency is None and checkpoint_at is None:
            raise ValueError("Must provide either `frequency` or `checkpoint_at`.")
        self.path = Path(path)
        self.frequency = frequency
        self.checkpoint_at = set(checkpoint_at) if checkpoint_at is not None else None
        self.max_to_keep = max_to_keep
        self.checkpoint_files: list[tuple[int, Path]] = []

        # Create checkpoint directory
        self.path.mkdir(parents=True, exist_ok=True)

    def _get_checkpoint_path(self, step: int) -> Path:
        """Get the path for a checkpoint at a given step."""
        return self.path / f"checkpoint_{step}.eqx"

    def _save_checkpoint(self, trainer: "VariationalTrainer", step: int):  # type: ignore[override]
        """Save a checkpoint."""
        checkpoint_path = self._get_checkpoint_path(step)

        # Create a state dictionary with all necessary components
        state = {
            "params": trainer.params,
            "static": trainer.static,
            "opt_state": trainer.opt_state,
            "key": jax.random.key_data(trainer.key),  # Save key data (uint32 array)
            "step": step,
            "loss": trainer._last_loss,
            "cum_loss": trainer._cum_loss_since_start,
        }

        # Save using Equinox's serialization
        eqx.tree_serialise_leaves(checkpoint_path, state)

        # Track this checkpoint
        self.checkpoint_files.append((step, checkpoint_path))

        # Remove old checkpoints if we exceed max_to_keep
        if len(self.checkpoint_files) > self.max_to_keep:
            # Sort by step and remove oldest
            self.checkpoint_files.sort(key=lambda x: x[0])
            _, old_path = self.checkpoint_files.pop(0)
            if old_path.exists():
                old_path.unlink()

    def on_batch_end(self, trainer: "VariationalTrainer"):  # type: ignore[override]
        """Save checkpoint if at the right frequency."""
        step = int(trainer.step)
        if self.checkpoint_at is not None:
            if step + 1 not in self.checkpoint_at:
                return
        elif self.frequency is not None:
            if step <= 0 or step % self.frequency != 0:
                return
        else:
            return

        self._save_checkpoint(trainer, step)

    def on_train_end(self, trainer: "VariationalTrainer"):  # type: ignore[override]
        """Save final checkpoint at end of training."""
        step = int(trainer.step)
        self._save_checkpoint(trainer, step)

    def restore(self, trainer: "VariationalTrainer", step: int | None = None):  # type: ignore[override]
        """Restore checkpoint into the trainer.

        Args:
            trainer: VariationalTrainer to restore state into
            step: Specific step to restore. If None, restores the latest checkpoint.
        """
        if step is None:
            # Find latest checkpoint
            checkpoints = list(self.path.glob("checkpoint_*.eqx"))
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {self.path}")

            # Extract step numbers and find max
            steps = [int(p.stem.split("_")[1]) for p in checkpoints]
            step = max(steps)

        checkpoint_path = self._get_checkpoint_path(step)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint using the structure from params and static
        template = {
            "params": trainer.params,
            "static": trainer.static,
            "opt_state": trainer.opt_state,
            "key": jax.random.key_data(trainer.key),
            "step": 0,
            "loss": 0.0,
            "cum_loss": 0.0,
        }

        state = eqx.tree_deserialise_leaves(checkpoint_path, template)

        # Restore state to trainer
        trainer.params = state["params"]
        trainer.static = state["static"]
        trainer.opt_state = state["opt_state"]
        trainer.key = jax.random.wrap_key_data(state["key"])
        trainer.step = int(state["step"])
        trainer._last_loss = float(state["loss"])
        trainer._cum_loss_since_start = float(state["cum_loss"])

        self.log.info(f"Restored checkpoint from step {step}")

        return step


class VariationalLossLogger(BaseCallback):
    """Log training loss and custom evaluation metrics."""

    def __init__(
        self,
        log_freq: int,
        eval_batch_size: int = 256,
        evals_dict: dict[str, Callable] | None = None,
        wandb_logger: Run | None = None,
        log: logging.Logger | None = None,
        group_name: str = "variational",
    ):
        super().__init__(log)
        self.wandb_logger = wandb_logger
        self.eval_batch_size = eval_batch_size
        self.evals_dict = evals_dict or {}
        self.log_frequency = log_freq
        self.group_name = group_name
        self.last_log_step = 0
        self._reset_window_metrics()
        if self.wandb_logger is None:
            self.save_log: list[dict] = []

    def _reset_window_metrics(self) -> None:
        self._window_metrics: dict[str, list[float]] = {}
        self._window_size = 0

    def _extract_numeric_step_metrics(
        self, step_metrics: dict[str, Any]
    ) -> dict[str, float]:
        numeric_metrics: dict[str, float] = {}
        for metric_name, metric_value in step_metrics.items():
            try:
                numeric_metrics[metric_name] = float(metric_value)
            except (TypeError, ValueError):
                continue
        return numeric_metrics

    def _append_window_metrics(self, numeric_step_metrics: dict[str, float]) -> None:
        if not numeric_step_metrics:
            return

        self._window_size += 1
        for metric_name, metric_value in numeric_step_metrics.items():
            self._window_metrics.setdefault(metric_name, []).append(metric_value)

    def _consume_window_aggregate_metrics(
        self,
        clip_norm: float | None,
    ) -> dict[str, float]:
        if self._window_size == 0:
            return {}

        window_metrics: dict[str, float] = {
            f"{self.group_name}/window_size": float(self._window_size),
        }

        for metric_name, metric_values in self._window_metrics.items():
            metric_window = jnp.asarray(metric_values)
            window_metrics[f"{self.group_name}/{metric_name}_window_mean"] = float(
                jnp.mean(metric_window)
            )
            window_metrics[f"{self.group_name}/{metric_name}_window_max"] = float(
                jnp.max(metric_window)
            )

        if clip_norm is not None and "clip_applied" in self._window_metrics:
            clip_applied_window = jnp.asarray(self._window_metrics["clip_applied"])
            window_metrics[f"{self.group_name}/clip_rate_window"] = float(
                jnp.mean(clip_applied_window)
            )

        self._reset_window_metrics()
        return window_metrics

    def _prefix_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        return {f"{self.group_name}/{k}": v for k, v in metrics.items()}

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        if self.wandb_logger is not None:
            self.wandb_logger.log(metrics)
        else:
            self.save_log.append(metrics)
            self.log.info("Variational metrics: %s", metrics)

    def on_train_start(self, trainer: "VariationalTrainer"):  # type: ignore[override]
        self.last_log_step = 0
        self._reset_window_metrics()

    def on_batch_end(self, trainer: "VariationalTrainer"):  # type: ignore[override]
        step = int(trainer.step)
        if self.log_frequency is None:
            return

        step_metrics = getattr(trainer, "last_step_metrics", {})
        numeric_step_metrics = self._extract_numeric_step_metrics(step_metrics)
        self._append_window_metrics(numeric_step_metrics)

        if step % self.log_frequency != 0:
            return

        # Get losses
        curr_loss = float(trainer._last_loss)
        av_loss_since_last_log = (
            float(trainer._cum_loss_since_last_log) / (step - self.last_log_step)
            if step > self.last_log_step
            else curr_loss
        )

        # Update last log step and reset cumulative loss
        self.last_log_step = step
        trainer._cum_loss_since_last_log = 0.0

        self.log.info(
            f"Variational Loss: {av_loss_since_last_log:.4f}, "
            f"Step: {step}/{trainer._num_iters}"
        )

        # current_metrics = self._prefix_metrics(numeric_step_metrics)
        clip_norm = getattr(trainer, "grad_clip_norm", None)

        window_metrics = self._consume_window_aggregate_metrics(clip_norm)

        full_metrics = {
            f"{self.group_name}/train_loss": curr_loss,
            f"{self.group_name}/av_loss_since_last_log": av_loss_since_last_log,
            f"{self.group_name}_train_step": float(step),
            # **current_metrics,
            **window_metrics,
        }

        self._log_metrics(full_metrics)

    def on_train_end(self, trainer: "VariationalTrainer"):  # type: ignore[override]
        step = int(trainer.step)
        # step_metrics = getattr(trainer, "last_step_metrics", {})
        # numeric_step_metrics = self._extract_numeric_step_metrics(step_metrics)

        curr_loss = float(trainer._last_loss)
        av_loss_since_last_log = (
            float(trainer._cum_loss_since_last_log) / (step - self.last_log_step)
            if step > self.last_log_step
            else curr_loss
        )

        # current_metrics = self._prefix_metrics(numeric_step_metrics)
        clip_norm = getattr(trainer, "grad_clip_norm", None)

        window_metrics = self._consume_window_aggregate_metrics(clip_norm)
        self._log_metrics(
            {
                f"{self.group_name}/train_loss": curr_loss,
                f"{self.group_name}/av_loss_since_last_log": av_loss_since_last_log,
                f"{self.group_name}_train_step": float(step),
                # **current_metrics,
                **window_metrics,
            }
        )
        self.log.info("Variational training complete")


class VariationalPlotCallback(BaseCallback):
    """Generate plots during variational training."""

    def __init__(
        self,
        log_freq: int | None,
        plot_fn_dict: dict[str, Callable] | None = None,
        wandb_logger: Run | None = None,
        log: logging.Logger | None = None,
    ):
        super().__init__(log)
        self.wandb_logger = wandb_logger
        self.plot_fn_dict = plot_fn_dict or {}
        self.log_frequency = log_freq

    def on_batch_end(self, trainer: "VariationalTrainer"):  # type: ignore[override]
        step = int(trainer.step)
        if self.log_frequency is None or step % self.log_frequency != 0:
            return

        for name, plot_fn in self.plot_fn_dict.items():
            try:
                fig = plot_fn(trainer)
                if self.wandb_logger is not None and fig is not None:
                    self.wandb_logger.log({f"variational/{name}": Image(fig)})
                plt.close(fig)
            except Exception as e:
                self.log.warning(f"Plot {name} failed: {e}")


class VariationalSamplePlot(BaseCallback):
    """Plot samples from the variational distribution."""

    def __init__(
        self,
        log_freq: int,
        num_samples: int = 500,
        wandb_logger: Run | None = None,
        log: logging.Logger | None = None,
    ):
        super().__init__(log)
        self.wandb_logger = wandb_logger
        self.log_frequency = log_freq
        self.num_samples = num_samples

    def on_batch_end(self, trainer: "VariationalTrainer"):  # type: ignore[override]
        step = int(trainer.step)
        if self.log_frequency is None or step % self.log_frequency != 0:
            return

        trainer.key, sample_key, target_key = jax.random.split(trainer.key, 3)

        # Sample from variational distribution
        y_var, xi_var = trainer.sample(sample_key, (self.num_samples,))

        # Sample from target distribution for comparison
        y_target, xi_target, _, _, _ = trainer.target.sample(
            target_key, (self.num_samples,)
        )

        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot y marginals (first dimension, first time step)
        axes[0, 0].hist(
            y_var[:, 0, 0].flatten(),
            bins=50,
            alpha=0.5,
            label="Variational",
            density=True,
        )
        axes[0, 0].hist(
            y_target[:, 0, 0].flatten(),
            bins=50,
            alpha=0.5,
            label="Target",
            density=True,
        )
        axes[0, 0].set_title("y marginal (t=0, dim=0)")
        axes[0, 0].legend()

        # Plot xi marginals (first dimension, first time step)
        axes[0, 1].hist(
            xi_var[:, 0, 0].flatten(),
            bins=50,
            alpha=0.5,
            label="Variational",
            density=True,
        )
        axes[0, 1].hist(
            xi_target[:, 0, 0].flatten(),
            bins=50,
            alpha=0.5,
            label="Target",
            density=True,
        )
        axes[0, 1].set_title("xi marginal (t=0, dim=0)")
        axes[0, 1].legend()

        # 2D scatter: y vs xi (first dim, first time step)
        axes[1, 0].scatter(
            xi_var[:, 0, 0],
            y_var[:, 0, 0],
            alpha=0.3,
            s=5,
            label="Variational",
        )
        axes[1, 0].scatter(
            xi_target[:, 0, 0],
            y_target[:, 0, 0],
            alpha=0.3,
            s=5,
            label="Target",
        )
        axes[1, 0].set_xlabel("xi")
        axes[1, 0].set_ylabel("y")
        axes[1, 0].set_title("Joint (y, xi) scatter")
        axes[1, 0].legend()

        # Plot log probability histogram
        log_prob_fn = trainer.get_log_prob_fn()
        log_probs = log_prob_fn(y_target, xi_target)
        axes[1, 1].hist(log_probs.flatten(), bins=50)
        axes[1, 1].set_title("log q(y, xi) on target samples")
        axes[1, 1].set_xlabel("log probability")

        plt.tight_layout()

        if self.wandb_logger is not None:
            self.wandb_logger.log({"variational/sample_comparison": Image(fig)})
        else:
            plt.show()

        plt.close(fig)
