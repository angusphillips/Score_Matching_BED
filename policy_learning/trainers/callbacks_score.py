import logging
from typing import TYPE_CHECKING, Any, TypeAlias

from wandb.sdk.wandb_run import Run

from policy_learning.trainers.callbacks_base import (
    BaseCallback,
    WindowedMetricAggregator,
)

if TYPE_CHECKING:
    from policy_learning.trainers.train_scores import ScoreTrainer

    Trainer: TypeAlias = ScoreTrainer


class ScoreTrainingMetrics(BaseCallback):
    """Score-training metrics: windowed loss components, grad norms, clip rate.

    Task: *score training metrics*. Aggregates ``trainer.last_step_metrics``
    (loss components, grad/param/update norms, clip indicator) over each
    ``log_freq`` window and logs window mean/max under ``{group_name}/``.
    Wandb keys: ``{k}_window_{mean,max}``,
    ``clip_rate_window`` for the clip indicator, and the y/x weighted-loss
    ratio diagnostic.
    """

    def __init__(
        self,
        log_freq: int | None,
        group_name: str = "score",
        wandb_logger: Run | None = None,
        log: logging.Logger | None = None,
    ):
        super().__init__(log)
        self.wandb_logger = wandb_logger
        self.group_name = group_name
        self.log_frequency = log_freq
        self._window = WindowedMetricAggregator()

    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
        *,
        phase: str = "score",
        wandb_logger: Run | None = None,
        log: logging.Logger | None = None,
    ) -> "ScoreTrainingMetrics":
        """Build from ``cfg[phase]["log_freq"]``; ``phase`` is the cfg section
        and wandb group name (e.g. "score")."""
        return cls(
            log_freq=cfg[phase].get("log_freq"),
            group_name=phase,
            wandb_logger=wandb_logger,
            log=log,
        )

    def _window_metric_name(self, key: str, aggregate: str) -> str:
        if key.endswith("clip_applied"):
            key_prefix = key[: -len("clip_applied")]
            if aggregate == "mean":
                return f"{self.group_name}/{key_prefix}clip_rate_window"
            return f"{self.group_name}/{key_prefix}clip_rate_window_{aggregate}"
        return f"{self.group_name}/{key}_window_{aggregate}"

    def _consume_window_metrics(self) -> dict[str, float]:
        aggregates, _ = self._window.consume()
        window_metrics: dict[str, float] = {}
        for key, (mean_value, max_value) in aggregates.items():
            window_metrics[self._window_metric_name(key, "mean")] = mean_value
            window_metrics[self._window_metric_name(key, "max")] = max_value

        # Diagnostic: ratio-of-window-means for weighted y/x loss components.
        y_mean_key = self._window_metric_name("loss_y_component", "mean")
        xw_mean_key = self._window_metric_name("loss_x_component_weighted", "mean")
        if y_mean_key in window_metrics and xw_mean_key in window_metrics:
            xw_mean = window_metrics[xw_mean_key]
            window_metrics[
                f"{self.group_name}/loss_y_to_x_weighted_ratio_from_window_means"
            ] = float(window_metrics[y_mean_key] / (xw_mean + 1e-8))

        return window_metrics

    def _log(self, metrics: dict[str, float]) -> None:
        if self.wandb_logger is not None:
            self.wandb_logger.log(metrics)
        else:
            self.save_log.append(metrics)
            self.log.info("Score training metrics: %s", metrics)

    def on_train_start(self, trainer: "Trainer"):  # type: ignore[override]
        self._window.reset()
        if self.wandb_logger is None:
            self.save_log: list[dict] = []

    def on_batch_end(self, trainer: "Trainer"):  # type: ignore[override]
        if self.log_frequency is None:
            return
        step = int(trainer.step) + 1
        self._window.append(getattr(trainer, "last_step_metrics", {}))
        if step % self.log_frequency != 0:
            return
        self._log(
            {
                f"{self.group_name}/score_train_step": step,
                **self._consume_window_metrics(),
            }
        )

    def on_train_end(self, trainer: "Trainer"):  # type: ignore[override]
        if not hasattr(self, "save_log") and self.wandb_logger is None:
            self.save_log = []
        self._log(
            {
                f"{self.group_name}/score_train_step": int(trainer.step),
                **self._consume_window_metrics(),
            }
        )


