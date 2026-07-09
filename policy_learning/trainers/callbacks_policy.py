import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import wandb
from jaxtyping import PRNGKeyArray as Key
from wandb.sdk.data_types.image import Image
from wandb.sdk.wandb_run import Run

from policy_learning.bed_models import BEDModel
from policy_learning.trainers.callbacks_base import (
    BaseCallback,
    WindowedMetricAggregator,
)

if TYPE_CHECKING:
    from policy_learning.trainers.train_policies import PolicyTrainer


class PolicyTrainingMetrics(BaseCallback):
    """Policy-training metrics: windowed grad/update norms, clip rate.

    Task: *policy training metrics* (cheap). Aggregates
    ``trainer.last_step_metrics`` over each ``log_freq`` window; logs window
    mean/max plus the current step's values, clip threshold/rate, and the
    grad-to-clip ratio under ``policy/``.
    """

    def __init__(
        self,
        log_freq: int | None,
        group_name: str = "policy",
        wandb_logger: Run | None = None,
        log: logging.Logger | None = None,
    ):
        super().__init__(log)
        self.log_frequency = log_freq
        self.group_name = group_name
        self.wandb_logger = wandb_logger
        self._window = WindowedMetricAggregator()
        if self.wandb_logger is None:
            self.save_log: list[dict] = []

    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
        *,
        group_name: str = "policy",
        wandb_logger: "Run | None" = None,
        log: logging.Logger | None = None,
    ) -> "PolicyTrainingMetrics":
        """Build from ``policy.optim_log_freq`` (null = off)."""
        return cls(
            log_freq=cfg["policy"].get("optim_log_freq"),
            group_name=group_name,
            wandb_logger=wandb_logger,
            log=log,
        )

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

    def _consume_window_metrics(self, clip_norm: float | None) -> dict[str, float]:
        aggregates, window_size = self._window.consume()
        if window_size == 0:
            return {}

        window_metrics: dict[str, float] = {
            f"{self.group_name}/window_size": float(window_size),
        }
        for metric_name, (mean_value, max_value) in aggregates.items():
            window_metrics[f"{self.group_name}/{metric_name}_window_mean"] = mean_value
            window_metrics[f"{self.group_name}/{metric_name}_window_max"] = max_value

        if clip_norm is not None and "clip_applied" in aggregates:
            window_metrics[f"{self.group_name}/clip_rate_window"] = aggregates[
                "clip_applied"
            ][0]

        return window_metrics

    def _prefix_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        return {f"{self.group_name}/{k}": v for k, v in metrics.items()}

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        if self.wandb_logger is not None:
            self.wandb_logger.log(metrics)
        else:
            self.save_log.append(metrics)
            self.log.info("Policy training metrics: %s", metrics)

    def _current_step_metrics(self, trainer: "PolicyTrainer") -> dict[str, float]:
        # Sync point: convert the current step's metrics to float for logging.
        step_metrics = getattr(trainer, "last_step_metrics", {})
        numeric_step_metrics = self._extract_numeric_step_metrics(step_metrics)
        current_metrics = self._prefix_metrics(numeric_step_metrics)
        clip_norm = getattr(trainer, "grad_clip_norm", None)
        if clip_norm is not None:
            clip_norm_float = float(clip_norm)
            current_metrics[f"{self.group_name}/clip_threshold"] = clip_norm_float
            if "grad_norm_preclip" in numeric_step_metrics:
                current_metrics[f"{self.group_name}/grad_to_clip_ratio"] = (
                    numeric_step_metrics["grad_norm_preclip"]
                    / max(clip_norm_float, 1e-12)
                )
        return current_metrics

    def on_train_start(self, trainer: "PolicyTrainer"):  # type: ignore[override]
        self._window.reset()

    def on_batch_end(self, trainer: "PolicyTrainer"):  # type: ignore[override]
        if self.log_frequency is None:
            return

        step = int(trainer.step) + 1
        # Accumulate raw metrics (JAX arrays) — defer float() conversion to avoid a
        # device sync on every step.  Conversion happens once per log period below.
        self._window.append(getattr(trainer, "last_step_metrics", {}))

        if step % self.log_frequency != 0:
            return

        clip_norm = getattr(trainer, "grad_clip_norm", None)
        self._log_metrics(
            {
                f"{self.group_name}/policy_train_step": float(step),
                **self._current_step_metrics(trainer),
                **self._consume_window_metrics(clip_norm),
            }
        )

    def on_train_end(self, trainer: "PolicyTrainer"):  # type: ignore[override]
        if self.log_frequency is None:
            return
        clip_norm = getattr(trainer, "grad_clip_norm", None)
        self._log_metrics(
            {
                f"{self.group_name}/policy_train_step": float(int(trainer.step)),
                **self._current_step_metrics(trainer),
                **self._consume_window_metrics(clip_norm),
            }
        )


class PolicyEIGBoundEvals(BaseCallback):
    def __init__(
        self,
        key: Key,
        policy_eval_fn: Callable,
        log_freq: int | None,
        eval_at: list[int] | None,
        inner_samples: int,
        outer_samples: int,
        batch_size: int,
        plot_fn: Callable | None,
        reuse_inner_samples: bool = False,
        n_plot_rollouts: int = 1,
        plot_during_training: bool = True,
        group_name: str | None = None,
        wandb_logger: Run | None = None,
        log: logging.Logger | None = None,
        log_step_0: bool = True,
    ):
        super().__init__(log)
        self.key = key
        self.wandb_logger = wandb_logger
        self.policy_eval_fn = policy_eval_fn
        self.log_frequency = log_freq
        self.eval_at = eval_at
        self.plot_during_training = plot_during_training
        self.inner_samples = inner_samples
        self.outer_samples = outer_samples
        self.batch_size = batch_size
        self.reuse_inner_samples = reuse_inner_samples
        self.plot_fn = plot_fn
        self.n_plot_rollouts = n_plot_rollouts
        self.group_name = group_name if group_name else "policy"
        self.log_step_0 = log_step_0
        self.suppress_on_train_end = False

        self.skip = (self.log_frequency is None) and (self.eval_at is None)

    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
        key: Key,
        *,
        policy_eval_fn: Callable,
        plot_fn: Callable | None = None,
        n_plot_rollouts: int = 5,
        group_name: str = "policy",
        log_step_0: bool = True,
        wandb_logger: Run | None = None,
        log: logging.Logger | None = None,
    ) -> "PolicyEIGBoundEvals":
        """Build from ``policy.eig_bounds.{freq, eval_at}`` (freq null = off)
        plus the shared eval sample sizes ``policy.eval_{inner,outer}_samples``
        and ``policy.eval_outer_batch_size``."""
        eig_bounds_cfg = cfg["policy"].get("eig_bounds", None) or {}
        return cls(
            key=key,
            policy_eval_fn=policy_eval_fn,
            log_freq=eig_bounds_cfg.get("freq"),
            eval_at=eig_bounds_cfg.get("eval_at"),
            inner_samples=int(cfg["policy"]["eval_inner_samples"]),
            outer_samples=int(cfg["policy"]["eval_outer_samples"]),
            batch_size=int(cfg["policy"]["eval_outer_batch_size"]),
            plot_fn=plot_fn,
            n_plot_rollouts=n_plot_rollouts,
            plot_during_training=bool(cfg["policy"].get("plot_during_training", True)),
            group_name=group_name,
            log_step_0=log_step_0,
            wandb_logger=wandb_logger,
            log=log,
        )

    def on_train_start(self, trainer: "PolicyTrainer"):  # type: ignore[override]
        if self.skip:
            return
        if self.log_step_0:
            mean_lb, mean_ub, std_lb, std_ub, self.key, self.batch_size = (
                self.policy_eval_fn(
                    self.key,
                    trainer.target,
                    trainer.policy_network,
                    trainer.training_state.params_ema,
                    L=self.inner_samples,
                    reuse_inner_samples=self.reuse_inner_samples,
                    outer_samples=self.outer_samples,
                    outer_batch_size=self.batch_size,
                )
            )

            pce_bound_dict = {
                f"{self.group_name}/ub": mean_ub,
                f"{self.group_name}/lb": mean_lb,
                f"{self.group_name}/se_ub": std_ub,
                f"{self.group_name}/se_lb": std_lb,
                f"{self.group_name}/policy_train_step": 0,
            }

            if self.wandb_logger is not None:
                self.wandb_logger.log(pce_bound_dict)
                if self.plot_during_training:
                    plot_dict: dict[str, Any] = {}
                    if self.plot_fn is not None:
                        plot_dict, self.key = self.plot_fn(
                            self.key,
                            trainer.target,
                            trainer.policy_network,
                            trainer.training_state.params_ema,
                            self.n_plot_rollouts,
                        )
                        self.wandb_logger.log(
                            {
                                f"{self.group_name}/{name}": Image(fig)
                                for name, fig in plot_dict.items()
                            }
                        )
                        for fig in plot_dict.values():
                            plt.close(fig)

            if self.wandb_logger is None:
                self.log_hist: list[dict] = []

            self.log.info(f"Step: {0}/{int(trainer._num_iters)}, lb: {mean_lb:.2f}.")

        self.t0 = time.time()

    def on_batch_end(self, trainer: "PolicyTrainer"):  # type: ignore[override]
        if self.skip:
            return

        step = int(trainer.step) + 1
        trigger = False
        if self.log_frequency is not None:
            if step % self.log_frequency == 0:
                trigger = True
        if self.eval_at is not None:
            if step in self.eval_at:
                trigger = True
        if not trigger:
            return

        t1 = time.time()
        elapsed = t1 - self.t0
        # if step == 0 or elapsed <= 0:
        #     it_per_sec = 0.0
        # else:
        #     it_per_sec = float(self.log_frequency) / elapsed  # type: ignore

        mean_lb, mean_ub, std_lb, std_ub, self.key, self.batch_size = (
            self.policy_eval_fn(
                self.key,
                trainer.target,
                trainer.policy_network,
                trainer.training_state.params_ema,
                L=self.inner_samples,
                reuse_inner_samples=self.reuse_inner_samples,
                outer_samples=self.outer_samples,
                outer_batch_size=self.batch_size,
            )
        )

        pce_bound_dict = {
            f"{self.group_name}/ub": mean_ub,
            f"{self.group_name}/lb": mean_lb,
            f"{self.group_name}/se_ub": std_ub,
            f"{self.group_name}/se_lb": std_lb,
            f"{self.group_name}/policy_train_step": step,
        }

        if self.wandb_logger is not None:
            self.wandb_logger.log(pce_bound_dict)
            if self.plot_during_training:
                plot_dict: dict[str, Any] = {}
                if self.plot_fn is not None:
                    plot_dict, self.key = self.plot_fn(
                        self.key,
                        trainer.target,
                        trainer.policy_network,
                        trainer.training_state.params_ema,
                        self.n_plot_rollouts,
                    )
                    self.wandb_logger.log(
                        {
                            f"{self.group_name}/{name}": Image(fig)
                            for name, fig in plot_dict.items()
                        }
                    )
                    for fig in plot_dict.values():
                        plt.close(fig)
        else:
            self.log_hist.append(pce_bound_dict)

        t2 = time.time()
        eval_elapsed = t2 - t1

        self.log.info(
            f"Step: {step}/{int(trainer._num_iters)}, lb: {mean_lb:.2f}, train time/s: {elapsed:.2f}, eval time/s: {eval_elapsed:.2f}."
        )

        self.t0 = time.time()

    def on_train_end(self, trainer: "PolicyTrainer"):  # type: ignore[override]
        if self.suppress_on_train_end:
            return
        self.log.info("Evaluating final policy network")
        (
            self.mean_lb,
            self.mean_ub,
            self.std_lb,
            self.std_ub,
            self.key,
            self.batch_size,
        ) = self.policy_eval_fn(
            self.key,
            trainer.target,
            trainer.policy_network,
            trainer.training_state.params_ema,
            L=self.inner_samples,
            outer_samples=self.outer_samples,
            outer_batch_size=self.batch_size,
        )

        pce_bound_dict = {
            f"{self.group_name}/final_ub": self.mean_ub,
            f"{self.group_name}/final_lb": self.mean_lb,
            f"{self.group_name}/final_se_ub": self.std_ub,
            f"{self.group_name}/final_se_lb": self.std_lb,
            f"{self.group_name}/policy_train_step": int(trainer.step),
        }
        self.log.info(
            f"Final policy evaluation - lb: {self.mean_lb:.2f}, ub: {self.mean_ub:.2f}, se_lb: {self.std_lb:.2f}, se_ub: {self.std_ub:.2f}"
        )
        plot_dict: dict[str, Any] = {}
        if self.wandb_logger is not None:
            self.wandb_logger.log(pce_bound_dict)
            if self.plot_fn is not None:
                plot_dict, self.key = self.plot_fn(
                    self.key,
                    trainer.target,
                    trainer.policy_network,
                    trainer.training_state.params_ema,
                    self.n_plot_rollouts,
                )
                self.wandb_logger.log(
                    {
                        f"{self.group_name}/{name}": Image(fig)
                        for name, fig in plot_dict.items()
                    }
                )
                for fig in plot_dict.values():
                    plt.close(fig)
        else:
            self.log_hist.append(pce_bound_dict)
