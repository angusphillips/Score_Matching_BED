import faulthandler
import logging
import os
import signal
import socket
import sys
import threading
import time

import hydra
import jax
import tqdm
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from policy_learning.bed_models import BEDModel
from policy_learning.eig_estimators import get_grad_estimate_fn
from policy_learning.evaluate_policies import clear_eig_fn_cache, eval_policy_bounds
from policy_learning.inference.ground_truth_scores import (
    MonteCarloPosteriorScore,
)
from policy_learning.loaders import load_policy_trainer, load_score_trainer
from policy_learning.plotting.policy_rollouts import get_rollout_plot_fn
from policy_learning.trainers import (
    BaseCallback,
    Checkpointer,
    CountParameters,
    LearningRateLogger,
    PolicyEIGBoundEvals,
    PolicyTrainingMetrics,
    ProgressBar,
    ScoreTrainingMetrics,
)
from policy_learning.utils.func import noop
from policy_learning.utils.jax import derive_key
from policy_learning.utils.logging import configure_logging, get_git_info, instantiate
from policy_learning.utils.mem_check import resolve_outer_batch_size

tqdm.tqdm.monitor_interval = 0
faulthandler.enable(file=sys.stdout)
faulthandler.register(signal.SIGUSR1, file=sys.stdout)


def run(cfg):
    """Main training function - separated from hydra decorator for flexibility."""
    # Configure logging: suppress info logs from orbax.checkpoint
    configure_logging(
        root_level=logging.INFO, suppress_loggers={"absl": logging.WARNING}
    )

    log = logging.getLogger(__name__)
    log.info("Starting up...")
    log.info(f"Hostname: {socket.gethostname()}")

    # Get hydra output directory (where configs and logs are saved)
    hydra_cfg = HydraConfig.get()
    hydra_output_dir = hydra_cfg.runtime.output_dir

    # Initialize wandb with dir pointing to hydra output directory
    wandb_kwargs = dict(cfg["wandb"])
    wandb_kwargs["dir"] = hydra_output_dir
    wandb_kwargs["reinit"] = "finish_previous"

    seed = int(cfg["seed"])

    def derive(*namespace):
        return derive_key(seed, *namespace)

    # Define the target distribution
    target_dist: BEDModel = instantiate(cfg["bed_model"])  # type: ignore

    base_run_name = wandb_kwargs.get("name") or "random_restarts"
    parent_group = None
    parent_run_name = None

    # Score training run: this is the parent run for the restart sweep.
    score_run_kwargs = dict(wandb_kwargs)
    score_run_kwargs.setdefault("job_type", "score_train")
    score_run_kwargs["name"] = f"{base_run_name}"

    with wandb.init(**score_run_kwargs, config=cfg) as score_run:
        score_run.config.update({"git": get_git_info()})
        run_dir = str(score_run.dir)
        log.info(f"Running in cwd:, {os.getcwd()}")
        log.info(f"Hydra output dir: {hydra_output_dir}")
        log.info(f"Score wandb log dir: {run_dir}")
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        score_ckpt_dir = os.path.join(ckpt_dir, "score")
        os.makedirs(score_ckpt_dir, exist_ok=True)

        # std_stats = target_dist.get_std_stats_dict()
        # log.info(f"Standardisation statistics: {str(std_stats)}")

        parent_group = score_run.group or score_run.id
        parent_run_name = score_run.name or score_run.id

        score_backend = cfg["policy"].get("score_backend", "network")
        if "score" in cfg["policy"]["grad_est_type"]:
            if score_backend == "mc_posterior":
                mc_score_estimator = MonteCarloPosteriorScore.from_config(
                    cfg.get("mc_posterior", {}),
                    target_dist,
                    derive("mc_score_estimator"),
                    n_post_samples=cfg.get("train_n_post_samples"),
                )
                data_score_fn, design_score_fn = (
                    mc_score_estimator.get_unbatched_score_fns()
                )
                log.info("Using MC posterior score backend for policy gradients.")
            elif score_backend == "random":

                class RandomScoreTrainer:
                    def __init__(self):
                        pass

                    def get_sep_score_fns(
                        self,
                    ):
                        def data_score_fn(y, xi, aux_data=None, key=None):
                            return jax.random.normal(key, y.shape)[0]

                        def design_score_fn(y, xi, aux_data=None, key=None):
                            return jax.random.normal(key, xi.shape)[0]

                        return data_score_fn, design_score_fn

                    def get_score_fn(self):
                        def joint_score_fn(y, xi, aux_data=None, key=None):
                            full = jax.numpy.concatenate([y, xi], axis=-1)
                            return jax.random.normal(key, full.shape)

                        return joint_score_fn

                score_trainer = RandomScoreTrainer()
                data_score_fn, design_score_fn = score_trainer.get_sep_score_fns()

            elif score_backend == "network":
                n_iters = int(cfg["score"]["num_iters"])

                # Score-training callbacks, one per task (see
                # policy_learning/trainers/README.md for the full inventory).
                if cfg["score"].get("checkpoint_freq") is not None:
                    score_checkpointer = Checkpointer(
                        path=score_ckpt_dir,
                        frequency=int(cfg["score"]["checkpoint_freq"]),
                    )
                else:
                    score_checkpointer = Checkpointer(
                        path=score_ckpt_dir,
                        checkpoint_at=cfg["score"]["eval_iters"],
                    )
                score_metrics = ScoreTrainingMetrics.from_config(
                    cfg, phase="score", wandb_logger=score_run, log=log
                )
                score_callbacks: list[BaseCallback] = [
                    CountParameters("score", score_run, log),
                    score_metrics,
                    score_checkpointer,
                    LearningRateLogger("score", score_run, log),
                    ProgressBar(desc="Score training"),
                ]

                score_trainer, _ = load_score_trainer(
                    derive("score_trainer"),
                    cfg,
                    target_dist,
                    callbacks=score_callbacks,
                )

                if cfg.get("score_reload_path", None) is not None:
                    score_trainer.callbacks = [score_metrics]
                    score_reload_path = cfg["score_reload_path"]
                    score_reload_step = cfg.get("score_reload_step", None)
                    score_trainer.load_from_checkpoint(
                        score_reload_path,
                        step=score_reload_step - 1
                        if score_reload_step is not None
                        else None,
                        log=log,
                    )
                    score_trainer._trigger_event("on_train_end")

                else:
                    score_trainer.train(int(n_iters), int(cfg["score"]["batch_size"]))

                data_score_fn, design_score_fn = score_trainer.get_unbatched_score_fns()
            else:
                raise ValueError(
                    "policy.score_backend must be one of ['network', 'mc_posterior'], "
                    f"got: {score_backend}"
                )

        elif "score" not in cfg["policy"]["grad_est_type"]:
            data_score_fn = noop
            design_score_fn = noop

    wandb.finish()

    assert parent_group is not None
    assert parent_run_name is not None

    ############### POLICY TRAINING ####################

    n_restarts = int(cfg.get("n_restarts", 1))
    log.info(f"Running {n_restarts} random restarts for policy training")

    # Store results from all restarts
    restart_results = []
    # No-OOM outer batch sizes, probed on first use and reused across
    # restarts (see resolve_outer_batch_size). "general" serves SPCE-like
    # estimators (training + the diagnostics' SPCE reference); "chunked"
    # serves grad_est_type="score_chunked".
    oom_cache: dict[str, int | None] = {"general": None, "chunked": None}

    for restart_id in range(n_restarts):
        restart_wandb_kwargs = dict(wandb_kwargs)
        restart_wandb_kwargs.pop("id", None)
        restart_wandb_kwargs.pop("resume", None)
        restart_wandb_kwargs["group"] = parent_group
        restart_wandb_kwargs["job_type"] = "policy_restart"
        restart_wandb_kwargs["name"] = f"{parent_run_name}_restart_{restart_id:02d}"

        with wandb.init(**restart_wandb_kwargs, config=cfg) as restart_run:
            restart_run.config.update({"git": get_git_info()})
            log.info(f"Starting restart {restart_id + 1}/{n_restarts}")
            clear_eig_fn_cache()
            restart_run_dir = str(restart_run.dir)
            log.info(f"Restart {restart_id} wandb log dir: {restart_run_dir}")

            # Create restart-specific checkpoint directory
            restart_ckpt_dir = os.path.join(
                restart_run_dir, "checkpoints", "policy", f"restart_{restart_id}"
            )
            os.makedirs(restart_ckpt_dir, exist_ok=True)

            policy_template_trainer, _ = load_policy_trainer(
                key=derive("policy_template_trainer", restart_id),
                cfg=cfg,
                target_dist=target_dist,
                grad_fn=noop,
            )
            policy_net = policy_template_trainer.policy_network

            # Single throwaway key for this restart's OOM-probe calls (probes are
            # read-only w.r.t. seeding — they just compile/run grad fns to find a
            # no-OOM batch size, so reusing one stable key is fine).
            oom_key = derive("oom_probe", restart_id)

            plot_fn = get_rollout_plot_fn(target_dist)

            # Policy-training callbacks, one per task (see
            # policy_learning/trainers/README.md for the full inventory).
            eig_bounds = PolicyEIGBoundEvals.from_config(
                cfg,
                derive("policy_cb", "eig_bounds", restart_id),
                policy_eval_fn=eval_policy_bounds,
                plot_fn=plot_fn,
                wandb_logger=restart_run,
                log=log,
            )
            policy_callbacks = [
                CountParameters("policy", restart_run, log),
                Checkpointer(
                    path=restart_ckpt_dir,
                    frequency=int(cfg["policy"]["checkpoint_freq"]),
                    max_to_keep=int(cfg["policy"].get("max_to_keep", 21)),
                ),
                PolicyTrainingMetrics.from_config(
                    cfg, wandb_logger=restart_run, log=log
                ),
                eig_bounds,
                ProgressBar(
                    desc=f"policy training (restart {restart_id})", notebook=False
                ),
            ]

            # Resolve outer_batch_size for the policy-training grad fn, sharing
            # the OOM-probe cache with the diagnostics suite:
            #   - "score":         vmap'd, outer_batch_size unused -> 0.
            #   - "score_chunked": "chunked" cache slot.
            #   - else (spce/nmc/mlmc/...): "general" cache slot.
            grad_est_type = cfg["policy"]["grad_est_type"]
            if grad_est_type == "score":
                max_outer_batch_size = 0  # vmap'd estimator; outer_batch_size unused
            else:
                max_outer_batch_size = resolve_outer_batch_size(
                    oom_key,
                    policy_template_trainer.training_state.params,
                    policy_net=policy_net,
                    bed_model=target_dist,
                    policy_cfg=cfg["policy"],
                    estimate_type=grad_est_type,
                    data_score_fn=data_score_fn,
                    design_score_fn=design_score_fn,
                    init_batch_size=int(cfg["policy"]["train_outer_batch_size"]),
                    cache=oom_cache,
                    slot="chunked" if grad_est_type == "score_chunked" else "general",
                    log=log,
                    label="policy_train",
                )

            # Now set up and train with the batch size we found
            grad_fn = get_grad_estimate_fn(
                policy_net=policy_net,
                bed_model=target_dist,
                data_score_fn=data_score_fn,
                design_score_fn=design_score_fn,
                outer_batch_size=max_outer_batch_size,
                N=int(cfg["policy"]["train_outer_samples"]),
                M=int(cfg["policy"]["train_inner_samples"]),
                estimate_type=cfg["policy"]["grad_est_type"],
            )

            policy_trainer, _ = load_policy_trainer(
                key=derive("policy_trainer", restart_id),
                cfg=cfg,
                target_dist=target_dist,
                grad_fn=grad_fn,
                callbacks=policy_callbacks
                + [LearningRateLogger("policy", restart_run, log)],
                policy_net=policy_net,
            )

            policy_trainer.train(int(cfg["policy"]["num_iters"]))

            # Log final bounds for this restart
            restart_result = {
                "restart_id": restart_id,
                "final_lb": eig_bounds.mean_lb,
                "final_ub": eig_bounds.mean_ub,
                "final_se_lb": eig_bounds.std_lb,
                "final_se_ub": eig_bounds.std_ub,
            }
            restart_results.append(restart_result)

            # Log to wandb
            restart_run.log(
                {
                    "restarts/restart_id": restart_id,
                    "restarts/final_lb": eig_bounds.mean_lb,
                    "restarts/final_ub": eig_bounds.mean_ub,
                    "restarts/final_se_lb": eig_bounds.std_lb,
                    "restarts/final_se_ub": eig_bounds.std_ub,
                }
            )

            log.info(
                f"Restart {restart_id} complete - lb: {eig_bounds.mean_lb:.4f}, "
                f"ub: {eig_bounds.mean_ub:.4f}, se_lb: {eig_bounds.std_lb:.4f}, "
                f"se_ub: {eig_bounds.std_ub:.4f}"
            )

        wandb.finish()

    best_lb = None
    best_restart_idx = None
    if restart_results:
        best_restart = max(restart_results, key=lambda r: r["final_lb"])
        best_restart_idx = best_restart["restart_id"]
        best_lb = best_restart["final_lb"]

    if best_restart_idx is not None and best_lb is not None:
        log.info(
            f"All {n_restarts} restarts complete. Best restart: {best_restart_idx} with lb: {best_lb:.4f}"
        )
    else:
        log.info(f"All {n_restarts} restarts complete. No restart results recorded.")

    print("[DEBUG] End of script reached — starting exit watchdog", flush=True)

    def _exit_watchdog(delay=300):
        def _dump():
            time.sleep(delay)
            print(f"\n[DEBUG] Exit taking >{delay}s — dumping stacks\n", flush=True)
            faulthandler.dump_traceback(file=sys.stdout)

        t = threading.Thread(target=_dump, daemon=True)
        t.start()

    _exit_watchdog(30)  # 30s after script end

    # Optional but useful diagnostics
    print("[DEBUG] Active threads before exit:", flush=True)
    for t in threading.enumerate():
        print(f"[DEBUG] Thread: {t.name}, daemon={t.daemon}", flush=True)

    print("[DEBUG] Starting cleanup", flush=True)

    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()

    print("[DEBUG] Cleanup complete", flush=True)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """
    Hydra entry point. Uses modular configs from config/ directory.

    Example usage:
        # Run with default experiment config
        python run_scripts/run_random_restarts.py

        # Run a specific experiment
        python run_scripts/run_random_restarts.py experiment=lf_policy

        # Override specific components
        python run_scripts/run_random_restarts.py policy_network=deterministic_rnn n_restarts=5
    """
    # Print config for confirmation
    print("Final config:\n")
    print(OmegaConf.to_yaml(cfg))

    # Convert to dict for compatibility with existing code
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    run(cfg_dict)


if __name__ == "__main__":
    main()
