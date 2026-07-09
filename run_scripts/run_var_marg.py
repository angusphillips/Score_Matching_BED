import logging
import os
import socket

import hydra
import jax
import optax
import wandb
from flax import linen as nn
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from policy_learning.bed_models import BEDModel
from policy_learning.eig_estimators import get_grad_estimate_fn
from policy_learning.evaluate_policies import eval_policy_bounds
from policy_learning.plotting.policy_rollouts import plot_location_finding_rollout
from policy_learning.trainers import (
    Checkpointer,
    CountParameters,
    LearningRateLogger,
    PolicyEIGBoundEvals,
    PolicyTrainer,
    ProgressBar,
)
from policy_learning.utils.func import noop
from policy_learning.utils.logging import get_git_info, instantiate
from policy_learning.utils.mem_check import get_maximal_outer_batch_size
from policy_learning.variational import (
    EquinoxCheckpointer,
    VariationalLossLogger,
    VariationalSamplePlot,
    VariationalTrainer,
    create_flow_model,
)


def run(cfg):
    """Main training function - separated from hydra decorator for flexibility."""
    log = logging.getLogger(__name__)
    log.info("Starting up...")
    log.info(f"Hostname: {socket.gethostname()}")

    # Get hydra output directory (where configs and logs are saved)
    hydra_cfg = HydraConfig.get()
    hydra_output_dir = hydra_cfg.runtime.output_dir

    # Initialize wandb with dir pointing to hydra output directory
    wandb_kwargs = dict(cfg["wandb"])
    wandb_kwargs["dir"] = hydra_output_dir

    with wandb.init(**wandb_kwargs, config=cfg) as run:
        run.config.update({"git": get_git_info()})
        run_dir = run.dir
        log.info(f"Running in cwd:, {os.getcwd()}")
        log.info(f"Hydra output dir: {hydra_output_dir}")
        log.info(f"Wandb log dir: {run_dir}")
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        var_ckpt_dir = os.path.join(ckpt_dir, "var")
        policy_ckpt_dir = os.path.join(ckpt_dir, "policy")
        os.makedirs(var_ckpt_dir, exist_ok=True)
        os.makedirs(policy_ckpt_dir, exist_ok=True)

        # Jax key iterator
        key = jax.random.key(cfg["seed"])

        # Define the target distribution
        target_dist: BEDModel = instantiate(cfg["bed_model"])  # type: ignore

        ######################## VARIATIONAL ################################
        if "var" in cfg["policy"]["grad_est_type"]:
            # Get variational network configuration
            key, subkey = jax.random.split(key)
            variational_dist = create_flow_model(
                key=subkey,
                dim_y=target_dist.d_y,
                dim_xi=target_dist.d,
                T=target_dist.T,
                **cfg["variational"]["dist"],
            )
            var_cfg = cfg["variational"]

            # Define optimizer
            lr_schedule: optax.Schedule = instantiate(var_cfg["lr_schedule"])  # type: ignore
            var_grad_clip_norm = var_cfg.get("grad_clip_norm", 1.0)
            optimizer_transforms = []
            if var_grad_clip_norm is not None:
                optimizer_transforms.append(
                    optax.clip_by_global_norm(float(var_grad_clip_norm))
                )

            optimizer_transforms.extend(
                [
                    optax.scale_by_adam(),
                    optax.scale_by_schedule(lr_schedule),
                    optax.scale(-1),
                ]
            )
            var_optimizer = optax.chain(*optimizer_transforms)

            # Setup callbacks
            param_counter = CountParameters("variational", run, log)
            checkpointer = EquinoxCheckpointer(
                path=var_ckpt_dir,
                checkpoint_at=cfg["variational"].get("eval_iters", None),
                frequency=cfg["variational"].get("checkpoint_freq", None),
            )
            loss_logger = VariationalLossLogger(
                log_freq=int(var_cfg.get("log_freq", 100)),
                eval_batch_size=int(var_cfg.get("eval_batch_size", 256)),
                wandb_logger=run,
                log=log,
            )
            sample_plot = VariationalSamplePlot(
                log_freq=int(var_cfg.get("plot_freq", 1000)),
                num_samples=5000,
                wandb_logger=run,
                log=log,
            )
            progress_bar = ProgressBar(desc="variational training")

            var_reload_path = cfg.get("var_reload_path", None)
            if var_reload_path is not None:
                callbacks: list = [loss_logger]
            else:
                callbacks = [
                    param_counter,
                    checkpointer,
                    loss_logger,
                    sample_plot,
                    progress_bar,
                ]

            # Create trainer
            key, subkey = jax.random.split(key)
            trainer = VariationalTrainer(
                variational_dist=variational_dist,
                target_model=target_dist,
                optimizer=var_optimizer,
                lr_schedule=lr_schedule,
                loss_type=var_cfg["loss"],
                key=subkey,
                callbacks=callbacks,
                grad_clip_norm=(
                    float(var_grad_clip_norm)
                    if var_grad_clip_norm is not None
                    else None
                ),
            )

            if var_reload_path is not None:
                var_reload_step = cfg.get("var_reload_step", None)
                reload_checkpointer = EquinoxCheckpointer(
                    path=var_reload_path,
                    frequency=1,
                )
                reload_checkpointer.restore(
                    trainer,
                    step=var_reload_step - 1 if var_reload_step is not None else None,
                )
                trainer._trigger_event("on_train_end")
            else:
                # Train
                num_iters = int(var_cfg.get("num_iters", 10000))
                batch_size = int(var_cfg.get("batch_size", 256))

                log.info(
                    f"Training for {num_iters} iterations with batch size {batch_size}"
                )
                trainer.train(num_iters=num_iters, batch_size=batch_size)

            # Get trained score functions
            data_score_fn, design_score_fn = trainer.get_unbatched_score_fns()
            log.info("Training complete. Score functions ready for EIG estimation.")
        else:
            data_score_fn, design_score_fn = noop, noop
        ############### POLICY TRAINING ####################

        # Define policy network
        policy_net: nn.Module = instantiate(
            cfg["policy_network"],
            shape=(target_dist.T, target_dist.d),
        )  # type: ignore

        # Define policy optimiser
        policy_lr_schedule: optax.Schedule = instantiate(cfg["policy"]["lr_schedule"])  # type: ignore
        policy_optim = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.scale_by_adam(
                b1=cfg["policy"].get("b1", 0.9), b2=cfg["policy"].get("b2", 0.999)
            ),
            optax.scale_by_schedule(policy_lr_schedule),
        )
        param_counter = CountParameters("policy", run, log)
        key, eig_bounds_key = jax.random.split(key)
        eig_bounds = PolicyEIGBoundEvals.from_config(
            cfg,
            eig_bounds_key,
            policy_eval_fn=eval_policy_bounds,
            plot_fn=plot_location_finding_rollout,
            wandb_logger=run,
            log=log,
        )
        pb = ProgressBar(desc="policy training", notebook=False)
        policy_checkpointer = Checkpointer(
            path=policy_ckpt_dir, frequency=int(cfg["policy"]["num_iters"]) // 10
        )
        key, subkey = jax.random.split(key)

        # Figure out maximum outer batch size to guard against memory issues
        dummy_grad_fn = get_grad_estimate_fn(
            policy_net=policy_net,
            bed_model=target_dist,
            data_score_fn=data_score_fn,
            design_score_fn=design_score_fn,
            outer_batch_size=int(cfg["policy"]["train_outer_batch_size"]),
            N=int(cfg["policy"]["train_outer_samples"]),
            M=int(cfg["policy"]["train_inner_samples"]),
            estimate_type=cfg["policy"]["grad_est_type"],
        )
        dummy_policy_trainer = PolicyTrainer(
            policy_network=policy_net,
            target_model=target_dist,
            grad_fn=dummy_grad_fn,
            optimizer=policy_optim,
            lr_schedule=policy_lr_schedule,
            key=subkey,
            callbacks=[param_counter, policy_checkpointer, eig_bounds, pb],
        )

        def partial_grad_fn(outer_batch_size):
            return get_grad_estimate_fn(
                policy_net=policy_net,
                bed_model=target_dist,
                data_score_fn=data_score_fn,
                design_score_fn=design_score_fn,
                outer_batch_size=outer_batch_size,
                N=int(cfg["policy"]["train_outer_samples"]),
                M=int(cfg["policy"]["train_inner_samples"]),
                estimate_type=cfg["policy"]["grad_est_type"],
            )

        max_outer_batch_size = get_maximal_outer_batch_size(
            key,
            dummy_policy_trainer.training_state.params,
            partial_grad_fn,
            int(cfg["policy"]["train_outer_batch_size"]),
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

        policy_trainer = PolicyTrainer(
            policy_network=policy_net,
            target_model=target_dist,
            grad_fn=grad_fn,
            optimizer=policy_optim,
            lr_schedule=policy_lr_schedule,
            key=subkey,
            callbacks=[
                param_counter,
                policy_checkpointer,
                eig_bounds,
                LearningRateLogger("policy", run, log),
                pb,
            ],
        )

        policy_trainer.train(int(cfg["policy"]["num_iters"]))


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="config",
)
def main(cfg: DictConfig):
    """
    Hydra entry point. Uses modular configs from config/ directory.

    Example usage:
        # Run with a variational-marginal experiment config
        python run_scripts/run_var_marg.py experiment=location_finding_var_marg

        # Override specific components
        python run_scripts/run_var_marg.py experiment=location_finding_var_marg policy.num_iters=10000 seed=42
    """
    # Print config for confirmation
    print("Final config:\n")
    print(OmegaConf.to_yaml(cfg))

    # Convert to dict for compatibility with existing code
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    run(cfg_dict)


if __name__ == "__main__":
    main()
