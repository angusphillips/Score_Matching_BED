"""Variational posterior (var_post) training of policy and conditional posterior.

Trains both policy network and conditioned posterior MoG parameters on a
user-specified objective using simultaneous same-step updates.
"""

import logging
import os
import socket

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import wandb
from flax import linen as nn
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from policy_learning.bed_models import BEDModel
from policy_learning.utils.logging import get_git_info
from policy_learning.evaluate_policies import eval_policy_bounds
from policy_learning.inference.importance_sampler import sample_posterior_is
from policy_learning.var_post_objectives import (
    get_objective_fn,
    make_var_post_value_and_grad_fn,
)
from policy_learning.plotting.policy_rollouts import get_rollout_plot_fn
from policy_learning.utils.flax import create_train_state
from policy_learning.utils.logging import configure_logging, instantiate
from policy_learning.variational.conditional_posterior import (
    TransformerConditionalMoGPosterior as ConditionalPosterior,
)


def _count_tree_parameters(tree) -> int:
    """Count scalar parameters in a PyTree."""
    return int(sum(arr.size for arr in jax.tree_util.tree_leaves(tree)))


def _expand_theta_for_likelihood(
    theta_sample: jnp.ndarray, target_dist: BEDModel
) -> jnp.ndarray:
    """Convert a posterior theta sample to the shape expected by BEDModel.data_log_lik."""
    theta = theta_sample
    theta_shape = target_dist.theta_shape
    theta_flat_dim = target_dist.theta_dim

    if theta.shape[-1] == theta_flat_dim:
        theta = theta.reshape(*theta.shape[:-1], *theta_shape)

    if theta.ndim == len(theta_shape):
        theta = jnp.broadcast_to(theta[None, ...], (target_dist.T, *theta_shape))
    elif theta.ndim == len(theta_shape) + 1 and theta.shape[0] != target_dist.T:
        theta = jnp.broadcast_to(
            theta[:, None, ...], (theta.shape[0], target_dist.T, *theta_shape)
        )

    return theta


def _cloud_to_flat(theta_cloud: jnp.ndarray, target_dist: BEDModel) -> np.ndarray:
    """Flatten a theta cloud to ``[num_samples, theta_dim]``.

    Handles clouds with or without a time axis (theta is constant over T, so we
    take the first time step when present). Generic in ``theta_shape``.
    """
    cloud = jnp.asarray(theta_cloud)
    theta_trailing = len(target_dist.theta_shape)
    # An extra leading-of-theta axis means a time axis is present: drop it.
    if cloud.ndim == theta_trailing + 2:
        cloud = cloud[:, 0]
    return np.asarray(cloud.reshape(cloud.shape[0], -1))


def _sample_learned_posterior_cloud(
    key: jax.Array,
    posterior_net: nn.Module,
    posterior_params,
    posterior_static_vars,
    y: jnp.ndarray,
    x: jnp.ndarray,
    target_dist: BEDModel,
    num_samples: int,
) -> tuple[jnp.ndarray, jax.Array]:
    posterior_bound = posterior_net.bind(
        {
            "params": posterior_params,
            **posterior_static_vars,
        }
    )
    key, sample_seed = jax.random.split(key)
    sample_keys = jax.random.split(sample_seed, num_samples)

    def _sample_one(sample_key):
        # The posterior models the unconstrained latent z; map back to theta
        # space so the cloud is comparable to the MC posterior and likelihood.
        z_flat = posterior_bound.sample(sample_key, y[None, ...], x[None, ...])[0]
        theta_flat = target_dist.theta_transform(z_flat)
        return _expand_theta_for_likelihood(theta_flat, target_dist)

    theta_cloud = jax.vmap(_sample_one)(sample_keys)
    return theta_cloud, key


def _sample_mc_posterior_cloud(
    key: jax.Array,
    target_dist: BEDModel,
    y: jnp.ndarray,
    x: jnp.ndarray,
    aux_data: dict,
    num_samples: int,
    mala_steps: int,
    mala_step_size: float,
) -> tuple[jnp.ndarray, jax.Array]:
    @jax.jit
    def unbatched_log_lik(theta, y_cond=y, xi_cond=x, aux_data_cond=aux_data):
        theta_broadcast = jnp.broadcast_to(
            jnp.expand_dims(theta, 0),
            (target_dist.T, *target_dist.theta_shape),
        )
        return target_dist.log_lik(
            y_cond, xi_cond, theta_broadcast, aux_data_cond or {}
        )

    def prior_sampler(key, batch_shape):
        key, subkey = jax.random.split(key)
        return target_dist.theta_transform(
            jax.random.normal(subkey, (*batch_shape, *target_dist.theta_shape))
        ), key

    posterior_cloud, key, _ = sample_posterior_is(
        key,
        unbatched_log_lik,
        prior_sampler,
        target_dist.theta_shape,
        (num_samples * 100,),
        mala_steps=mala_steps,
        mala_step_size=mala_step_size,
    )
    key, subkey = jax.random.split(key)
    inds = jax.random.choice(
        subkey, jnp.arange(posterior_cloud.shape[0]), shape=(num_samples,)
    )
    return posterior_cloud[inds], key


def _make_posterior_comparison_figure(
    target_dist: BEDModel,
    theta_true: jnp.ndarray,
    xT: jnp.ndarray,
    learned_cloud: jnp.ndarray,
    mc_cloud: jnp.ndarray,
):
    """Per-dimension marginal histograms: variational vs MC posterior vs truth.

    Generic in ``theta_dim`` (unlike the old 2D scatter, which only made sense
    for LocationFinding). ``xT`` is accepted for caller compatibility but the
    design path is not meaningful across all models so it is omitted.
    """
    del xT  # design-path overlay dropped; not generic across models

    learned_flat = _cloud_to_flat(learned_cloud, target_dist)
    mc_flat = _cloud_to_flat(mc_cloud, target_dist)
    theta_true_flat = np.asarray(jnp.asarray(theta_true)[0].reshape(-1))

    theta_dim = learned_flat.shape[-1]
    ncols = min(4, theta_dim)
    nrows = int(np.ceil(theta_dim / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False
    )
    axes_flat = axes.ravel()

    for d in range(theta_dim):
        ax = axes_flat[d]
        both = np.concatenate([learned_flat[:, d], mc_flat[:, d]])
        bins = np.histogram_bin_edges(both, bins=40)
        ax.hist(
            mc_flat[:, d],
            bins=bins,
            density=True,
            alpha=0.45,
            color="tab:orange",
            label="MC posterior",
        )
        ax.hist(
            learned_flat[:, d],
            bins=bins,
            density=True,
            alpha=0.45,
            color="tab:blue",
            label="Variational posterior",
        )
        if d < theta_true_flat.shape[0]:
            ax.axvline(
                float(theta_true_flat[d]),
                color="tab:red",
                linewidth=1.5,
                label="true theta",
            )
        ax.set_title(f"theta[{d}]")
        if d == 0:
            ax.legend(loc="best", fontsize=8)

    for d in range(theta_dim, len(axes_flat)):
        axes_flat[d].axis("off")

    fig.tight_layout()
    return fig


def _log_var_post_diagnostics(
    key: jax.Array,
    run,
    log: logging.Logger,
    target_dist: BEDModel,
    policy_net: nn.Module,
    policy_train_state,
    posterior_net: nn.Module,
    posterior_params,
    posterior_static_vars,
    cfg: DictConfig,
    eval_outer_batch_size: int,
    training_iter: int,
) -> tuple[jax.Array, int]:
    eval_cfg = cfg["var_post"]
    posterior_only_random_policy = bool(
        eval_cfg.get("posterior_only_random_policy", False)
    )
    random_policy_std = float(eval_cfg.get("posterior_only_random_policy_std", 1.0))

    if posterior_only_random_policy:
        lb = jnp.asarray(jnp.nan)
        ub = jnp.asarray(jnp.nan)
        se_lb = jnp.asarray(jnp.nan)
        se_ub = jnp.asarray(jnp.nan)
    else:
        lb, ub, se_lb, se_ub, key, eval_outer_batch_size = eval_policy_bounds(
            key,
            target_dist,
            policy_net,
            policy_train_state.params_ema,
            L=int(eval_cfg.get("eval_inner_samples", 512)),
            outer_samples=int(eval_cfg.get("eval_outer_samples", 64)),
            outer_batch_size=eval_outer_batch_size,
            reuse_inner_samples=bool(eval_cfg.get("reuse_inner_samples", False)),
        )

    theta_true, key = target_dist.sample_prior(key, ())
    if posterior_only_random_policy:
        policy_epsilon, key = target_dist.sample_policy_epsilon(key, ())
        xT = random_policy_std * policy_epsilon
        yT, aux_data, key = target_dist.sample_data(key, xT, theta_true)
    else:
        epsilon, key = target_dist.sample_epsilon(key, ())
        policy_epsilon, key = target_dist.sample_policy_epsilon(key, ())
        yT, xT, aux_data = target_dist.reparam_sample(
            policy_train_state.params_ema,
            policy_net,
            theta=theta_true,
            epsilon=epsilon,
            policy_epsilon=policy_epsilon,
        )

    learned_cloud, key = _sample_learned_posterior_cloud(
        key,
        posterior_net,
        posterior_params,
        posterior_static_vars,
        yT,
        xT,
        target_dist,
        int(eval_cfg.get("posterior_plot_samples", 256)),
    )
    mc_cloud, key = _sample_mc_posterior_cloud(
        key,
        target_dist,
        yT,
        xT,
        aux_data,
        int(eval_cfg.get("mc_posterior_plot_samples", 256)),
        int(eval_cfg.get("mc_posterior_mala_steps", 5)),
        float(eval_cfg.get("mc_posterior_mala_step_size", 0.1)),
    )

    fig = _make_posterior_comparison_figure(
        target_dist=target_dist,
        theta_true=theta_true,
        xT=xT,
        learned_cloud=learned_cloud,
        mc_cloud=mc_cloud,
    )

    eval_log: dict = {
        "eval/training_iter": training_iter,
        "eval/policy/lb": lb,
        "eval/policy/ub": ub,
        "eval/policy/se_lb": se_lb,
        "eval/policy/se_ub": se_ub,
        "eval/posterior_comparison": wandb.Image(fig),  # type: ignore
    }

    # Policy rollout plots (skip in posterior-only mode: no real policy yet).
    if not posterior_only_random_policy:
        plot_fn = get_rollout_plot_fn(target_dist)
        if plot_fn is not None:
            n_plot_rollouts = int(eval_cfg.get("n_plot_rollouts", 3))
            plot_dict, key = plot_fn(  # type: ignore[arg-type]
                key,
                target_dist,  # type: ignore
                policy_net,
                policy_train_state.params_ema,
                n_plot_rollouts,
            )
            for name, rollout_fig in plot_dict.items():
                eval_log[f"eval/policy_rollouts/{name}"] = wandb.Image(rollout_fig)  # type: ignore
                plt.close(rollout_fig)

    run.log(eval_log)
    plt.close(fig)
    log.info(
        "Eval @ iter=%d bounds lb=%.4f ub=%.4f se_lb=%.4f se_ub=%.4f",
        training_iter,
        float(lb),
        float(ub),
        float(se_lb),
        float(se_ub),
    )
    return key, int(eval_outer_batch_size)


def run(cfg):
    """Main var_post training function."""
    configure_logging(
        root_level=logging.INFO, suppress_loggers={"absl": logging.WARNING}
    )

    log = logging.getLogger(__name__)
    log.info("Starting var_post policy-posterior training...")
    log.info(f"Hostname: {socket.gethostname()}")

    hydra_cfg = HydraConfig.get()
    hydra_output_dir = hydra_cfg.runtime.output_dir

    wandb_kwargs = dict(cfg["wandb"])
    wandb_kwargs["dir"] = hydra_output_dir

    with wandb.init(**wandb_kwargs, config=cfg) as run:
        run.config.update({"git": get_git_info()})
        run_dir = str(run.dir)
        log.info(f"Running in cwd: {os.getcwd()}")
        log.info(f"Hydra output dir: {hydra_output_dir}")
        log.info(f"Wandb log dir: {run_dir}")

        ckpt_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        policy_ckpt_dir = os.path.join(ckpt_dir, "policy")
        posterior_ckpt_dir = os.path.join(ckpt_dir, "posterior")
        os.makedirs(policy_ckpt_dir, exist_ok=True)
        os.makedirs(posterior_ckpt_dir, exist_ok=True)

        key = jax.random.key(cfg["seed"])

        target_dist: BEDModel = instantiate(cfg["bed_model"])  # type: ignore
        log.info(f"Target distribution: {cfg['bed_model']['_target_']}")

        policy_net: nn.Module = instantiate(
            cfg["policy_network"],
            shape=(target_dist.T, target_dist.d),
        )  # type: ignore
        log.info(f"Policy network: {cfg['policy_network']['_target_']}")

        key, subkey = jax.random.split(key)
        y_init, xi_init, _, _, subkey = target_dist.sample(subkey, (10,))
        policy_epsilon_init, subkey = target_dist.sample_policy_epsilon(subkey, (10,))

        policy_lr_schedule: optax.Schedule = instantiate(
            cfg["var_post"]["policy_lr_schedule"]
        )  # type: ignore
        policy_optim = optax.chain(
            optax.clip_by_global_norm(cfg["var_post"].get("policy_grad_clip", 1.0)),
            optax.scale_by_adam(
                b1=cfg["var_post"].get("policy_b1", 0.9),
                b2=cfg["var_post"].get("policy_b2", 0.999),
            ),
            optax.scale_by_schedule(policy_lr_schedule),
        )

        key, subkey = jax.random.split(key)
        policy_train_state, key = create_train_state(
            policy_net,
            subkey,
            policy_optim,
            y=y_init,
            x=xi_init,
            t=jnp.ones((10,)),
            z=policy_epsilon_init,
        )

        posterior_net = ConditionalPosterior(
            dim_theta=target_dist.theta_dim,
            theta_shape=target_dist.theta_shape,
            num_components=cfg["var_post"]["posterior_num_components"],
            embedding_dim=cfg["var_post"]["posterior_embedding_dim"],
            embed_mlp_dim=cfg["var_post"]["posterior_embed_mlp_dim"],
            model_dim=cfg["var_post"]["posterior_model_dim"],
            num_layers=cfg["var_post"]["posterior_num_layers"],
            num_heads=cfg["var_post"]["posterior_num_heads"],
            x_dim=target_dist.d,
            y_dim=target_dist.d_y,
            T=target_dist.T,
            std_stats=target_dist.get_std_stats_dict(),
            rff_sigma=cfg["var_post"].get("posterior_rff_sigma", 1.0),
            positional=cfg["var_post"].get("posterior_positional", False),
            canonicalize_sources=cfg["var_post"].get(
                "posterior_canonicalize_sources", False
            ),
        )
        log.info(
            "Posterior network: ConditionalMoGPosterior with %s components",
            cfg["var_post"]["posterior_num_components"],
        )

        key, subkey = jax.random.split(key)
        posterior_variables = posterior_net.init(subkey, y_init, xi_init)
        posterior_params = posterior_variables["params"]
        posterior_static_vars = {
            k: v for k, v in posterior_variables.items() if k != "params"
        }
        posterior_num_params = _count_tree_parameters(posterior_params)
        log.info(
            "Posterior variational approximation params: trainable=%d",
            posterior_num_params,
        )
        run.summary["posterior/num_params"] = posterior_num_params

        posterior_lr_schedule: optax.Schedule = instantiate(
            cfg["var_post"]["posterior_lr_schedule"]
        )  # type: ignore
        posterior_optim = optax.chain(
            optax.clip_by_global_norm(cfg["var_post"].get("posterior_grad_clip", 1.0)),
            optax.scale_by_adam(
                b1=cfg["var_post"].get("posterior_b1", 0.9),
                b2=cfg["var_post"].get("posterior_b2", 0.999),
            ),
            optax.scale_by_schedule(posterior_lr_schedule),
        )

        posterior_opt_state = posterior_optim.init(posterior_params)

        objective_name = cfg["var_post"]["objective"]
        batch_size = int(cfg["var_post"]["batch_size"])
        posterior_samples = int(cfg["var_post"].get("posterior_samples", 1))
        posterior_only_random_policy = bool(
            cfg["var_post"].get("posterior_only_random_policy", False)
        )
        posterior_only_random_policy_std = float(
            cfg["var_post"].get("posterior_only_random_policy_std", 1.0)
        )
        log.info(f"Using objective: {objective_name}")
        if posterior_only_random_policy:
            log.info(
                "Posterior-only diagnosis mode enabled: using random Gaussian designs with std=%.3f and freezing policy updates.",
                posterior_only_random_policy_std,
            )
        objective_fn = get_objective_fn(objective_name)
        value_and_grad_fn = make_var_post_value_and_grad_fn(objective_fn)

        @jax.jit
        def var_post_step(
            policy_train_state,
            posterior_params,
            posterior_opt_state,
            key,
        ):
            (loss, metrics), (grad_policy, grad_posterior), key = value_and_grad_fn(
                policy_train_state.params,
                posterior_params,
                posterior_static_vars,
                policy_net,
                posterior_net,
                target_dist,
                key,
                {
                    "batch_size": batch_size,
                    "posterior_samples": posterior_samples,
                    "posterior_only_random_policy": posterior_only_random_policy,
                    "posterior_only_random_policy_std": posterior_only_random_policy_std,
                },
            )

            policy_grad_norm = optax.global_norm(grad_policy)
            if posterior_only_random_policy:
                policy_opt_state_new = policy_train_state.opt_state
                policy_params_new = policy_train_state.params
                policy_params_ema_new = policy_train_state.params_ema
            else:
                policy_updates, policy_opt_state_new = policy_optim.update(
                    grad_policy, policy_train_state.opt_state, policy_train_state.params
                )
                policy_params_new = optax.apply_updates(
                    policy_train_state.params, policy_updates
                )
                policy_params_ema_new = jax.tree_util.tree_map(
                    lambda ema, param: ema * policy_train_state.ema_rate
                    + param * (1.0 - policy_train_state.ema_rate),
                    policy_train_state.params_ema,
                    policy_params_new,
                )

            posterior_grad_norm = optax.global_norm(grad_posterior)
            posterior_updates, posterior_opt_state_new = posterior_optim.update(
                grad_posterior, posterior_opt_state, posterior_params
            )
            posterior_params_new = optax.apply_updates(
                posterior_params, posterior_updates
            )

            new_policy_train_state = policy_train_state.replace(
                params=policy_params_new,
                params_ema=policy_params_ema_new,
                opt_state=policy_opt_state_new,
                step=policy_train_state.step + 1,
                rng=key,
                last_loss=loss,
                cum_loss_since_last_log=policy_train_state.cum_loss_since_last_log
                + loss,
                cum_loss_since_start=policy_train_state.cum_loss_since_start + loss,
            )

            all_metrics = {
                **metrics,
                "loss": loss,
                "policy_grad_norm": policy_grad_norm,
                "posterior_grad_norm": posterior_grad_norm,
            }

            return (
                new_policy_train_state,
                posterior_params_new,
                posterior_opt_state_new,
                all_metrics,
                key,
            )

        num_iters = int(cfg["var_post"]["num_iters"])
        log_freq = int(cfg["var_post"].get("log_freq", 100))
        eval_freq = int(cfg["var_post"].get("eval_freq", log_freq))
        eval_outer_batch_size = int(cfg["var_post"].get("eval_outer_batch_size", 64))

        key, eval_outer_batch_size = _log_var_post_diagnostics(
            key,
            run,
            log,
            target_dist,
            policy_net,
            policy_train_state,
            posterior_net,
            posterior_params,
            posterior_static_vars,
            cfg,
            eval_outer_batch_size,
            training_iter=0,
        )

        log.info(f"Training for {num_iters} iterations")
        for step in range(num_iters):
            policy_train_state, posterior_params, posterior_opt_state, metrics, key = (
                var_post_step(
                    policy_train_state,
                    posterior_params,
                    posterior_opt_state,
                    key,
                )
            )

            completed_iters = step + 1
            if step % log_freq == 0:
                log.info(f"Step {step}: loss={metrics['loss']:.6f}")
                wandb.log({**metrics, "training_iter": completed_iters})

            if eval_freq > 0 and completed_iters % eval_freq == 0:
                key, eval_outer_batch_size = _log_var_post_diagnostics(
                    key,
                    run,
                    log,
                    target_dist,
                    policy_net,
                    policy_train_state,
                    posterior_net,
                    posterior_params,
                    posterior_static_vars,
                    cfg,
                    eval_outer_batch_size,
                    training_iter=completed_iters,
                )

            # if ckpt_freq > 0 and step % ckpt_freq == 0:
            #     log.info(f"Saving checkpoint at step {step}")

        log.info("Training complete!")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """
    Hydra entry point. Uses modular configs from config/ directory.

    Example usage:
        # Run with a variational-posterior experiment config
        python run_scripts/run_var_ba.py experiment=location_finding_var_post

        # Override nested values
        python run_scripts/run_var_ba.py experiment=location_finding_var_post policy.num_iters=10000 seed=42

        # Other tasks
        python run_scripts/run_var_ba.py experiment=cart_pole_var_post
    """
    # Print config for confirmation
    print("Final config:\n")
    print(OmegaConf.to_yaml(cfg))

    # Convert to dict for compatibility with existing code
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    run(cfg_dict)


if __name__ == "__main__":
    main()
