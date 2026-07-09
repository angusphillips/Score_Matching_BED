from typing import Any

import flax.linen as nn
import jax
import optax
from jaxtyping import PRNGKeyArray as Key

from policy_learning.bed_models import BEDModel
from policy_learning.score_nets import ConservativeScoreNetwork
from policy_learning.trainers import PolicyTrainer, ScoreTrainer
from policy_learning.utils.func import noop
from policy_learning.utils.logging import instantiate


def _get_output_shape(
    score_network_cfg: dict[str, Any],
    target_dist: BEDModel,
    output_dim: int,
) -> int | tuple[int, int]:
    if "Transformer" in str(score_network_cfg.get("_target_", "")):
        return output_dim
    return (target_dist.T, output_dim)


def _resolve_policy_activation(
    target_dist: BEDModel,
) -> str:
    """Resolve the policy output activation from the model's parameterisation.

    Each BED model fixes its ``_policy_parameterisation`` in ``__init__``:
      * ``"no_param"`` -> ``"identity"`` (unbounded / linearly-scaled designs,
        e.g. location finding, gravimetry).
      * ``"x"`` -> ``"tanh"`` (policy emits box-constrained designs
        ``design_bounds * tanh(x)``, e.g. the dynamical systems).
    """
    # Models with a sequential (stateful) design constraint emit raw logits from
    # the policy; the constraint (e.g. stick-breaking) is applied inside the
    # reparam rollout, so the emitter activation must be the identity.
    if getattr(target_dist, "uses_sequential_design", False):
        return "identity"

    policy_param = target_dist._policy_parameterisation
    if policy_param == "no_param":
        return "identity"
    elif policy_param == "x":
        return "tanh"
    raise ValueError(
        f"Unknown policy_parameterisation: {policy_param!r}. Expected 'x' or 'no_param'."
    )


def _instantiate_score_network(
    score_network_cfg: dict[str, Any],
    target_dist: BEDModel,
    output_dim: int,
) -> nn.Module:
    target = str(score_network_cfg.get("_target_", ""))
    if target.endswith("ConservativeScoreNetwork"):
        base_cfg = score_network_cfg.get("base_score_network", {})
        base_output_dim = target_dist.d + target_dist.d_y
        base_output_shape = _get_output_shape(base_cfg, target_dist, base_output_dim)
        noise_sigma = target_dist.get_obvs_sigma()
        if noise_sigma is None:
            noise_sigma = jax.numpy.eye(target_dist.d_y)
        wrapper_kwargs = {k: v for k, v in score_network_cfg.items() if k != "_target_"}
        return ConservativeScoreNetwork(
            **wrapper_kwargs,
            output_shape=output_dim,
            base_output_shape=base_output_shape,
            x_dim=target_dist.d,
            y_dim=target_dist.d_y,
            T=target_dist.T,
            std_stats=target_dist.get_std_stats_dict(),
            noise_sigma=noise_sigma,
        )

    return instantiate(
        score_network_cfg,
        output_shape=_get_output_shape(score_network_cfg, target_dist, output_dim),
        x_dim=target_dist.d,
        y_dim=target_dist.d_y,
        T=target_dist.T,
        std_stats=target_dist.get_std_stats_dict(),
        noise_sigma=target_dist.get_obvs_sigma(),
    )  # type: ignore


def _build_score_network(
    cfg: dict[str, Any],
    target_dist: BEDModel,
) -> nn.Module:
    output_dim = target_dist.d + target_dist.d_y
    return _instantiate_score_network(
        dict(cfg["score_network"]),
        target_dist,
        output_dim,
    )


def _build_optimizer(
    score_cfg: dict[str, Any],
) -> tuple[optax.GradientTransformationExtraArgs, optax.Schedule, float]:
    """Return (optimizer, lr_schedule, grad_clip_norm) from a score config dict."""
    lr_schedule: optax.Schedule = instantiate(score_cfg["lr_schedule"])  # type: ignore
    grad_clip_norm = float(score_cfg.get("clip_grad_norm", 1.0))
    optim = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.scale_by_adam(),
        optax.scale_by_schedule(lr_schedule),
        optax.scale(-1.0),
    )
    return optim, lr_schedule, grad_clip_norm


def _score_design_sampler_cfg(
    cfg: dict[str, Any],
) -> tuple[str | None, dict[str, Any] | None]:
    """Extract the score network's design sampler type/kwargs from a config.

    Used both to configure the ScoreTrainer and to record on the PolicyTrainer so
    diagnostic callbacks can draw reference batches from the same sampler.
    """
    design_sampler_kwargs_cfg = cfg["score"].get("design_sampler_kwargs", None)
    design_sampler_kwargs = (
        None if design_sampler_kwargs_cfg is None else dict(design_sampler_kwargs_cfg)
    )
    return cfg["score"].get("design_sampler_type", None), design_sampler_kwargs


def load_score_trainer(
    key: Key,
    cfg: dict[str, Any],
    target_dist: BEDModel,
    callbacks: tuple[Any, ...] | list[Any] = (),
) -> tuple[ScoreTrainer, Key]:
    score_net = _build_score_network(cfg, target_dist)

    optim, lr_schedule, score_grad_clip_norm = _build_optimizer(cfg["score"])

    design_sampler_type, design_sampler_kwargs = _score_design_sampler_cfg(cfg)

    key, subkey = jax.random.split(key)
    score_trainer = ScoreTrainer(
        score_network=score_net,
        target_model=target_dist,
        optimizer=optim,
        lr_schedule=lr_schedule,
        key=subkey,
        callbacks=list(callbacks),
        grad_clip_norm=score_grad_clip_norm,
        y_score_loss_weight=cfg["score"].get("y_score_loss_weight", 1.0),
        design_sampler_type=design_sampler_type,
        design_sampler_kwargs=design_sampler_kwargs,
    )

    return score_trainer, key


def build_policy_network(
    target_dist: BEDModel,
    policy_network_cfg: dict[str, Any],
) -> nn.Module:
    """Instantiate the policy net with the output activation the model's
    ``_policy_parameterisation`` implies (``x`` -> ``design_bounds*tanh``;
    ``no_param`` / sequential-design -> identity).

    Single source of truth: every caller that builds a policy net for training or
    reload must go through this. Hand-instantiating the module instead silently takes
    its ``lambda x: x`` default, which mismatches the reload for ``x``-parameterised
    models (the dynamical systems) — the policy trains with no output squashing but
    reloads with ``tanh``.
    """
    activation_type = _resolve_policy_activation(target_dist)
    output_activation = target_dist.get_policy_output_activation(activation_type)
    return instantiate(
        policy_network_cfg,
        shape=(target_dist.T, target_dist.d),
        output_activation=output_activation,
    )  # type: ignore


def load_policy_trainer(
    key: Key,
    cfg: dict[str, Any],
    target_dist: BEDModel,
    grad_fn: Any = noop,
    callbacks: tuple[Any, ...] | list[Any] = (),
    policy_net: nn.Module | None = None,
) -> tuple[PolicyTrainer, Key]:
    if policy_net is None:
        policy_net = build_policy_network(target_dist, cfg["policy_network"])

    policy_grad_clip_norm = cfg["policy"].get("norm_clip", 1.0)
    policy_lr_schedule: optax.Schedule = instantiate(cfg["policy"]["lr_schedule"])  # type: ignore
    if policy_grad_clip_norm is not None:
        policy_optim = optax.chain(
            optax.clip_by_global_norm(float(policy_grad_clip_norm)),
            optax.scale_by_adam(
                b1=cfg["policy"].get("b1", 0.9),
                b2=cfg["policy"].get("b2", 0.999),
            ),
            optax.scale_by_schedule(policy_lr_schedule),
        )
    else:
        policy_optim = optax.chain(
            optax.scale_by_adam(
                b1=cfg["policy"].get("b1", 0.9),
                b2=cfg["policy"].get("b2", 0.999),
            ),
            optax.scale_by_schedule(policy_lr_schedule),
        )

    design_sampler_type, design_sampler_kwargs = _score_design_sampler_cfg(cfg)
    key, subkey = jax.random.split(key)
    policy_trainer = PolicyTrainer(
        policy_network=policy_net,  # type: ignore
        target_model=target_dist,
        grad_fn=grad_fn,
        optimizer=policy_optim,
        lr_schedule=policy_lr_schedule,
        key=subkey,
        callbacks=list(callbacks),
        grad_clip_norm=policy_grad_clip_norm,
        design_sampler_type=design_sampler_type,
        design_sampler_kwargs=design_sampler_kwargs,
    )

    return policy_trainer, key
