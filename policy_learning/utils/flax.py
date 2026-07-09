# Optimisers and training loop


import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training import train_state
from jaxtyping import PRNGKeyArray as Key
from jaxtyping import PyTree


@struct.dataclass
class TrainState(train_state.TrainState):
    step: int
    rng: Key
    rff_freqs: PyTree | None
    params_ema: PyTree
    batch_stats: PyTree | None
    ema_rate: float
    last_loss: float
    cum_loss_since_last_log: float
    cum_loss_since_start: float


def create_train_state(module, init_key, optim, ema_rate=0.99, **init_kwargs):
    key, init_key = jax.random.split(init_key)
    init_variables = module.init(init_key, **init_kwargs)
    init_params = init_variables["params"]
    init_rff_freqs = init_variables.get("rff_freqs", None)
    init_batch_stats = init_variables.get("batch_stats", None)
    key, init_key = jax.random.split(key)
    return (
        TrainState.create(
            params=init_params,
            params_ema=init_params,
            rff_freqs=init_rff_freqs,
            batch_stats=init_batch_stats,
            ema_rate=ema_rate,
            apply_fn=module.apply,
            tx=optim,
            last_loss=0.0,
            cum_loss_since_last_log=0.0,
            cum_loss_since_start=0.0,
            rng=init_key,
        ),
        key,
    )


def get_train_step(loss_fn, grad_clip_norm: float | None = None, **kwargs):
    """Get train step for the given loss function.

    Params
    ------
    loss_fn: Callable taking (key, score_network, batch, **kwargs)"""

    @jax.jit
    def train_step(state: TrainState, batch):
        def loss_fn_partial(params, rff_freqs, batch_stats):
            return loss_fn(state.rng, params, rff_freqs, batch_stats, batch, **kwargs)

        value_and_grad_fn = jax.value_and_grad(loss_fn_partial, has_aux=True)
        (obj_value, (key, new_batch_stats, aux_metrics)), grads = value_and_grad_fn(
            state.params, state.rff_freqs, state.batch_stats
        )

        grad_norm = optax.global_norm(grads)
        updates, new_opt_state = state.tx.update(grads, state.opt_state)
        update_norm = optax.global_norm(updates)
        new_params = optax.apply_updates(state.params, updates)
        param_norm_after = optax.global_norm(new_params)

        if grad_clip_norm is None:
            clipping_applied = jnp.asarray(0.0, dtype=grad_norm.dtype)
        else:
            clipping_applied = jnp.asarray(
                grad_norm > jnp.asarray(grad_clip_norm, dtype=grad_norm.dtype),
                dtype=grad_norm.dtype,
            )

        new_params_ema = jax.tree_util.tree_map(
            lambda p_ema, p: p_ema * state.ema_rate + p * (1.0 - state.ema_rate),
            state.params_ema,
            new_params,
        )
        new_state = state.replace(
            params=new_params,
            params_ema=new_params_ema,
            batch_stats=new_batch_stats,
            opt_state=new_opt_state,
            rng=key,
            step=state.step + 1,
            last_loss=obj_value,
            cum_loss_since_last_log=state.cum_loss_since_last_log + obj_value,
            cum_loss_since_start=state.cum_loss_since_start + obj_value,
        )

        aux_metrics = {
            **aux_metrics,
            "grad_norm_preclip": grad_norm,
            "param_norm": param_norm_after,
            "update_norm": update_norm,
            "clip_applied": clipping_applied,
        }
        return new_state, aux_metrics

    return train_step


def get_train_step_gradfn(
    grad_fn,
    grad_clip_norm: float | None = None,
    **kwargs,
):
    """Get train step for the given loss gradient function.

    Params
    ------
    grad_fn: Callable taking (key, score_network, batch, **kwargs)"""

    @jax.jit
    def train_step(state: TrainState):
        grads, rng = grad_fn(state.rng, state.params, **kwargs)
        grad_norm = optax.global_norm(grads)
        updates, new_opt_state = state.tx.update(grads, state.opt_state)
        update_norm = optax.global_norm(updates)

        new_params = optax.apply_updates(state.params, updates)
        param_norm_after = optax.global_norm(new_params)

        if grad_clip_norm is None:
            clipping_applied = jnp.asarray(0.0, dtype=grad_norm.dtype)
        else:
            clipping_applied = jnp.asarray(
                grad_norm > jnp.asarray(grad_clip_norm, dtype=grad_norm.dtype),
                dtype=grad_norm.dtype,
            )

        new_params_ema = jax.tree_util.tree_map(
            lambda p_ema, p: p_ema * state.ema_rate + p * (1.0 - state.ema_rate),
            state.params_ema,
            new_params,
        )
        new_state = state.replace(
            params=new_params,
            params_ema=new_params_ema,
            opt_state=new_opt_state,
            rng=rng,
            step=state.step + 1,
        )

        aux_metrics = {
            "grad_norm_preclip": grad_norm,
            "param_norm": param_norm_after,
            "update_norm": update_norm,
            "clip_applied": clipping_applied,
        }
        return new_state, aux_metrics, grads

    return train_step
