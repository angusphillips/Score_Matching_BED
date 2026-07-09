import flax.linen as nn
import jax
import jax.numpy as jnp
from check_shapes import check_shapes
from jaxtyping import Array, PyTree
from jaxtyping import PRNGKeyArray as Key

from policy_learning.bed_models import BEDModel


def get_score_matching_loss(
    score_net: nn.Module,
    bed_model: BEDModel,
    y_score_loss_weight: float = 1.0,
):
    """Marginal score-matching loss.

    Fits the network to the analytic conditional score ``∇_z log p(y|xi,θ)`` of the
    marginal (over θ), split into a y-component and an xi-component. The y-component
    is weighted by the scalar ``y_score_loss_weight`` (a task-dependent balance knob
    so the y- and xi-loss magnitudes are comparable); the xi-component is
    unit-weighted. Both components use the standard squared (L2) residual.
    """

    @check_shapes(
        "batch[0]: [*batch, T, d_y]",
        "batch[1]: [*batch, T, d]",
        "batch[2]: [*batch, T, *theta_shape]",
        "return[0]: []",
    )
    def loss_fn_marginal(
        key: Key,
        params: PyTree,
        rff_freqs: PyTree,
        batch_stats: PyTree | None,
        batch: tuple[Array, Array, Array, Array],
        train: bool = True,
    ):
        """Marginal score matching."""
        y, xi, theta, aux_data = batch
        d_y = y.shape[-1]
        d_x = xi.shape[-1]
        T = y.shape[-2]

        # Apply score network.
        variables = (
            {"params": params, "rff_freqs": rff_freqs, "batch_stats": batch_stats}
            if batch_stats is not None
            else (
                {"params": params, "rff_freqs": rff_freqs}
                if rff_freqs is not None
                else {"params": params}
            )
        )
        score, updates = score_net.apply(
            variables,
            y=y,
            xi=xi,
            mask=None,
            sigma=jnp.ones(y.shape[:-2], dtype=y.dtype),
            aux_data=aux_data,
            train=train,
            mutable=["batch_stats"] if train and batch_stats is not None else [],
        )

        # Analytic conditional-score target ∇_z log p(y|xi,θ).
        z = jnp.concatenate([y, xi], axis=-1)
        if not aux_data:
            aux_data = jnp.zeros((z.shape[0],))

        def log_lik_unbatched(z, theta, aux_data):
            y_in = z[:, :d_y]
            x_in = z[:, d_y:]
            return bed_model.data_log_lik(y_in, x_in, theta, aux_data)

        grad_log_lik = jax.vmap(
            lambda z_, theta_, aux_data_: jax.grad(
                lambda z: log_lik_unbatched(z, theta_, aux_data_)
            )(z_)
        )(z, theta, aux_data)

        residual = score - grad_log_lik  # [*batch, T, d_y + d_x]
        residual_y = residual[..., :d_y]
        residual_x = residual[..., d_y:]

        # Per-(b, t) summed-over-d_y squared y residual.
        ey_sq_per_step = jnp.sum(jnp.square(residual_y), axis=-1)  # [*batch, T]

        per_sample_y = (
            jnp.sum(y_score_loss_weight * ey_sq_per_step, axis=-1) / (T * d_y)
        )
        per_sample_y_unw = jnp.sum(ey_sq_per_step, axis=-1) / (T * d_y)
        per_sample_x = jnp.sum(jnp.square(residual_x), axis=(-1, -2)) / (T * d_x)

        loss_y_component = jnp.mean(per_sample_y)
        loss_x_component = jnp.mean(per_sample_x)
        loss = loss_y_component + loss_x_component

        aux_metrics = {
            "loss_total": loss,
            "loss_y_component": loss_y_component,  # weighted, training y loss
            "loss_y_unweighted": jnp.mean(per_sample_y_unw),  # mean(e_y^2)
            "loss_x_component": loss_x_component,
            "loss_unweighted_total": jnp.mean(per_sample_y_unw + per_sample_x),
        }

        if train and batch_stats is not None:
            bs = updates["batch_stats"]
        else:
            bs = None
        return loss, (key, bs, aux_metrics)

    return loss_fn_marginal
