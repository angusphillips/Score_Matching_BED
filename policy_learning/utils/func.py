import logging

import jax.numpy as jnp
from jaxtyping import Array

# Module-level guard so the residual-parameterisation warning is emitted once, not
# on every trace of the score network forward pass.
_residual_param_warned = False


def noop(*args, **kwargs):
    return None


def _apply_score_net_parameterisation(
    out: Array,
    y: Array,
    xi: Array,
    aux_data: dict | None,
    noise_sigma: Array,
    y_residual_mode: str = "default",
    noise_condition_mode: str = "none",
    xi_output_mode: str = "none",
    y_output_mode: str = "none",
):
    """Creates the parameterisation of the score network.

    Order: additive y/xi parameterisations are applied BEFORE the divisor below.

    IMPORTANT — divisor is the std σ, NOT the variance σ². The final step divides
    the (additively-parameterised) output by ``full_noise_conditioning``, whose y
    entries are ``diag(noise_sigma) = σ`` (a standard deviation). This is the correct
    choice for *output magnitude*: the Gaussian conditional score is ``-(y-μ)/σ²`` but
    ``|y-μ| ~ σ``,
    so the target magnitude is ``~ 1/σ`` and dividing a unit-scale raw output by σ
    lands it at the right scale.

    Consequence for the residual parameterisations (``y_output_mode`` in
    {plus_y, minus_y}, and ``y_residual_mode='y_t_diff'``): they do NOT achieve the
    clean "network learns dt·drift" residual learning one might expect. That would
    require dividing by σ² so that
        (raw_NN - (y - y_{t-1})) / σ²  ==  -(y - y_{t-1} - dt·drift)/σ²,
    i.e. raw_NN == dt·drift. With the σ (not σ²) divisor actually used here the
    leading ``-(y - y_{t-1})`` term does NOT fully cancel against the Gaussian
    score, so the raw network output does not reduce to dt·drift. These modes are
    not currently used in any live config; a warning is emitted if they are
    enabled. (See local/latex/scaling_audit.md, finding F4.1.)

    Args:
        noise_sigma: observation-model noise matrix.
        y_residual_mode: 'default' uses y as the residual base; 'y_t_diff' uses
            y_t - y_{t-1} (requires aux_data['y_init']). The latter is the
            natural choice for Markov dynamical-systems likelihoods.
        noise_condition_mode: we condition the y component with σ^{-1}; xi has
            no natural σ^{-1} so 'y_and_xi_mean' uses the mean σ for xi instead.
    """
    # Warn (once) if a residual parameterisation is active: the σ (not σ²) divisor
    # means these do not perform residual learning as their naming suggests.
    global _residual_param_warned
    if not _residual_param_warned and (
        y_output_mode in ("plus_y", "minus_y") or y_residual_mode == "y_t_diff"
    ):
        _residual_param_warned = True
        logging.getLogger(__name__).warning(
            "Residual parameterisation active (y_output_mode=%r, "
            "y_residual_mode=%r), but the output divisor is the std σ, not σ², so "
            "the raw network output does NOT reduce to dt·drift and these modes do "
            "NOT perform the expected residual learning. See "
            "local/latex/scaling_audit.md (F4.1).",
            y_output_mode,
            y_residual_mode,
        )

    if not isinstance(noise_sigma, float):
        assert noise_sigma.shape == (y.shape[-1], y.shape[-1])

    if isinstance(noise_sigma, float):
        diag_noise_sigma = noise_sigma * jnp.ones((y.shape[-1],))
    else:
        diag_noise_sigma = jnp.diag(noise_sigma)

    if noise_condition_mode == "none":
        full_noise_conditioning = jnp.ones((y.shape[-1] + xi.shape[-1],))
    elif noise_condition_mode == "y_only":
        full_noise_conditioning = jnp.concatenate(
            [diag_noise_sigma, jnp.ones((xi.shape[-1],))], axis=-1
        )
    elif noise_condition_mode == "y_and_xi_mean":
        mean_noise_sigma = jnp.mean(diag_noise_sigma)
        xi_conditioning = mean_noise_sigma * jnp.ones((xi.shape[-1],))
        full_noise_conditioning = jnp.concatenate(
            [diag_noise_sigma, xi_conditioning], axis=-1
        )

    # Compute the y residual choice (y_t - y_{t-1} for 'y_t_diff', else y).
    if y_residual_mode == "y_t_diff":
        if aux_data is None:
            raise ValueError("aux_data must be provided with y_init")
        y_res = _compute_t_diff_residual(y, aux_data)
    else:
        y_res = y

    # Apply additive y/xi parameterisations BEFORE the σ divisor below. (Note:
    # because the divisor is σ, not σ², this does NOT fully cancel the leading
    # -(y - y_{t-1}) term of the Gaussian-likelihood score; see the docstring.)
    if y_output_mode == "none":
        pass
    elif y_output_mode == "plus_y":
        out = out + jnp.concatenate([y_res, jnp.zeros_like(xi)], axis=-1)
    elif y_output_mode == "minus_y":
        out = out - jnp.concatenate([y_res, jnp.zeros_like(xi)], axis=-1)

    if xi_output_mode == "none":
        pass
    elif xi_output_mode == "plus_x":
        out = out + jnp.concatenate([jnp.zeros_like(y), xi], axis=-1)
    elif xi_output_mode == "minus_x":
        out = out - jnp.concatenate([jnp.zeros_like(y), xi], axis=-1)

    # Apply the σ divisor LAST so it scales the full (raw - y_res) output.
    out = out / (full_noise_conditioning)

    return out


def _compute_t_diff_residual(y, aux_data: dict):
    y_init = aux_data["y_init"]
    y_tminus1 = jnp.concatenate([jnp.expand_dims(y_init, -2), y[..., :-1, :]], axis=-2)
    return y - y_tminus1
