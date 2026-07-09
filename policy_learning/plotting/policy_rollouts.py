import warnings

import flax.linen as nn
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import PRNGKeyArray as Key
from jaxtyping import PyTree

from policy_learning.bed_models import (
    CartPole,
    DoublePendulum,
    Gravimetry,
    LocationFinding,
    StochasticPendulum,
)
from policy_learning.policies import StaticDesignNet


def get_rollout_plot_fn(target_dist):
    if isinstance(target_dist, LocationFinding):
        return plot_location_finding_rollout
    elif isinstance(target_dist, CartPole):
        return plot_cart_pole_rollout
    elif isinstance(target_dist, StochasticPendulum):
        return plot_stochastic_pendulum_rollout
    elif isinstance(target_dist, DoublePendulum):
        return plot_double_pendulum_rollout
    elif isinstance(target_dist, Gravimetry):
        return plot_gravimetry_rollout


def plot_location_finding_rollout(
    key: Key,
    target_dist: LocationFinding,
    policy_net: nn.Module,
    params: PyTree,
    n_rollouts: int = 1,
):
    plot_dict = {}
    for i in range(n_rollouts):
        theta, key = target_dist.sample_prior(key, ())
        epsilon, key = target_dist.sample_epsilon(key, ())
        policy_epsilon, key = target_dist.sample_policy_epsilon(key, ())

        # Generate roll out
        yT, xT, aux_data = target_dist.reparam_sample(
            params,
            policy_net,
            theta=theta,
            epsilon=epsilon,
            policy_epsilon=policy_epsilon,
        )

        if target_dist.d > 1:
            fig, ax = plt.subplots(1, 1)
            theta_prior_samples, key = target_dist.sample_prior(key, (5000,))
            ax.scatter(
                x=theta_prior_samples[:, 0, 0, 0],
                y=theta_prior_samples[:, 0, 0, 1],
                s=1,
                alpha=0.1,
                color="tab:blue",
            )
            ax.scatter(
                x=theta_prior_samples[:, 0, 1, 0],
                y=theta_prior_samples[:, 0, 1, 1],
                s=1,
                alpha=0.1,
                color="tab:blue",
            )
            ax.set_ylim((-3, 3))
            ax.set_xlim((-3, 3))
            ax.scatter(x=theta[0, :, 0], y=theta[0, :, 1], marker="x", color="tab:red")
            if not isinstance(policy_net, StaticDesignNet):
                ax.plot(xT[:, 0], xT[:, 1], "k-", linewidth=0.5)
            scatter = ax.scatter(
                x=xT[:, 0],
                y=xT[:, 1],
                s=20,
                c=jnp.arange(len(xT)),
                cmap="viridis_r",
                color=None,
            )
            plt.colorbar(scatter, ax=ax, label="Step")

        elif target_dist.d == 1:
            fig, ax = plt.subplots(1, 1)
            ax.set_ylim((-0.5, 0.5))
            ax.set_xlim((-3, 3))
            ax.scatter(
                x=theta[0, :, 0],
                y=jnp.zeros_like(theta[0, :, 0]),
                marker="x",
                color="red",
            )
            ax.scatter(
                x=xT[:, 0], y=jnp.zeros_like(xT[:, 0]), s=20
            )  # , s=5 * jnp.clip(a=jnp.exp(yT), a_min=1e-5))
        plot_dict[f"rollout_{i}"] = fig
    return plot_dict, key


def plot_stochastic_pendulum_rollout(
    key: Key,
    target_dist: StochasticPendulum,
    policy_net: nn.Module,
    params: PyTree,
    n_rollouts: int = 1,
):
    plot_dict = {}
    for i in range(n_rollouts):
        theta, key = target_dist.sample_prior(key, ())
        epsilon, key = target_dist.sample_epsilon(key, ())
        policy_epsilon, key = target_dist.sample_policy_epsilon(key, ())

        # Generate roll out
        yT, xT, aux_data = target_dist.reparam_sample(
            params,
            policy_net,
            theta=theta,
            epsilon=epsilon,
            policy_epsilon=policy_epsilon,
        )

        fig, ax = plt.subplots(1, 1)
        yTfull = jnp.concatenate([aux_data["y_init"][None], yT], axis=0)
        ax.plot(jnp.arange(target_dist.T + 1), yTfull[:, 0], label="q")
        ax.plot(jnp.arange(target_dist.T + 1), yTfull[:, 1], label="qdot")
        ax.plot(
            jnp.arange(target_dist.T) + 1,
            target_dist.design_bijector(xT[:, 0]),
            label="xi",
        )
        ax.legend()

        plot_dict[f"rollout_{i}"] = fig
    return plot_dict, key


def plot_cart_pole_rollout(
    key: Key,
    target_dist: CartPole,
    policy_net: nn.Module,
    params: PyTree,
    n_rollouts: int,
):
    plot_dict = {}
    for i in range(n_rollouts):
        theta, key = target_dist.sample_prior(key, ())
        epsilon, key = target_dist.sample_epsilon(key, ())
        policy_epsilon, key = target_dist.sample_policy_epsilon(key, ())

        # Generate roll out
        yT, xT, aux_data = target_dist.reparam_sample(
            params,
            policy_net,
            theta=theta,
            epsilon=epsilon,
            policy_epsilon=policy_epsilon,
        )

        fig, ax = plt.subplots(1, 1)
        yTfull = jnp.concatenate([aux_data["y_init"][None], yT], axis=0)
        ax.plot(jnp.arange(target_dist.T + 1), yTfull[:, 0], label="s")
        ax.plot(jnp.arange(target_dist.T + 1), yTfull[:, 1], label="q")
        ax.plot(jnp.arange(target_dist.T + 1), yTfull[:, 2], label="sdot")
        ax.plot(jnp.arange(target_dist.T + 1), yTfull[:, 3], label="qdot")
        ax.plot(
            jnp.arange(target_dist.T) + 1,
            target_dist.design_bijector(xT[:, 0]),
            label="xi",
        )
        ax.legend()
        plot_dict[f"rollout_{i}"] = fig
    return plot_dict, key


def plot_double_pendulum_rollout(
    key: Key,
    target_dist: DoublePendulum,
    policy_net: nn.Module,
    params: PyTree,
    n_rollouts: int,
):
    plot_dict = {}
    for i in range(n_rollouts):
        theta, key = target_dist.sample_prior(key, ())
        epsilon, key = target_dist.sample_epsilon(key, ())
        policy_epsilon, key = target_dist.sample_policy_epsilon(key, ())

        # Generate roll out
        yT, xT, aux_data = target_dist.reparam_sample(
            params,
            policy_net,
            theta=theta,
            epsilon=epsilon,
            policy_epsilon=policy_epsilon,
        )

        fig, ax = plt.subplots(1, 1)
        yTfull = jnp.concatenate([aux_data["y_init"][None], yT], axis=0)
        ax.plot(jnp.arange(target_dist.T + 1), yTfull[:, 0], label="q1")
        ax.plot(jnp.arange(target_dist.T + 1), yTfull[:, 1], label="q2")
        ax.plot(jnp.arange(target_dist.T + 1), yTfull[:, 2], label="dq1")
        ax.plot(jnp.arange(target_dist.T + 1), yTfull[:, 3], label="dq2")
        ax.plot(
            jnp.arange(target_dist.T) + 1,
            target_dist.design_bijector(xT)[:, 0],
            label="xi1",
        )
        ax.plot(
            jnp.arange(target_dist.T) + 1,
            target_dist.design_bijector(xT)[:, 1],
            label="xi2",
        )
        ax.legend()
        plot_dict[f"rollout_{i}"] = fig
    return plot_dict, key


def plot_gravimetry_rollout(
    key: Key,
    target_dist: Gravimetry,
    policy_net: nn.Module,
    params: PyTree,
    n_rollouts: int = 1,
):
    """Policy-rollout plot for the Gravimetry model.

    search_dim=1: top panel = observed anomaly y vs measurement location (with the
    true g(x; theta) curve); bottom panel = a to-scale cross-section showing the
    buried sphere as a ball at horizontal position x0, depth z, with radius equal to
    the equivalent physical source radius R = (kappa / kappa_coeff)**(1/3), plus the
    measurement positions on the surface.

    search_dim=2: scatter of the 2D measurement locations (coloured by step) over a
    contour of the surface anomaly field, with the source (x0, y0) marked.
    """
    import numpy as np

    plot_dict = {}
    for i in range(n_rollouts):
        theta, key = target_dist.sample_prior(key, ())
        epsilon, key = target_dist.sample_epsilon(key, ())
        policy_epsilon, key = target_dist.sample_policy_epsilon(key, ())
        yT, xT, _ = target_dist.reparam_sample(
            params,
            policy_net,
            theta=theta,
            epsilon=epsilon,
            policy_epsilon=policy_epsilon,
        )
        th0 = theta[0]  # theta is constant over the trajectory
        # Rollout designs are in network space; map to physical metres for plotting.
        xT = target_dist.design_bijector(xT)

        if target_dist.search_dim == 1:
            x0 = float(th0[0])
            z = float(th0[1])
            kappa = float(th0[2])
            R = float((kappa / target_dist.kappa_coeff) ** (1.0 / 3.0))  # equiv. radius
            x_meas = np.asarray(xT[:, 0])
            y_meas = np.asarray(yT[:, 0])

            span = max(
                float(np.abs(x_meas).max()),
                abs(x0) + 3 * z,
                2 * target_dist.design_sigma,
            )
            grid = jnp.linspace(-span, span, 400)
            g_curve = np.asarray(
                target_dist._forward(
                    grid[:, None],
                    jnp.broadcast_to(th0, (grid.shape[0], target_dist.theta_dim)),
                )[:, 0]
            )

            fig, (ax_top, ax_bot) = plt.subplots(
                2, 1, figsize=(8, 7), height_ratios=[1.2, 1.0], sharex=True
            )
            ax_top.plot(
                np.asarray(grid), g_curve, color="tab:blue", lw=1.0, label="g(x; theta)"
            )
            sc = ax_top.scatter(
                x_meas,
                y_meas,
                c=np.arange(target_dist.T),
                cmap="viridis_r",
                s=40,
                zorder=3,
                label="observations",
            )
            plt.colorbar(sc, ax=ax_top, label="step")
            ax_top.axhline(0.0, color="k", lw=0.5, alpha=0.4)
            ax_top.set_ylabel("anomaly y (microGal)")
            ax_top.legend(fontsize=8)

            # Cross-section: surface at depth 0, sphere at (x0, z) drawn to scale.
            ax_bot.axhline(0.0, color="0.4", lw=1.0)  # ground surface
            ax_bot.scatter(
                x_meas,
                np.zeros_like(x_meas),
                marker="v",
                c=np.arange(target_dist.T),
                cmap="viridis_r",
                s=40,
                zorder=3,
                label="measurements",
            )
            void = kappa < 0
            circ = plt.Circle(
                (x0, -z),
                radius=max(R, 1e-3),
                facecolor=("none" if void else "tab:red"),
                edgecolor=("tab:red" if void else "k"),
                hatch=("//" if void else None),
                lw=1.5,
                alpha=0.7,
                label=f"source (R={R:.1f} m, {'void' if void else 'mass'})",
            )
            ax_bot.add_patch(circ)
            ax_bot.plot([x0], [-z], "k+", ms=8)
            ax_bot.set_xlabel("horizontal position x (m)")
            ax_bot.set_ylabel("depth (m)")
            ax_bot.set_ylim(-(abs(x0) + 3 * z), max(5.0, R + 1))
            ax_bot.set_aspect("equal", adjustable="datalim")
            ax_bot.legend(fontsize=8, loc="lower right")
        else:
            x0 = float(th0[0])
            y0 = float(th0[1])
            z = float(th0[2])
            xy = np.asarray(xT)  # [T, 2]
            span = max(
                float(np.abs(xy).max()),
                abs(x0) + 3 * z,
                abs(y0) + 3 * z,
                2 * target_dist.design_sigma,
            )
            gx = jnp.linspace(-span, span, 120)
            gy = jnp.linspace(-span, span, 120)
            GX, GY = jnp.meshgrid(gx, gy)
            grid = jnp.stack([GX.ravel(), GY.ravel()], axis=-1)  # [G, 2]
            field = np.asarray(
                target_dist._forward(
                    grid, jnp.broadcast_to(th0, (grid.shape[0], target_dist.theta_dim))
                )[:, 0]
            ).reshape(GX.shape)

            fig, ax = plt.subplots(figsize=(7, 6))
            cf = ax.contourf(
                np.asarray(GX),
                np.asarray(GY),
                field,
                levels=20,
                cmap="RdBu_r",
                alpha=0.8,
            )
            plt.colorbar(cf, ax=ax, label="anomaly g (microGal)")
            sc = ax.scatter(
                xy[:, 0],
                xy[:, 1],
                c=np.arange(target_dist.T),
                cmap="viridis_r",
                s=45,
                edgecolor="k",
                lw=0.5,
                zorder=3,
                label="measurements",
            )
            plt.colorbar(sc, ax=ax, label="step")
            ax.plot([x0], [y0], "kx", ms=12, mew=2, label="source (x0, y0)")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_title(f"z={z:.1f} m")
            ax.set_aspect("equal")
            ax.legend(fontsize=8)

        plot_dict[f"rollout_{i}"] = fig
    return plot_dict, key
