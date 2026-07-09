"""Variational approximation module for joint marginal p(y, xi).

All distributions are built on flowjax/equinox for a unified training interface.

Example usage:
    from policy_learning.variational import (
        create_mean_field_gaussian,
        create_mixture_of_gaussians,
        create_maf,
        VariationalTrainer,
        VariationalLossLogger,
    )

    # Create a MAF distribution
    dist = create_maf(key, dim_y=1, dim_xi=2, T=10)

    # Train with forward KL
    trainer = VariationalTrainer(
        variational_dist=dist,
        target_model=bed_model,
        optimizer=optax.adam(1e-3),
        key=key,
        callbacks=[VariationalLossLogger(log_every=100)],
    )
    trainer.train(num_iters=10000, batch_size=256)

    # Get learned score function
    score_fn = trainer.get_score_fn()
"""

from policy_learning.variational.callbacks_variational import (
    EquinoxCheckpointer,
    VariationalLossLogger,
    VariationalPlotCallback,
    VariationalSamplePlot,
)
from policy_learning.variational.distributions import (
    VariationalDistribution,
    create_flow_model,
    create_maf,
    create_mean_field_gaussian,
    create_mixture_of_gaussians,
)
from policy_learning.variational.train_variational import (
    VariationalTrainer,
    get_forward_kl_loss_fn,
)

__all__ = [
    # Distribution wrapper
    "VariationalDistribution",
    # Factory functions for creating distributions
    "create_flow_model",
    "create_mean_field_gaussian",
    "create_mixture_of_gaussians",
    "create_maf",
    # Loss functions
    "get_forward_kl_loss_fn",
    # Trainer
    "VariationalTrainer",
    # Callbacks
    "EquinoxCheckpointer",
    "VariationalLossLogger",
    "VariationalPlotCallback",
    "VariationalSamplePlot",
]
