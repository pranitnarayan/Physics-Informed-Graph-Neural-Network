"""Classical estimators the learned models must be measured against."""

from pigg.baselines.least_squares import (
    GaussianPosterior,
    LeastSquaresResult,
    estimate_noise_variance,
    exact_posterior,
    ols_base_parameters,
)

__all__ = [
    "GaussianPosterior",
    "LeastSquaresResult",
    "estimate_noise_variance",
    "exact_posterior",
    "ols_base_parameters",
]
