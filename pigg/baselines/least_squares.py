r"""Classical identification: least squares, and the exact Bayesian posterior.

Two estimators live here, and the second one is more useful than it first looks.

``ols_base_parameters`` is the textbook approach: regress the torque on the
*independent* regressor columns.  Since ``Y pi = Y_1 (pi_1 + K pi_2) = Y_1
beta``, ordinary least squares on ``Y_1`` recovers the base parameters ``beta``
directly, and nothing more can be recovered from any amount of data.

``exact_posterior`` exploits a structural feature of this problem that removes
the need for sampling.  The observation model ``tau = Y pi + eps`` is *linear*
in the parameters with Gaussian noise, so under a Gaussian prior the posterior
over the full ``5n`` standard parameters is available in closed form:

.. math::
    \Sigma_\text{post} = \left(\Sigma_0^{-1} + Y^\top Y / \sigma^2\right)^{-1},
    \qquad
    \mu_\text{post} = \Sigma_\text{post}
        \left(\Sigma_0^{-1}\mu_0 + Y^\top \tau / \sigma^2\right)

This is the exact posterior, not an approximation, so it is the reference every
uncertainty method in the project is scored against — no MCMC required.  Along
the unidentifiable directions ``Y^\top Y`` contributes nothing and the posterior
collapses to the prior, which is precisely the behaviour a correct uncertainty
estimate must reproduce and an overconfident one will not.

MCMC only becomes necessary once that linear-Gaussian structure is broken — by
non-Gaussian noise, or by hard physical-consistency constraints on ``pi``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pigg.physics.identifiability import Identifiability


@dataclass
class LeastSquaresResult:
    """Base-parameter point estimate with its frequentist covariance."""

    beta: np.ndarray
    """``(rank,)`` estimated base parameters."""
    cov: np.ndarray
    """``(rank, rank)`` covariance, ``sigma^2 (Y1^T Y1)^-1``."""
    noise_var: float
    residual_rms: float
    names: list[str]

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    def summary(self) -> list[str]:
        return [
            f"{name:<28s} {val: .6f} +/- {sd:.6f}"
            for name, val, sd in zip(self.names, self.beta, self.std)
        ]


@dataclass
class GaussianPosterior:
    """Exact Gaussian posterior over the full standard parameter vector."""

    mean: np.ndarray
    cov: np.ndarray
    names: list[str]

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    def interval(self, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Central credible interval at the given level."""
        from scipy.stats import norm

        z = norm.ppf(0.5 + level / 2)
        half = z * self.std
        return self.mean - half, self.mean + half

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        return rng.multivariate_normal(self.mean, self.cov, size=n)


def estimate_noise_variance(
    regressor: np.ndarray, tau: np.ndarray, beta: np.ndarray
) -> tuple[float, float]:
    """Residual noise variance and RMS, with the usual ``n - p`` correction."""
    residual = np.asarray(tau, float) - np.asarray(regressor, float) @ beta
    dof = max(regressor.shape[0] - regressor.shape[1], 1)
    return float(residual @ residual / dof), float(np.sqrt(np.mean(residual**2)))


def ols_base_parameters(
    regressor: np.ndarray,
    tau: np.ndarray,
    identifiability: Identifiability,
    weights: np.ndarray | None = None,
) -> LeastSquaresResult:
    """Least squares on the identifiable subspace.

    Parameters
    ----------
    regressor, tau
        Stacked ``(n_rows, 5n)`` and ``(n_rows,)``, e.g. from
        :func:`pigg.physics.simulate.prepare_identification_data`.
    weights
        Optional per-row weights (use ``1/sigma_j`` when joints have different
        torque noise levels); applied as a diagonal reweighting of both sides.
    """
    regressor = np.asarray(regressor, dtype=float)
    tau = np.asarray(tau, dtype=float)

    design = regressor[:, identifiability.independent]
    if weights is not None:
        w = np.sqrt(np.asarray(weights, dtype=float))
        design = design * w[:, None]
        tau = tau * w

    beta, *_ = np.linalg.lstsq(design, tau, rcond=None)
    noise_var, residual_rms = estimate_noise_variance(design, tau, beta)

    gram = design.T @ design
    cov = noise_var * np.linalg.pinv(gram)

    names = [identifiability.names[i] for i in identifiability.independent]
    return LeastSquaresResult(
        beta=beta, cov=cov, noise_var=noise_var, residual_rms=residual_rms, names=names
    )


def exact_posterior(
    regressor: np.ndarray,
    tau: np.ndarray,
    names: list[str],
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    noise_var: float,
) -> GaussianPosterior:
    """Closed-form posterior over the full ``5n`` parameter vector.

    Unlike :func:`ols_base_parameters`, this returns a distribution over the
    *standard* parameters, including the directions the data cannot identify —
    where it reduces exactly to the prior.  That makes it the right object to
    feed to :func:`pigg.physics.identifiability.nullspace_alignment`.

    Parameters
    ----------
    prior_mean, prior_cov
        Gaussian prior over ``pi``. A broad but proper prior is required: with
        an improper flat prior the posterior is undefined along the null space.
    noise_var
        Torque noise variance. Under-stating it is the usual route to a
        falsely-confident posterior, so estimate it rather than guessing —
        :func:`estimate_noise_variance` does this from the OLS residuals.
    """
    regressor = np.asarray(regressor, dtype=float)
    tau = np.asarray(tau, dtype=float)
    prior_mean = np.asarray(prior_mean, dtype=float)
    prior_cov = np.asarray(prior_cov, dtype=float)

    if noise_var <= 0:
        raise ValueError(f"noise_var must be positive, got {noise_var}")

    # Whiten by the prior, then invert via an SVD of the design matrix rather
    # than of the normal equations. In whitened coordinates
    #     Sigma_post = L (I + A^T A)^-1 L^T,   A = Y L / sigma
    # and with A = U S V^T this is L V diag(1/(1+s^2)) V^T L^T.
    #
    # Going through A rather than A^T A matters here. The singular values span
    # ~1e9, so forming A^T A squares that to 1e18 and destroys roughly nine
    # digits precisely along the unidentifiable directions -- the ones this
    # whole analysis is about. Via the SVD those directions have s = 0 to
    # machine precision, so 1/(1+s^2) = 1 exactly and the posterior reproduces
    # the prior there to full accuracy.
    chol_prior = np.linalg.cholesky(prior_cov)
    design = regressor @ chol_prior / np.sqrt(noise_var)

    _, sv, vt = np.linalg.svd(design, full_matrices=True)
    sv = np.pad(sv, (0, regressor.shape[1] - sv.size))
    inner_inv = (vt.T * (1.0 / (1.0 + sv**2))) @ vt

    post_cov = chol_prior @ inner_inv @ chol_prior.T
    residual = tau - regressor @ prior_mean
    post_mean = prior_mean + post_cov @ (regressor.T @ residual) / noise_var

    # symmetrise to kill accumulated asymmetry from the solves
    post_cov = 0.5 * (post_cov + post_cov.T)
    return GaussianPosterior(mean=post_mean, cov=post_cov, names=list(names))
