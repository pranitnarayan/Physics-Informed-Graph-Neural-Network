r"""Which parameters the data can actually determine.

The regressor ``Y`` is structurally rank-deficient: the dynamics depend on the
``5n`` standard parameters only through a smaller set of **base parameters**.
Writing ``Y P = Q R`` with column pivoting and splitting the permuted columns
into an independent block ``Y1`` (``r`` columns) and a dependent block ``Y2``,

.. math::
    Y_1 \pi_1 + Y_2 \pi_2 = Q_1 (R_{11} \pi_1 + R_{12} \pi_2)
                          = Q_1 R_{11} (\pi_1 + R_{11}^{-1} R_{12}\, \pi_2)

so the torque sees only ``beta = pi1 + K pi2`` with ``K = R11^-1 R12``.  Any
perturbation with ``dpi1 = -K dpi2`` leaves every observable unchanged; those
directions span the null space and are *unknowable from any amount of data*.

For a planar chain this recovers the textbook result analytically: the base-link
mass never appears at all (its proximal joint is the fixed base, so it has no
velocity and no height), and every other mass is absorbed into the preceding
link's inertia and first moment.  The rank is therefore ``4n`` and the null
space has dimension ``n``.

This matters for uncertainty quantification.  A posterior that reports tight
intervals on individual masses is provably overconfident, and a correct one is
*flat along this null space* — which turns "we produced error bars" into a
falsifiable claim.  See :func:`nullspace_alignment`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import qr

from pigg.physics.derive import Dynamics


@dataclass
class Identifiability:
    """Result of a numerical identifiability analysis of a regressor."""

    rank: int
    n_params: int
    singular_values: np.ndarray
    nullspace: np.ndarray
    """``(n_params, n_params - rank)``, orthonormal columns spanning the
    unidentifiable directions."""
    independent: np.ndarray
    """Indices of the parameters chosen as the base set."""
    dependent: np.ndarray
    regroup: np.ndarray
    """``K``, shape ``(rank, n_params - rank)``: how dependent parameters fold
    into the base ones."""
    names: list[str]

    @property
    def n_unidentifiable(self) -> int:
        return self.n_params - self.rank

    @property
    def condition_number(self) -> float:
        """Conditioning *restricted to the identifiable subspace*.

        The raw condition number of ``Y`` is infinite by construction, so it
        says nothing about excitation quality; this ratio does.
        """
        sv = self.singular_values[: self.rank]
        return float(sv[0] / sv[-1])

    def base_values(self, pi: np.ndarray) -> np.ndarray:
        """Project standard parameters onto the base set: ``pi1 + K pi2``."""
        pi = np.asarray(pi, dtype=float)
        return pi[..., self.independent] + pi[..., self.dependent] @ self.regroup.T

    def describe(self, tol: float = 1e-9) -> list[str]:
        """Human-readable regrouping, e.g. ``J1 + 1.000*m2``."""
        out = []
        for row, idx in enumerate(self.independent):
            terms = [self.names[idx]]
            for col, dep in enumerate(self.dependent):
                coeff = self.regroup[row, col]
                if abs(coeff) > tol:
                    terms.append(f"{coeff:+.4f}*{self.names[dep]}")
            out.append(" ".join(terms))
        return out


def stack_regressor(
    dyn: Dynamics,
    n_samples: int = 400,
    rng: np.random.Generator | None = None,
    velocity_scale: float = 2.0,
    accel_scale: float = 2.0,
) -> np.ndarray:
    """Stack ``Y`` over random states into ``(n_samples * n, 5n)``.

    Random sampling is deliberate here: it excites every mode the structure
    allows, so any rank deficiency found is *structural* rather than an artefact
    of a particular trajectory.  Use :func:`analyse` on real trajectory data to
    see how much a given motion actually excites.
    """
    rng = np.random.default_rng() if rng is None else rng
    n = dyn.n_links
    q = rng.uniform(-np.pi, np.pi, (n_samples, n))
    dq = rng.normal(0.0, velocity_scale, (n_samples, n))
    ddq = rng.normal(0.0, accel_scale, (n_samples, n))
    sgn = np.sign(dq)
    return dyn.regressor(q, dq, ddq, sgn=sgn).reshape(-1, dyn.n_params)


def mass_last_preference(n_links: int) -> np.ndarray:
    """Column order that keeps inertias and friction, discarding masses first.

    Any maximal independent column subset spans the same identifiable subspace,
    so the choice is free; this one yields the textbook regrouping in which each
    mass is absorbed into the preceding link's inertia and first moment.
    """
    from pigg.physics.derive import N_PARAMS_PER_LINK

    idx = np.arange(N_PARAMS_PER_LINK * n_links)
    is_mass = (idx % N_PARAMS_PER_LINK) == 0
    return np.concatenate([idx[~is_mass], idx[is_mass]])


def _greedy_independent(
    regressor: np.ndarray, rank: int, order: np.ndarray, tol: float
) -> np.ndarray:
    """Take columns in preference order, keeping each one that raises the rank."""
    chosen: list[int] = []
    for j in order:
        trial = chosen + [int(j)]
        if np.linalg.matrix_rank(regressor[:, trial], tol=tol) == len(trial):
            chosen.append(int(j))
        if len(chosen) == rank:
            break
    return np.sort(np.asarray(chosen, dtype=int))


def analyse(
    regressor: np.ndarray,
    names: list[str],
    rtol: float = 1e-9,
    preferred_order: np.ndarray | None = None,
) -> Identifiability:
    """Numerical rank, null space, and base-parameter regrouping of a regressor.

    Parameters
    ----------
    regressor
        Stacked ``(n_rows, n_params)`` matrix.
    rtol
        Singular values below ``rtol * sigma_max`` are treated as zero.  The gap
        is many orders of magnitude for this system, so the result is not
        sensitive to the exact value.
    preferred_order
        Column indices in the order they should be *preferred* as base
        parameters.  Only the choice of basis changes; ``rank``, ``nullspace``
        and every downstream metric are invariant.  ``None`` uses column-pivoted
        QR, which maximises conditioning but ignores physical meaning — see
        :func:`mass_last_preference` for the interpretable alternative.
    """
    regressor = np.asarray(regressor, dtype=float)
    n_params = regressor.shape[1]
    if len(names) != n_params:
        raise ValueError(f"got {len(names)} names for {n_params} columns")

    sv = np.linalg.svd(regressor, compute_uv=False)
    _, _, vt = np.linalg.svd(regressor, full_matrices=True)
    rank = int(np.sum(sv > rtol * sv[0]))
    nullspace = vt[rank:].T  # (n_params, n_params - rank)

    if preferred_order is None:
        # column-pivoted QR picks a well-conditioned independent column subset
        _, _, piv = qr(regressor, mode="economic", pivoting=True)
        independent = np.sort(piv[:rank])
    else:
        independent = _greedy_independent(regressor, rank, preferred_order, rtol * sv[0])
        if independent.size != rank:
            raise RuntimeError(
                f"preferred_order yielded {independent.size} independent columns, expected {rank}"
            )
    dependent = np.setdiff1d(np.arange(n_params), independent)

    # exact, since the dependent columns lie in the span of the independent ones
    regroup, *_ = np.linalg.lstsq(regressor[:, independent], regressor[:, dependent], rcond=None)

    return Identifiability(
        rank=rank,
        n_params=n_params,
        singular_values=sv,
        nullspace=nullspace,
        independent=independent,
        dependent=dependent,
        regroup=regroup,
        names=names,
    )


def analyse_chain(
    dyn: Dynamics,
    n_samples: int = 400,
    rng: np.random.Generator | None = None,
    rtol: float = 1e-9,
    interpretable: bool = True,
) -> Identifiability:
    """Convenience wrapper: structural identifiability of a chain.

    ``interpretable`` selects the mass-last basis, which reproduces the textbook
    regrouping. It has no effect on rank, null space, or any metric derived from
    them.
    """
    return analyse(
        stack_regressor(dyn, n_samples, rng),
        dyn.param_names,
        rtol=rtol,
        preferred_order=mass_last_preference(dyn.n_links) if interpretable else None,
    )


def expected_rank(n_links: int) -> int:
    """Analytically predicted rank ``4n`` for a planar chain.

    One unidentifiable direction per link: the base-link mass is absent from the
    dynamics entirely, and each subsequent mass is absorbed into the preceding
    link's inertia and first moment.
    """
    return 4 * n_links


def nullspace_alignment(cov: np.ndarray, nullspace: np.ndarray) -> float:
    """Fraction of a posterior's variance lying in the unidentifiable subspace.

    A well-calibrated posterior over standard parameters should be *flat* along
    directions the data cannot constrain, driving this ratio towards 1; an
    overconfident one reports spurious certainty there and drives it to 0.

    Parameters
    ----------
    cov
        ``(n_params, n_params)`` posterior covariance over standard parameters.
    nullspace
        Orthonormal basis from :class:`Identifiability`.

    Returns
    -------
    ``trace(N^T C N) / trace(C)``, in ``[0, 1]``.
    """
    cov = np.asarray(cov, dtype=float)
    total = np.trace(cov)
    if total <= 0:
        raise ValueError("covariance has non-positive trace")
    return float(np.trace(nullspace.T @ cov @ nullspace) / total)
