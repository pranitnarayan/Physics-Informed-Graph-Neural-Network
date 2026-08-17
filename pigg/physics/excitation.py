r"""Periodic excitation trajectories, in the Swevers finite-Fourier form.

Each joint follows

.. math::
    q_j(t) = q_{j,0} + \sum_{k=1}^{H}
        \frac{a_{jk}}{\omega k}\sin(\omega k t) - \frac{b_{jk}}{\omega k}\cos(\omega k t)

so that velocity and acceleration are available *analytically* rather than by
differentiation, and the motion is periodic with zero-mean velocity.  That makes
these trajectories the natural identification dataset: evaluate ``q, qdot,
qddot`` in closed form, apply inverse dynamics for ``tau``, and the regressor is
exact to machine precision with no integration or filtering in the loop.

Why bother optimising them: which parameters are identifiable at all is fixed by
the structure (see :mod:`pigg.physics.identifiability`), but *how well* they can
be estimated from finite noisy data is set by the conditioning of ``Y`` on the
identifiable subspace, and that is a property of the trajectory.  Deliberately
poorly-conditioned trajectories from :func:`lazy_trajectory` are equally useful:
they are the stress case in which honest uncertainty estimates must widen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from pigg.physics.derive import Dynamics
from pigg.physics.simulate import Trajectory


@dataclass
class FourierTrajectory:
    """Finite Fourier series reference motion for an ``n``-joint chain."""

    a: np.ndarray
    """``(n, n_harmonics)`` sine coefficients, in rad/s."""
    b: np.ndarray
    """``(n, n_harmonics)`` cosine coefficients, in rad/s."""
    q_offset: np.ndarray
    """``(n,)`` mean joint angles, in rad."""
    base_freq: float
    """Fundamental frequency in Hz; the period is ``1 / base_freq``."""

    @property
    def n_links(self) -> int:
        return self.a.shape[0]

    @property
    def n_harmonics(self) -> int:
        return self.a.shape[1]

    @property
    def period(self) -> float:
        return 1.0 / self.base_freq

    def evaluate(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Analytic ``(q, dq, ddq)``, each ``(len(t), n)``."""
        t = np.asarray(t, dtype=float)[:, None, None]  # (T, 1, 1)
        w = 2 * np.pi * self.base_freq
        k = np.arange(1, self.n_harmonics + 1)[None, None, :]  # (1, 1, H)
        wk = w * k

        phase = wk * t
        sin, cos = np.sin(phase), np.cos(phase)
        a, b = self.a[None], self.b[None]  # (1, n, H)

        q = self.q_offset[None] + np.sum(a / wk * sin - b / wk * cos, axis=-1)
        dq = np.sum(a * cos + b * sin, axis=-1)
        ddq = np.sum(-a * wk * sin + b * wk * cos, axis=-1)
        return q, dq, ddq

    def to_trajectory(self, dyn: Dynamics, pi: np.ndarray, t: np.ndarray) -> Trajectory:
        """Exact trajectory with the torque required to track it."""
        q, dq, ddq = self.evaluate(t)
        tau = dyn.inverse_dynamics(q, dq, ddq, pi)
        return Trajectory(t=np.asarray(t, dtype=float), q=q, dq=dq, ddq=ddq, tau=tau)

    def regressor(self, dyn: Dynamics, t: np.ndarray) -> np.ndarray:
        """Stacked ``(len(t) * n, 5n)`` regressor along this trajectory."""
        q, dq, ddq = self.evaluate(t)
        return dyn.regressor(q, dq, ddq, sgn=np.sign(dq)).reshape(-1, dyn.n_params)


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------


def random_trajectory(
    n_links: int,
    n_harmonics: int = 5,
    base_freq: float = 0.1,
    velocity_scale: float = 1.0,
    rng: np.random.Generator | None = None,
) -> FourierTrajectory:
    """Random coefficients, scaled so joint speeds are order ``velocity_scale``."""
    rng = np.random.default_rng() if rng is None else rng
    scale = velocity_scale / np.sqrt(n_harmonics)
    return FourierTrajectory(
        a=rng.normal(0.0, scale, (n_links, n_harmonics)),
        b=rng.normal(0.0, scale, (n_links, n_harmonics)),
        q_offset=rng.uniform(-np.pi, np.pi, n_links),
        base_freq=base_freq,
    )


def lazy_trajectory(
    n_links: int,
    base_freq: float = 0.1,
    velocity_scale: float = 0.05,
    rng: np.random.Generator | None = None,
) -> FourierTrajectory:
    """A deliberately *badly* exciting trajectory: one slow, small harmonic.

    Near-static motion leaves the inertial terms barely excited, so the
    identifiable subspace becomes ill-conditioned and several base parameters
    are only weakly determined.  Any uncertainty estimate worth reporting must
    widen here relative to :func:`optimise_excitation`.
    """
    rng = np.random.default_rng() if rng is None else rng
    return FourierTrajectory(
        a=rng.normal(0.0, velocity_scale, (n_links, 1)),
        b=rng.normal(0.0, velocity_scale, (n_links, 1)),
        q_offset=rng.uniform(-0.2, 0.2, n_links),
        base_freq=base_freq,
    )


# --------------------------------------------------------------------------
# conditioning
# --------------------------------------------------------------------------


def condition_number(regressor: np.ndarray, rank: int) -> float:
    """``sigma_0 / sigma_{rank-1}``: conditioning on the identifiable subspace.

    The full condition number is infinite by construction (the regressor is
    structurally rank-deficient), so it must be truncated at the known rank to
    say anything about excitation quality.
    """
    sv = np.linalg.svd(np.asarray(regressor, dtype=float), compute_uv=False)
    if rank > sv.size:
        raise ValueError(f"rank {rank} exceeds {sv.size} singular values")
    if sv[rank - 1] <= 0:
        return np.inf
    return float(sv[0] / sv[rank - 1])


def _pack(traj: FourierTrajectory) -> np.ndarray:
    return np.concatenate([traj.a.ravel(), traj.b.ravel()])


def _unpack(x: np.ndarray, template: FourierTrajectory) -> FourierTrajectory:
    n, h = template.a.shape
    return FourierTrajectory(
        a=x[: n * h].reshape(n, h),
        b=x[n * h :].reshape(n, h),
        q_offset=template.q_offset,
        base_freq=template.base_freq,
    )


def optimise_excitation(
    dyn: Dynamics,
    rank: int,
    n_harmonics: int = 5,
    base_freq: float = 0.1,
    n_samples: int = 200,
    q_limit: float = np.pi,
    dq_limit: float = 3.0,
    n_restarts: int = 4,
    max_iter: int = 120,
    rng: np.random.Generator | None = None,
) -> tuple[FourierTrajectory, float]:
    """Minimise the truncated condition number subject to joint/speed limits.

    Limits are imposed as a smooth penalty rather than hard constraints: the
    objective is already non-convex and multi-modal, so random restarts with a
    penalised smooth objective is both simpler and more reliable here than a
    constrained solver.

    Returns
    -------
    The best trajectory found and its condition number.
    """
    rng = np.random.default_rng() if rng is None else rng
    t = np.linspace(0.0, 1.0 / base_freq, n_samples, endpoint=False)

    def objective(x: np.ndarray, template: FourierTrajectory) -> float:
        traj = _unpack(x, template)
        q, dq, ddq = traj.evaluate(t)
        penalty = (
            np.sum(np.clip(np.abs(q - traj.q_offset) - q_limit, 0, None) ** 2)
            + np.sum(np.clip(np.abs(dq) - dq_limit, 0, None) ** 2)
        )
        regressor = dyn.regressor(q, dq, ddq, sgn=np.sign(dq)).reshape(-1, dyn.n_params)
        cond = condition_number(regressor, rank)
        if not np.isfinite(cond):
            return 1e6 + 1e3 * penalty
        return float(np.log(cond) + 1e3 * penalty)

    best_traj, best_cond = None, np.inf
    for _ in range(n_restarts):
        template = random_trajectory(
            dyn.n_links, n_harmonics, base_freq, velocity_scale=0.5 * dq_limit, rng=rng
        )
        res = minimize(
            objective,
            _pack(template),
            args=(template,),
            method="Nelder-Mead",
            options={"maxiter": max_iter * len(_pack(template)), "fatol": 1e-4, "xatol": 1e-4},
        )
        traj = _unpack(res.x, template)
        cond = condition_number(traj.regressor(dyn, t), rank)
        if cond < best_cond:
            best_traj, best_cond = traj, cond

    assert best_traj is not None
    return best_traj, best_cond
