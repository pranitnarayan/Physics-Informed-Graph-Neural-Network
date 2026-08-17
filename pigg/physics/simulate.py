r"""Forward simulation, sensor models, and derivative estimation.

The identification pipeline never sees ``qddot``.  Real encoders measure angles
only, and naive finite differencing amplifies noise by ``1/dt**2``, which would
dominate the error budget and be misread as model error.  The route taken here
is the standard one from robot identification: zero-phase low-pass filtering,
central differences on the *filtered* signal, and then the **same** filter
applied to both sides of ``tau = Y pi``.  Because that relation is linear and
holds pointwise, any linear filter ``F`` preserves it::

    F(tau) = F(Y) pi

so the parameters are unchanged while out-of-band noise is removed from both
sides.  See :func:`prepare_identification_data`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import butter, filtfilt

from pigg.physics.derive import Dynamics

#: width of the tanh used in place of sign(qdot) during integration; a hard
#: discontinuity would stall any adaptive-step solver near zero velocity
DEFAULT_COULOMB_EPS = 1e-3


@dataclass
class Trajectory:
    """A sampled trajectory. ``t`` is ``(T,)``; every other field is ``(T, n)``."""

    t: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray
    tau: np.ndarray

    @property
    def n_links(self) -> int:
        return self.q.shape[1]

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])

    def __len__(self) -> int:
        return self.t.shape[0]


def smooth_sign(x: np.ndarray, eps: float = DEFAULT_COULOMB_EPS) -> np.ndarray:
    """Continuous stand-in for ``sign`` so Coulomb friction stays integrable."""
    return np.tanh(np.asarray(x, dtype=float) / eps)


def simulate(
    dyn: Dynamics,
    pi: np.ndarray,
    q0: np.ndarray,
    dq0: np.ndarray,
    t_eval: np.ndarray,
    torque_fn=None,
    coulomb_eps: float = DEFAULT_COULOMB_EPS,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> Trajectory:
    """Integrate the forward dynamics with a high-accuracy explicit solver.

    Parameters
    ----------
    torque_fn
        ``f(t, q, dq) -> (n,)`` applied joint torques. ``None`` means unforced,
        which for two or more links is chaotic; keep horizons short and prefer
        :mod:`pigg.physics.excitation` trajectories for identification data.
    rtol, atol
        Deliberately tight. The simulator is the ground truth against which
        every learned model is scored, so integration error must sit far below
        the sensor noise being studied.
    """
    n = dyn.n_links
    q0 = np.asarray(q0, dtype=float)
    dq0 = np.asarray(dq0, dtype=float)
    t_eval = np.asarray(t_eval, dtype=float)

    def torque_at(t, q, dq):
        if torque_fn is None:
            return np.zeros(n)
        return np.asarray(torque_fn(t, q, dq), dtype=float)

    def rhs(t, y):
        q, dq = y[:n], y[n:]
        tau = torque_at(t, q, dq)
        ddq = dyn.forward_dynamics(q, dq, tau, pi, sgn=smooth_sign(dq, coulomb_eps))
        return np.concatenate([dq, ddq])

    sol = solve_ivp(
        rhs,
        (float(t_eval[0]), float(t_eval[-1])),
        np.concatenate([q0, dq0]),
        t_eval=t_eval,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")

    q = sol.y[:n].T
    dq = sol.y[n:].T

    tau = np.stack([torque_at(t_eval[k], q[k], dq[k]) for k in range(len(t_eval))])
    ddq = dyn.forward_dynamics(q, dq, tau, pi, sgn=smooth_sign(dq, coulomb_eps))

    return Trajectory(t=t_eval, q=q, dq=dq, ddq=ddq, tau=tau)


# --------------------------------------------------------------------------
# sensor model
# --------------------------------------------------------------------------


def add_sensor_noise(
    traj: Trajectory,
    angle_noise_std: float = 0.0,
    encoder_counts: int | None = None,
    torque_noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Trajectory:
    """Corrupt a trajectory the way a real rig would.

    Only ``q`` and ``tau`` are measured. ``dq`` and ``ddq`` are set to ``nan``
    so that any code path accidentally consuming ground-truth derivatives fails
    loudly instead of silently reporting optimistic results.

    Parameters
    ----------
    angle_noise_std
        Standard deviation of additive Gaussian angle noise, in radians.
    encoder_counts
        Counts per revolution; applies uniform quantisation if given.
    """
    rng = np.random.default_rng() if rng is None else rng

    q = traj.q.copy()
    if angle_noise_std > 0:
        q = q + rng.normal(0.0, angle_noise_std, size=q.shape)
    if encoder_counts is not None:
        step = 2 * np.pi / encoder_counts
        q = np.round(q / step) * step

    tau = traj.tau.copy()
    if torque_noise_std > 0:
        tau = tau + rng.normal(0.0, torque_noise_std, size=tau.shape)

    return replace(traj, q=q, tau=tau, dq=np.full_like(traj.dq, np.nan), ddq=np.full_like(traj.ddq, np.nan))


# --------------------------------------------------------------------------
# derivative estimation and parallel filtering
# --------------------------------------------------------------------------


def _lowpass(x: np.ndarray, dt: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass along axis 0 (no group delay)."""
    nyq = 0.5 / dt
    wn = cutoff_hz / nyq
    if not 0 < wn < 1:
        raise ValueError(
            f"cutoff {cutoff_hz} Hz is not below the Nyquist frequency {nyq:.1f} Hz"
        )
    b, a = butter(order, wn, btype="low")
    padlen = 3 * max(len(a), len(b))
    if x.shape[0] <= padlen:
        raise ValueError(f"need more than {padlen} samples to filter, got {x.shape[0]}")
    return filtfilt(b, a, x, axis=0)


def estimate_derivatives(
    t: np.ndarray,
    q_meas: np.ndarray,
    cutoff_hz: float,
    order: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filtered ``(q, dq, ddq)`` from measured angles alone.

    Differentiation is applied to the *filtered* signal, and re-filtered after
    each stage, because each differentiation re-injects high-frequency noise.
    """
    t = np.asarray(t, dtype=float)
    dt = float(t[1] - t[0])

    q = _lowpass(np.asarray(q_meas, dtype=float), dt, cutoff_hz, order)
    dq = _lowpass(np.gradient(q, dt, axis=0), dt, cutoff_hz, order)
    ddq = _lowpass(np.gradient(dq, dt, axis=0), dt, cutoff_hz, order)
    return q, dq, ddq


def prepare_identification_data(
    dyn: Dynamics,
    traj: Trajectory,
    cutoff_hz: float,
    trim: int = 50,
    coulomb_eps: float = DEFAULT_COULOMB_EPS,
    order: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the filtered regressor and torque for least squares or a physics loss.

    Returns stacked ``(Y, tau)`` of shapes ``(T' * n, 5n)`` and ``(T' * n,)``,
    ready for ``lstsq``. The same low-pass is applied to both, which leaves
    ``pi`` invariant while removing out-of-band noise from each side.

    Parameters
    ----------
    trim
        Samples dropped from each end. ``filtfilt`` edge effects are worst
        there, and leaving them in visibly biases the parameter estimates.
    """
    dt = traj.dt
    q, dq, ddq = estimate_derivatives(traj.t, traj.q, cutoff_hz, order)

    regressor = dyn.regressor(q, dq, ddq, sgn=smooth_sign(dq, coulomb_eps))  # (T, n, 5n)
    tau = traj.tau

    # parallel filtering: F(tau) = F(Y) pi
    n_p = regressor.shape[-1]
    regressor = _lowpass(regressor.reshape(len(traj), -1), dt, cutoff_hz, order)
    regressor = regressor.reshape(len(traj), traj.n_links, n_p)
    tau = _lowpass(tau, dt, cutoff_hz, order)

    if trim > 0:
        if 2 * trim >= len(traj):
            raise ValueError(f"trim={trim} removes the whole trajectory of length {len(traj)}")
        regressor = regressor[trim:-trim]
        tau = tau[trim:-trim]

    return regressor.reshape(-1, n_p), tau.reshape(-1)
