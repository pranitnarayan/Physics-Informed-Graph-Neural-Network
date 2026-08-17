"""Invariants of the derived dynamics.

These are cheap and catch nearly every derivation bug. Nothing downstream is
meaningful if any of them fail, so they run first.
"""

import numpy as np
import pytest

from pigg.physics import get_dynamics, standard_to_pi
from pigg.physics.excitation import FourierTrajectory, random_trajectory
from pigg.physics.simulate import simulate


def _random_state(rng, n, scale=1.5):
    return (
        rng.uniform(-np.pi, np.pi, n),
        rng.normal(0.0, scale, n),
        rng.normal(0.0, scale, n),
    )


# --------------------------------------------------------------------------
# mass matrix
# --------------------------------------------------------------------------


def test_mass_matrix_symmetric_and_positive_definite(dyn2, pi2, rng):
    for _ in range(25):
        q, _, _ = _random_state(rng, dyn2.n_links)
        m = dyn2.mass_matrix(q, pi2)
        assert np.allclose(m, m.T, atol=1e-12)
        assert np.linalg.eigvalsh(m).min() > 0


def test_mass_matrix_matches_textbook_two_link(dyn2):
    """Compare against the hand-written planar 2R mass matrix."""
    m1, m2 = 1.3, 0.8
    lc1, lc2 = 0.4, 0.25
    i1, i2 = 0.06, 0.03
    l1 = dyn2.link_lengths[0]
    pi = standard_to_pi([m1, m2], [lc1, lc2], [i1, i2])

    theta2 = 0.7
    got = dyn2.mass_matrix(np.array([0.3, theta2]), pi)

    m11 = m1 * lc1**2 + i1 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * np.cos(theta2)) + i2
    m12 = m2 * (lc2**2 + l1 * lc2 * np.cos(theta2)) + i2
    m22 = m2 * lc2**2 + i2

    assert np.allclose(got, np.array([[m11, m12], [m12, m22]]), atol=1e-12)


def test_base_link_mass_does_not_affect_dynamics(dyn2, pi2, rng):
    """``m1`` sits at the fixed base, so it cannot influence any torque."""
    q, dq, ddq = _random_state(rng, dyn2.n_links)
    perturbed = pi2.copy()
    perturbed[0] += 5.0  # m1
    assert np.allclose(
        dyn2.inverse_dynamics(q, dq, ddq, pi2),
        dyn2.inverse_dynamics(q, dq, ddq, perturbed),
        atol=1e-12,
    )


# --------------------------------------------------------------------------
# passivity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", ["dyn2", "dyn3"])
def test_mdot_minus_2c_is_skew_symmetric(fixture_name, request, rng):
    """The Christoffel factorisation of C must make Mdot - 2C skew-symmetric."""
    dyn = request.getfixturevalue(fixture_name)
    pi = request.getfixturevalue("pi2" if fixture_name == "dyn2" else "pi3")

    for _ in range(10):
        q, dq, _ = _random_state(rng, dyn.n_links)
        c = dyn.coriolis(q, dq, pi)
        eps = 1e-6
        mdot = (dyn.mass_matrix(q + eps * dq, pi) - dyn.mass_matrix(q - eps * dq, pi)) / (2 * eps)
        s = mdot - 2 * c
        assert np.abs(s + s.T).max() < 1e-6


def test_coriolis_not_derived_for_long_chains():
    dyn = get_dynamics(6)
    with pytest.raises(NotImplementedError, match="only derived"):
        dyn.coriolis(np.zeros(6), np.zeros(6), np.zeros(dyn.n_params))


def test_stale_cache_is_discarded(tmp_path, monkeypatch):
    """A pickle written by different code must not be loaded silently."""
    from pigg.physics import derive

    monkeypatch.setattr(derive, "_CACHE_DIR", tmp_path)
    derive.get_dynamics(2)
    cache_file = tmp_path / "planar_chain_2.pkl"
    assert cache_file.exists()

    # forge a cache entry carrying a wrong stamp and a poisoned mass matrix
    import pickle

    with cache_file.open("rb") as fh:
        payload = pickle.load(fh)
    payload["_stamp"] = {"version": -1, "coriolis_limit": -1}
    payload["M"] = None
    with cache_file.open("wb") as fh:
        pickle.dump(payload, fh)

    dyn = derive.get_dynamics(2)  # must re-derive rather than crash on M=None
    assert dyn.mass_matrix(np.zeros(2), np.zeros(dyn.n_params)).shape == (2, 2)


# --------------------------------------------------------------------------
# regressor
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_links", [1, 2, 3, 4, 6])
def test_regressor_identity(n_links, rng):
    """``Y(q, qdot, qddot) @ pi`` must equal the inverse dynamics exactly."""
    dyn = get_dynamics(n_links, link_lengths=np.linspace(1.0, 0.5, n_links))
    pi = standard_to_pi(
        mass=rng.uniform(0.5, 2.0, n_links),
        com=rng.uniform(0.2, 0.5, n_links),
        inertia=rng.uniform(0.01, 0.1, n_links),
        viscous=rng.uniform(0.0, 0.2, n_links),
        coulomb=rng.uniform(0.0, 0.1, n_links),
    )
    q, dq, ddq = _random_state(rng, n_links)
    sgn = np.sign(dq)

    tau = dyn.inverse_dynamics(q, dq, ddq, pi, sgn=sgn)
    assert np.allclose(dyn.regressor(q, dq, ddq, sgn=sgn) @ pi, tau, atol=1e-10)


def test_regressor_does_not_depend_on_parameters(dyn2, rng):
    q, dq, ddq = _random_state(rng, dyn2.n_links)
    y = dyn2.regressor(q, dq, ddq)
    assert np.isfinite(y).all()
    assert y.shape == (dyn2.n_links, dyn2.n_params)


def test_dynamics_assembly_matches_inverse_dynamics(dyn2, pi2, rng):
    """M qddot + C qdot + g + friction reproduces the Euler-Lagrange torque."""
    q, dq, ddq = _random_state(rng, dyn2.n_links)
    assembled = (
        dyn2.mass_matrix(q, pi2) @ ddq
        + dyn2.coriolis(q, dq, pi2) @ dq
        + dyn2.gravity_torque(q, pi2)
        + pi2[3::5] * dq
        + pi2[4::5] * np.sign(dq)
    )
    assert np.allclose(assembled, dyn2.inverse_dynamics(q, dq, ddq, pi2), atol=1e-10)


def test_forward_inverse_round_trip(dyn3, pi3, rng):
    q, dq, ddq = _random_state(rng, dyn3.n_links)
    tau = dyn3.inverse_dynamics(q, dq, ddq, pi3)
    assert np.allclose(dyn3.forward_dynamics(q, dq, tau, pi3), ddq, atol=1e-9)


def test_batched_matches_single(dyn3, pi3, rng):
    q = rng.uniform(-np.pi, np.pi, (11, 3))
    dq = rng.normal(size=(11, 3))
    ddq = rng.normal(size=(11, 3))

    batched = dyn3.inverse_dynamics(q, dq, ddq, pi3)
    assert batched.shape == (11, 3)
    for k in range(11):
        assert np.allclose(batched[k], dyn3.inverse_dynamics(q[k], dq[k], ddq[k], pi3), atol=1e-12)

    y_batched = dyn3.regressor(q, dq, ddq)
    assert y_batched.shape == (11, 3, dyn3.n_params)
    assert np.allclose(np.einsum("bij,j->bi", y_batched, pi3), batched, atol=1e-10)


# --------------------------------------------------------------------------
# parameter conversion
# --------------------------------------------------------------------------


def test_standard_to_pi_applies_parallel_axis():
    pi = standard_to_pi(mass=[2.0], com=[0.3], inertia=[0.05])
    m, h, j, fv, fc = pi
    assert m == pytest.approx(2.0)
    assert h == pytest.approx(2.0 * 0.3)
    assert j == pytest.approx(0.05 + 2.0 * 0.3**2)
    assert (fv, fc) == (0.0, 0.0)


# --------------------------------------------------------------------------
# energy
# --------------------------------------------------------------------------


def test_energy_conserved_when_unforced_and_frictionless(dyn2, pi2):
    """The strongest end-to-end check on both the Lagrangian and the solver."""
    pi = pi2.copy()
    pi[3::5] = 0.0  # viscous
    pi[4::5] = 0.0  # Coulomb

    t = np.linspace(0.0, 8.0, 1601)
    traj = simulate(dyn2, pi, q0=[1.1, -0.4], dq0=[0.0, 0.0], t_eval=t)

    energy = dyn2.total_energy(traj.q, traj.dq, pi)
    drift = np.abs(energy - energy[0]).max()
    assert drift / np.abs(energy[0]) < 1e-8, f"relative energy drift {drift / abs(energy[0]):.2e}"


def test_friction_dissipates_energy(dyn2, pi2):
    t = np.linspace(0.0, 8.0, 1601)
    traj = simulate(dyn2, pi2, q0=[1.1, -0.4], dq0=[0.0, 0.0], t_eval=t)
    energy = dyn2.total_energy(traj.q, traj.dq, pi2)
    assert energy[-1] < energy[0]
    # monotone decrease, up to solver noise
    assert np.diff(energy).max() < 1e-9


# --------------------------------------------------------------------------
# excitation trajectories
# --------------------------------------------------------------------------


def test_fourier_derivatives_are_analytic(rng):
    """Closed-form qdot/qddot must agree with numerical differentiation of q."""
    traj = random_trajectory(3, n_harmonics=4, base_freq=0.2, rng=rng)
    t = np.linspace(0.0, traj.period, 20001)
    q, dq, ddq = traj.evaluate(t)
    dt = t[1] - t[0]

    assert np.allclose(np.gradient(q, dt, axis=0)[5:-5], dq[5:-5], atol=1e-6)
    assert np.allclose(np.gradient(dq, dt, axis=0)[5:-5], ddq[5:-5], atol=1e-5)


def test_fourier_trajectory_is_periodic(rng):
    traj = random_trajectory(2, n_harmonics=3, base_freq=0.25, rng=rng)
    q0, dq0, ddq0 = traj.evaluate(np.array([0.0]))
    q1, dq1, ddq1 = traj.evaluate(np.array([traj.period]))
    assert np.allclose(q0, q1, atol=1e-9)
    assert np.allclose(dq0, dq1, atol=1e-9)
    assert np.allclose(ddq0, ddq1, atol=1e-9)


def test_fourier_offset_is_the_mean_angle():
    """With a single harmonic the mean of q over a period is exactly q_offset."""
    traj = FourierTrajectory(
        a=np.array([[0.5]]), b=np.array([[0.3]]), q_offset=np.array([0.2]), base_freq=0.5
    )
    t = np.linspace(0.0, traj.period, 100001)
    q, _, _ = traj.evaluate(t)
    assert np.mean(q) == pytest.approx(0.2, abs=1e-6)
