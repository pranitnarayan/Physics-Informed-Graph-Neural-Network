"""Identifiability structure and the classical estimators built on it."""

import numpy as np
import pytest

from pigg.baselines import estimate_noise_variance, exact_posterior, ols_base_parameters
from pigg.physics import get_dynamics, standard_to_pi
from pigg.physics.excitation import (
    condition_number,
    lazy_trajectory,
    optimise_excitation,
    random_trajectory,
)
from pigg.physics.identifiability import (
    analyse_chain,
    expected_rank,
    nullspace_alignment,
    stack_regressor,
)
from pigg.physics.simulate import add_sensor_noise, prepare_identification_data


# --------------------------------------------------------------------------
# structural identifiability
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_links", [2, 3, 4])
def test_rank_is_four_per_link(n_links, rng):
    """One unidentifiable direction per link, so rank is 4n not 5n."""
    dyn = get_dynamics(n_links, link_lengths=np.linspace(1.0, 0.6, n_links))
    idf = analyse_chain(dyn, n_samples=400, rng=rng)
    assert idf.rank == expected_rank(n_links)
    assert idf.n_unidentifiable == n_links


def test_singular_value_gap_is_unambiguous(dyn2, rng):
    """Rank is structural, so the spectrum should show a huge gap, not a taper."""
    idf = analyse_chain(dyn2, n_samples=400, rng=rng)
    sv = idf.singular_values
    assert sv[idf.rank - 1] / sv[0] > 1e-3
    assert sv[idf.rank] / sv[0] < 1e-10


def test_every_mass_is_unidentifiable(dyn3, rng):
    """The mass-last basis should discard exactly the masses."""
    idf = analyse_chain(dyn3, n_samples=400, rng=rng)
    assert [idf.names[i] for i in idf.dependent] == ["m1", "m2", "m3"]


def test_nullspace_annihilates_the_regressor(dyn3, rng):
    idf = analyse_chain(dyn3, n_samples=400, rng=rng)
    regressor = stack_regressor(dyn3, n_samples=200, rng=rng)
    scale = np.abs(regressor).max()
    assert np.abs(regressor @ idf.nullspace).max() / scale < 1e-12


def test_base_parameters_are_invariant_along_the_nullspace(dyn3, pi3, rng):
    """Moving pi along an unidentifiable direction changes no observable."""
    idf = analyse_chain(dyn3, n_samples=400, rng=rng)
    q, dq, ddq = (
        rng.uniform(-np.pi, np.pi, 3),
        rng.normal(size=3),
        rng.normal(size=3),
    )
    tau = dyn3.inverse_dynamics(q, dq, ddq, pi3)
    base = idf.base_values(pi3)

    for direction in idf.nullspace.T:
        shifted = pi3 + 0.75 * direction
        assert np.allclose(idf.base_values(shifted), base, atol=1e-9)
        assert np.allclose(dyn3.inverse_dynamics(q, dq, ddq, shifted), tau, atol=1e-9)


def test_regrouping_matches_the_analytic_form(dyn3, rng):
    """Base parameters should read h_i + sum_{k>i} l_i m_k and J_i + l_i^2 m_k."""
    idf = analyse_chain(dyn3, n_samples=400, rng=rng)
    lengths = dyn3.link_lengths
    lookup = {name: row for row, name in enumerate([idf.names[i] for i in idf.independent])}
    dep = {idf.names[j]: col for col, j in enumerate(idf.dependent)}

    for i in range(dyn3.n_links):
        for stem, power in (("h", 1), ("J", 2)):
            row = lookup[f"{stem}{i + 1}"]
            for k in range(dyn3.n_links):
                coeff = idf.regroup[row, dep[f"m{k + 1}"]]
                expected = lengths[i] ** power if k > i else 0.0
                assert coeff == pytest.approx(expected, abs=1e-8)


# --------------------------------------------------------------------------
# least squares
# --------------------------------------------------------------------------


def test_ols_recovers_base_parameters_exactly_on_clean_data(dyn2, pi2, rng):
    """Week-1 milestone: clean data must give essentially exact recovery."""
    idf = analyse_chain(dyn2, n_samples=400, rng=rng)
    traj = random_trajectory(2, n_harmonics=5, base_freq=0.1, velocity_scale=1.5, rng=rng)
    t = np.linspace(0.0, traj.period, 600, endpoint=False)

    regressor = traj.regressor(dyn2, t)
    tau = traj.to_trajectory(dyn2, pi2, t).tau.reshape(-1)

    result = ols_base_parameters(regressor, tau, idf)
    truth = idf.base_values(pi2)

    rel = np.abs(result.beta - truth).max() / np.abs(truth).max()
    assert rel < 1e-8, f"relative error {rel:.2e}\n" + "\n".join(result.summary())
    assert result.residual_rms < 1e-9


def test_ols_is_unaffected_by_nullspace_shifts(dyn2, pi2, rng):
    """Two parameter vectors differing only unobservably must fit identically."""
    idf = analyse_chain(dyn2, n_samples=400, rng=rng)
    traj = random_trajectory(2, n_harmonics=4, base_freq=0.1, rng=rng)
    t = np.linspace(0.0, traj.period, 400, endpoint=False)
    regressor = traj.regressor(dyn2, t)

    shifted = pi2 + 1.3 * idf.nullspace[:, 0]
    tau_a = traj.to_trajectory(dyn2, pi2, t).tau.reshape(-1)
    tau_b = traj.to_trajectory(dyn2, shifted, t).tau.reshape(-1)
    assert np.allclose(tau_a, tau_b, atol=1e-9)

    beta_a = ols_base_parameters(regressor, tau_a, idf).beta
    beta_b = ols_base_parameters(regressor, tau_b, idf).beta
    assert np.allclose(beta_a, beta_b, atol=1e-9)


def test_ols_survives_the_noisy_measurement_pipeline(dyn2, pi2, rng):
    """Encoder-only measurement, filtered derivatives, parallel-filtered fit."""
    idf = analyse_chain(dyn2, n_samples=400, rng=rng)
    traj = random_trajectory(2, n_harmonics=4, base_freq=0.1, velocity_scale=1.5, rng=rng)
    t = np.linspace(0.0, 4 * traj.period, 8000, endpoint=False)  # 200 Hz

    clean = traj.to_trajectory(dyn2, pi2, t)
    noisy = add_sensor_noise(
        clean, angle_noise_std=2e-4, encoder_counts=20000, torque_noise_std=5e-3, rng=rng
    )

    regressor, tau = prepare_identification_data(dyn2, noisy, cutoff_hz=3.0, trim=200)
    result = ols_base_parameters(regressor, tau, idf)
    truth = idf.base_values(pi2)

    rel = np.linalg.norm(result.beta - truth) / np.linalg.norm(truth)
    assert rel < 0.05, f"relative error {rel:.3f}\n" + "\n".join(result.summary())


def test_noise_variance_estimate_is_sane(dyn2, pi2, rng):
    idf = analyse_chain(dyn2, n_samples=400, rng=rng)
    traj = random_trajectory(2, n_harmonics=4, base_freq=0.1, rng=rng)
    t = np.linspace(0.0, traj.period, 800, endpoint=False)

    regressor = traj.regressor(dyn2, t)
    tau = traj.to_trajectory(dyn2, pi2, t).tau.reshape(-1)
    sigma = 0.02
    tau_noisy = tau + rng.normal(0.0, sigma, tau.shape)

    beta = ols_base_parameters(regressor, tau_noisy, idf).beta
    noise_var, _ = estimate_noise_variance(regressor[:, idf.independent], tau_noisy, beta)
    assert np.sqrt(noise_var) == pytest.approx(sigma, rel=0.15)


# --------------------------------------------------------------------------
# exact Bayesian posterior
# --------------------------------------------------------------------------


def _posterior_setup(dyn, pi, idf, rng, sigma=0.02, prior_std=3.0):
    traj = random_trajectory(dyn.n_links, n_harmonics=4, base_freq=0.1, rng=rng)
    t = np.linspace(0.0, traj.period, 800, endpoint=False)
    regressor = traj.regressor(dyn, t)
    tau = traj.to_trajectory(dyn, pi, t).tau.reshape(-1)
    tau = tau + rng.normal(0.0, sigma, tau.shape)

    prior_cov = (prior_std**2) * np.eye(dyn.n_params)
    post = exact_posterior(
        regressor, tau, dyn.param_names, np.zeros(dyn.n_params), prior_cov, sigma**2
    )
    return post, prior_cov


def test_posterior_reduces_to_the_prior_along_unidentifiable_directions(dyn2, pi2, rng):
    """The defining property of a correct posterior for this inverse problem."""
    idf = analyse_chain(dyn2, n_samples=400, rng=rng)
    post, prior_cov = _posterior_setup(dyn2, pi2, idf, rng, prior_std=3.0)

    scale = np.linalg.norm(post.mean)
    for direction in idf.nullspace.T:
        posterior_var = direction @ post.cov @ direction
        prior_var = direction @ prior_cov @ direction
        assert posterior_var == pytest.approx(prior_var, rel=1e-8)

        # The mean cannot move along these directions either. It is not exactly
        # zero because Y @ v vanishes only to ~1e-14, and the mean update scales
        # that by 1/noise_var (~2e4 here); judging it against the size of the
        # mean is the meaningful comparison.
        assert abs(direction @ post.mean) < 1e-5 * scale


def test_posterior_is_informative_along_identifiable_directions(dyn2, pi2, rng):
    """Data must sharply shrink the identifiable subspace, or nothing was learnt."""
    idf = analyse_chain(dyn2, n_samples=400, rng=rng)
    post, prior_cov = _posterior_setup(dyn2, pi2, idf, rng, prior_std=3.0)

    identifiable = np.linalg.svd(idf.nullspace, full_matrices=True)[0][:, idf.n_unidentifiable :]
    for direction in identifiable.T:
        shrinkage = (direction @ post.cov @ direction) / (direction @ prior_cov @ direction)
        assert shrinkage < 1e-2

    # Compare against the overall scale rather than entry-by-entry: several base
    # parameters (the Coulomb terms) are exactly zero here, so a relative
    # tolerance on them is meaningless.
    got = idf.base_values(post.mean)
    truth = idf.base_values(pi2)
    assert np.linalg.norm(got - truth) / np.linalg.norm(truth) < 0.05


def test_nullspace_alignment_separates_honest_from_overconfident(dyn2, pi2, rng):
    """The headline diagnostic: honest posteriors put their variance where the
    data says nothing; overconfident ones do not."""
    idf = analyse_chain(dyn2, n_samples=400, rng=rng)
    post, _ = _posterior_setup(dyn2, pi2, idf, rng, prior_std=3.0)

    honest = nullspace_alignment(post.cov, idf.nullspace)
    overconfident = nullspace_alignment(1e-6 * np.eye(dyn2.n_params), idf.nullspace)

    assert honest > 0.95, f"exact posterior scored only {honest:.3f}"
    assert overconfident == pytest.approx(idf.n_unidentifiable / dyn2.n_params, rel=1e-6)
    assert honest > overconfident


# --------------------------------------------------------------------------
# excitation quality
# --------------------------------------------------------------------------


def test_optimised_excitation_beats_a_lazy_trajectory(dyn2, rng):
    """Conditioning is a property of the motion, and it is worth optimising."""
    idf = analyse_chain(dyn2, n_samples=400, rng=rng)

    lazy = lazy_trajectory(2, base_freq=0.1, rng=rng)
    t_lazy = np.linspace(0.0, lazy.period, 200, endpoint=False)
    cond_lazy = condition_number(lazy.regressor(dyn2, t_lazy), idf.rank)

    best, cond_best = optimise_excitation(
        dyn2, rank=idf.rank, n_harmonics=4, n_restarts=2, max_iter=40, rng=rng
    )
    assert cond_best < cond_lazy, f"optimised {cond_best:.1f} vs lazy {cond_lazy:.1f}"
    assert np.isfinite(cond_best)
    assert best.n_harmonics == 4


def test_least_squares_is_less_certain_under_poor_excitation(dyn2, pi2, rng):
    """Uncertainty must grow when the trajectory stops exciting the dynamics."""
    idf = analyse_chain(dyn2, n_samples=400, rng=rng)
    sigma = 0.01

    def posterior_width(traj):
        t = np.linspace(0.0, traj.period, 600, endpoint=False)
        regressor = traj.regressor(dyn2, t)
        tau = traj.to_trajectory(dyn2, pi2, t).tau.reshape(-1)
        tau = tau + rng.normal(0.0, sigma, tau.shape)
        return np.mean(ols_base_parameters(regressor, tau, idf).std)

    lazy = posterior_width(lazy_trajectory(2, base_freq=0.1, rng=rng))
    lively = posterior_width(
        random_trajectory(2, n_harmonics=5, base_freq=0.1, velocity_scale=2.0, rng=rng)
    )
    assert lazy > 5 * lively, f"lazy {lazy:.4g} vs lively {lively:.4g}"
