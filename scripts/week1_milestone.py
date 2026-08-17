"""Week-1 milestone: the physics foundation, end to end.

Run with::

    python scripts/week1_milestone.py

Everything here is classical -- no learning yet. The point is to establish the
ground truth that every later model is scored against, and to check the two
claims Week 1 rests on: that the regressor is exact, and that the identifiability
structure is what theory predicts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pigg.baselines import exact_posterior, ols_base_parameters  # noqa: E402
from pigg.physics import get_dynamics, standard_to_pi  # noqa: E402
from pigg.physics.excitation import (  # noqa: E402
    condition_number,
    lazy_trajectory,
    optimise_excitation,
)
from pigg.physics.identifiability import (  # noqa: E402
    analyse_chain,
    expected_rank,
    nullspace_alignment,
)
from pigg.physics.simulate import (  # noqa: E402
    add_sensor_noise,
    prepare_identification_data,
    simulate,
)

RNG = np.random.default_rng(0)
RULE = "=" * 74


def banner(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def main() -> None:
    dyn = get_dynamics(2, link_lengths=[1.0, 0.6], gravity=9.81)
    pi_true = standard_to_pi(
        mass=[1.2, 0.7],
        com=[0.45, 0.30],
        inertia=[0.05, 0.02],
        viscous=[0.12, 0.06],
        coulomb=[0.0, 0.0],
    )

    # ---------------------------------------------------------------- 1
    banner("1. Identifiability: rank is 4n, not 5n")
    idf = analyse_chain(dyn, n_samples=600, rng=RNG)
    print(f"parameters {idf.n_params}, rank {idf.rank} (predicted {expected_rank(2)})")
    print(f"unidentifiable directions: {idf.n_unidentifiable}")
    print(f"singular value gap: {idf.singular_values[idf.rank - 1]:.3e} -> "
          f"{idf.singular_values[idf.rank]:.3e}")
    print(f"discarded: {[idf.names[i] for i in idf.dependent]}")
    print("base parameters:")
    for line in idf.describe():
        print(f"   {line}")

    print("\nrank across chain lengths:")
    for n in (2, 3, 4, 5, 6):
        other = get_dynamics(n, link_lengths=np.linspace(1.0, 0.6, n))
        r = analyse_chain(other, n_samples=400, rng=RNG).rank
        flag = "ok" if r == expected_rank(n) else "MISMATCH"
        print(f"   n={n}: rank {r:3d} / {5 * n:3d}   predicted {expected_rank(n):3d}   {flag}")

    # ---------------------------------------------------------------- 2
    banner("2. Excitation quality drives conditioning")
    lazy = lazy_trajectory(2, base_freq=0.1, rng=RNG)
    t_lazy = np.linspace(0.0, lazy.period, 400, endpoint=False)
    cond_lazy = condition_number(lazy.regressor(dyn, t_lazy), idf.rank)

    best, cond_best = optimise_excitation(
        dyn, rank=idf.rank, n_harmonics=5, n_restarts=3, max_iter=60, rng=RNG
    )
    print(f"lazy trajectory      cond = {cond_lazy:10.1f}")
    print(f"optimised excitation cond = {cond_best:10.1f}")
    print(f"improvement factor        = {cond_lazy / cond_best:10.1f}x")

    # ---------------------------------------------------------------- 3
    banner("3. Least squares on clean data (Week-1 milestone: < 1% error)")
    t = np.linspace(0.0, best.period, 1200, endpoint=False)
    regressor = best.regressor(dyn, t)
    tau = best.to_trajectory(dyn, pi_true, t).tau.reshape(-1)

    result = ols_base_parameters(regressor, tau, idf)
    truth = idf.base_values(pi_true)
    rel = np.linalg.norm(result.beta - truth) / np.linalg.norm(truth)
    print(f"relative error {rel:.3e}   residual RMS {result.residual_rms:.3e}")
    for name, got, want in zip(result.names, result.beta, truth):
        print(f"   {name:<5s} estimated {got: .8f}   true {want: .8f}")

    # ---------------------------------------------------------------- 4
    banner("4. Least squares through the noisy measurement pipeline")
    t_long = np.linspace(0.0, 4 * best.period, 8000, endpoint=False)  # 200 Hz
    clean = best.to_trajectory(dyn, pi_true, t_long)
    noisy = add_sensor_noise(
        clean, angle_noise_std=2e-4, encoder_counts=20000, torque_noise_std=5e-3, rng=RNG
    )
    y_filt, tau_filt = prepare_identification_data(dyn, noisy, cutoff_hz=3.0, trim=200)
    noisy_result = ols_base_parameters(y_filt, tau_filt, idf)
    rel_noisy = np.linalg.norm(noisy_result.beta - truth) / np.linalg.norm(truth)
    print("encoder-only measurement, filtered derivatives, parallel-filtered fit")
    print(f"relative error {rel_noisy:.4f}   ({rel_noisy * 100:.2f}%)")
    for line in noisy_result.summary():
        print(f"   {line}")

    # ---------------------------------------------------------------- 5
    banner("5. Exact posterior and the null-space diagnostic")
    sigma2 = noisy_result.noise_var
    prior_cov = 9.0 * np.eye(dyn.n_params)
    post = exact_posterior(
        y_filt, tau_filt, dyn.param_names, np.zeros(dyn.n_params), prior_cov, sigma2
    )
    honest = nullspace_alignment(post.cov, idf.nullspace)
    overconfident = nullspace_alignment(1e-6 * np.eye(dyn.n_params), idf.nullspace)
    print("fraction of posterior variance lying in the unidentifiable subspace")
    print(f"   exact posterior        {honest:.4f}   <- should approach 1")
    print(f"   overconfident (tiny I) {overconfident:.4f}   <- ignores the degeneracy")
    print("\nper-parameter posterior (note the masses stay at the prior width 3.0):")
    lo, hi = post.interval(0.95)
    for name, mean, sd, a, b in zip(dyn.param_names, post.mean, post.std, lo, hi):
        print(f"   {name:<5s} {mean: .5f} +/- {sd:.5f}   95% [{a: .4f}, {b: .4f}]")

    # ---------------------------------------------------------------- 6
    banner("6. Simulator sanity: unforced frictionless energy drift")
    pi_nf = pi_true.copy()
    pi_nf[3::5] = 0.0
    pi_nf[4::5] = 0.0
    t_sim = np.linspace(0.0, 10.0, 2001)
    traj = simulate(dyn, pi_nf, q0=[1.1, -0.4], dq0=[0.0, 0.0], t_eval=t_sim)
    energy = dyn.total_energy(traj.q, traj.dq, pi_nf)
    drift = np.abs(energy - energy[0]).max() / abs(energy[0])
    print(f"relative energy drift over 10 s: {drift:.3e}")


if __name__ == "__main__":
    main()
