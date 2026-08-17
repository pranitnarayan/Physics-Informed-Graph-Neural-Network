"""Make ``src/`` importable without requiring an editable install."""

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pigg.physics import get_dynamics, standard_to_pi  # noqa: E402


@pytest.fixture(scope="session")
def dyn2():
    return get_dynamics(2, link_lengths=[1.0, 0.6], gravity=9.81)


@pytest.fixture(scope="session")
def dyn3():
    return get_dynamics(3, link_lengths=[1.0, 0.8, 0.6], gravity=9.81)


@pytest.fixture(scope="session")
def pi2():
    return standard_to_pi(
        mass=[1.2, 0.7],
        com=[0.45, 0.30],
        inertia=[0.05, 0.02],
        viscous=[0.12, 0.06],
        coulomb=[0.0, 0.0],
    )


@pytest.fixture(scope="session")
def pi3():
    return standard_to_pi(
        mass=[1.5, 1.0, 0.6],
        com=[0.5, 0.4, 0.3],
        inertia=[0.08, 0.04, 0.015],
        viscous=[0.10, 0.07, 0.04],
        coulomb=[0.0, 0.0, 0.0],
    )


@pytest.fixture
def rng():
    return np.random.default_rng(20260814)
