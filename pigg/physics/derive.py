r"""Symbolic Lagrangian derivation for a planar N-link serial chain.

Coordinates
-----------
``q[i]`` are *relative* joint angles.  The absolute orientation of link ``i`` is
the cumulative sum ``phi[i] = q[0] + ... + q[i]``.  Link ``i`` has known length
``l[i]``; its proximal joint sits at ``p[i-1]`` with ``p[-1] = (0, 0)``.

Parameterisation
----------------
Link lengths are treated as *known* geometry (from CAD), which is standard in
robot identification.  The unknown inertial parameters per link are chosen so
that the dynamics are exactly **linear** in them:

===========  =========================================================
``m[i]``     link mass
``h[i]``     first moment of mass, ``m[i] * lc[i]``
``J[i]``     inertia about the *proximal joint*, ``I[i] + m[i] * lc[i]**2``
``fv[i]``    viscous friction coefficient
``fc[i]``    Coulomb friction coefficient
===========  =========================================================

Writing the kinetic energy in these terms,

.. math::
    T = \sum_i \tfrac12 m_i \lVert\dot p_{i-1}\rVert^2
            + h_i \dot\phi_i (\dot p_{i-1} \cdot u'(\phi_i))
            + \tfrac12 J_i \dot\phi_i^2

the ``lc[i]**2`` terms fold entirely into ``J[i]``, leaving no quadratic
dependence on any unknown.  Hence

.. math::  \tau = Y(q, \dot q, \ddot q)\, \pi

with ``pi`` the stacked ``5N`` parameter vector.  That linearity is what makes
the physics loss a single matmul and gives an exact least-squares baseline.

Note that ``m[0]`` never appears: ``p[-1]`` is the fixed base, so both
``\dot p_{-1}`` and its height vanish.  This is one of several *exactly known*
unidentifiable directions used to validate uncertainty estimates downstream.
"""

from __future__ import annotations

import pickle
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import sympy as sp

#: mass, first moment, joint inertia, viscous friction, Coulomb friction
N_PARAMS_PER_LINK = 5
_PARAM_STEMS = ("m", "h", "J", "fv", "fc")

_CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache" / "symbolic"

#: bump whenever the derivation changes; stale caches are then discarded
_CACHE_VERSION = 1


def param_names(n_links: int) -> list[str]:
    """Names of the ``5 * n_links`` entries of ``pi``, in packing order."""
    return [f"{stem}{i + 1}" for i in range(n_links) for stem in _PARAM_STEMS]


def standard_to_pi(
    mass: Sequence[float],
    com: Sequence[float],
    inertia: Sequence[float],
    viscous: Sequence[float] | None = None,
    coulomb: Sequence[float] | None = None,
) -> np.ndarray:
    """Convert textbook parameters to the identification parameterisation.

    Parameters
    ----------
    mass, com, inertia
        Per-link mass ``m``, centre-of-mass distance ``lc`` measured from the
        proximal joint along the link, and inertia ``I`` about the centre of
        mass.
    viscous, coulomb
        Friction coefficients; default to zero.

    Returns
    -------
    ``pi`` of shape ``(5 * n_links,)``.
    """
    m = np.asarray(mass, dtype=float)
    lc = np.asarray(com, dtype=float)
    inertia = np.asarray(inertia, dtype=float)
    n = m.shape[0]
    fv = np.zeros(n) if viscous is None else np.asarray(viscous, dtype=float)
    fc = np.zeros(n) if coulomb is None else np.asarray(coulomb, dtype=float)

    h = m * lc
    j = inertia + m * lc**2  # parallel axis: COM -> proximal joint
    return np.stack([m, h, j, fv, fc], axis=1).reshape(-1)


# --------------------------------------------------------------------------
# symbolic derivation
# --------------------------------------------------------------------------


def _symbols(n: int) -> dict[str, list[sp.Symbol]]:
    return {
        "q": list(sp.symbols(f"q1:{n + 1}", real=True)),
        "dq": list(sp.symbols(f"dq1:{n + 1}", real=True)),
        "ddq": list(sp.symbols(f"ddq1:{n + 1}", real=True)),
        # sign(dq) carried as its own symbol so tau stays *linear* in pi and
        # sympy never has to differentiate a discontinuity.
        "sgn": list(sp.symbols(f"s1:{n + 1}", real=True)),
        "l": list(sp.symbols(f"l1:{n + 1}", positive=True)),
        "pi": [sp.Symbol(name, real=True) for name in param_names(n)],
    }


#: above this link count the O(n^3) Christoffel construction is skipped; it is
#: only needed for the passivity test, which n <= 4 already covers.
MAX_CORIOLIS_LINKS = 4


def _derive_expressions(n: int) -> dict[str, object]:
    """Build M, C, g, tau, Y, T, V for an ``n``-link planar chain.

    The energies are assembled directly in ``cos(phi_k - phi_j)`` form rather
    than by accumulating Cartesian velocities.  Both are algebraically the same,
    but the dot products ``u'(phi_k) . u'(phi_j)`` collapse to a single cosine
    analytically, so expressions stay small.  Expanding the trig instead (e.g.
    ``expand_trig`` on ``cos(q1 + ... + q6)``) blows up combinatorially and
    exhausts memory well before six links.
    """
    s = _symbols(n)
    q, dq, ddq, sgn, l = s["q"], s["dq"], s["ddq"], s["sgn"], s["l"]
    grav = sp.Symbol("g", positive=True)

    pi_by_link = [
        dict(zip(_PARAM_STEMS, s["pi"][i * N_PARAMS_PER_LINK : (i + 1) * N_PARAMS_PER_LINK]))
        for i in range(n)
    ]

    # cumulative (absolute) link angles and their rates
    phi: list[sp.Expr] = []
    dphi: list[sp.Expr] = []
    acc_q, acc_dq = sp.Integer(0), sp.Integer(0)
    for i in range(n):
        acc_q = acc_q + q[i]
        acc_dq = acc_dq + dq[i]
        phi.append(sp.expand(acc_q))
        dphi.append(sp.expand(acc_dq))

    def between(a: int, b: int) -> sp.Expr:
        """``phi[b] - phi[a]`` as a short sum; sign is irrelevant under cosine."""
        lo, hi = (a, b) if a <= b else (b, a)
        return sp.Add(*q[lo + 1 : hi + 1]) if hi > lo else sp.Integer(0)

    kinetic = sp.Integer(0)
    potential = sp.Integer(0)

    for i in range(n):
        p = pi_by_link[i]

        # J_i: rotation of link i about its own proximal joint
        kinetic += sp.Rational(1, 2) * p["J"] * dphi[i] ** 2

        # h_i: coupling between the base-point velocity and the link's rotation,
        #      since  pdot_{i-1} . u'(phi_i) = sum_k l_k dphi_k cos(phi_k - phi_i)
        kinetic += (
            p["h"]
            * dphi[i]
            * sp.Add(*[l[k] * dphi[k] * sp.cos(between(k, i)) for k in range(i)])
        )

        # m_i: translation of the whole link, ||pdot_{i-1}||^2
        kinetic += (
            sp.Rational(1, 2)
            * p["m"]
            * sp.Add(
                *[
                    l[k] * l[j] * dphi[k] * dphi[j] * sp.cos(between(k, j))
                    for k in range(i)
                    for j in range(i)
                ]
            )
        )

        # height of the proximal joint, then of the centre of mass
        potential += p["m"] * grav * sp.Add(*[l[k] * sp.sin(phi[k]) for k in range(i)])
        potential += p["h"] * grav * sp.sin(phi[i])

    # T is quadratic in the velocities, so the Hessian is exactly M(q)
    mass_matrix = sp.Matrix(n, n, lambda i, j: sp.diff(kinetic, dq[i], dq[j]))

    # Equations of motion straight from Euler-Lagrange:
    #   tau_i = d/dt(dL/dqdot_i) - dL/dq_i
    # This is O(n^2) second derivatives, versus O(n^3) for building C explicitly.
    lagrangian = kinetic - potential
    tau_expr = []
    for i in range(n):
        dl_ddq = sp.diff(lagrangian, dq[i])
        total_deriv = sp.Add(
            *[sp.diff(dl_ddq, q[j]) * dq[j] + sp.diff(dl_ddq, dq[j]) * ddq[j] for j in range(n)]
        )
        tau_expr.append(
            total_deriv
            - sp.diff(lagrangian, q[i])
            + pi_by_link[i]["fv"] * dq[i]
            + pi_by_link[i]["fc"] * sgn[i]
        )
    tau = sp.Matrix(tau_expr)

    gravity = sp.Matrix([sp.diff(potential, q[i]) for i in range(n)])

    # Coriolis via Christoffel symbols of the first kind. This particular
    # factorisation is the one that makes (Mdot - 2C) skew-symmetric, which the
    # passivity test checks. Only needed for small n.
    coriolis = None
    if n <= MAX_CORIOLIS_LINKS:
        coriolis = sp.zeros(n, n)
        dm = {
            (i, j, k): sp.diff(mass_matrix[i, j], q[k])
            for i in range(n)
            for j in range(n)
            for k in range(n)
        }
        for i in range(n):
            for j in range(n):
                coriolis[i, j] = sp.Rational(1, 2) * sp.Add(
                    *[
                        (dm[(i, j, k)] + dm[(i, k, j)] - dm[(j, k, i)]) * dq[k]
                        for k in range(n)
                    ]
                )

    # tau is linear in pi by construction, so the Jacobian *is* the regressor.
    regressor = tau.jacobian(sp.Matrix(s["pi"]))

    return {
        "n": n,
        "symbols": s,
        "gravity_symbol": grav,
        "M": mass_matrix,
        "C": coriolis,
        "g": gravity,
        "tau": tau,
        "Y": regressor,
        "T": kinetic,
        "V": potential,
    }


def _cache_stamp() -> dict[str, object]:
    """Everything about the derivation that changes what gets cached.

    Bump ``version`` whenever :func:`_derive_expressions` changes shape. Without
    a stamp a pickle written by older code loads silently and produces wrong
    results that look plausible -- the failure mode is a stale ``C`` matrix or a
    quietly different parameterisation, neither of which raises.
    """
    return {"version": _CACHE_VERSION, "coriolis_limit": MAX_CORIOLIS_LINKS}


def _load_or_derive(n: int, use_cache: bool = True) -> dict[str, object]:
    cache_file = _CACHE_DIR / f"planar_chain_{n}.pkl"
    if use_cache and cache_file.exists():
        try:
            with cache_file.open("rb") as fh:
                cached = pickle.load(fh)
            if cached.get("_stamp") == _cache_stamp():
                return cached
        except Exception:  # corrupt pickle: fall through and re-derive
            pass

    expr = _derive_expressions(n)
    expr["_stamp"] = _cache_stamp()

    if use_cache:
        # write-then-rename so a crash mid-write cannot leave a torn cache file
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix(".pkl.tmp")
        with tmp.open("wb") as fh:
            pickle.dump(expr, fh)
        tmp.replace(cache_file)
    return expr


# --------------------------------------------------------------------------
# numeric interface
# --------------------------------------------------------------------------


#: lambdify emits one deeply-nested Python expression per matrix entry, and
#: CPython's compiler recurses over that tree. Six-link regressors exceed the
#: default limit of 1000 -- especially under pytest, whose own stack is already
#: deep -- so the limit is raised for the duration of the compile only.
_LAMBDIFY_RECURSION_LIMIT = 20000


@contextmanager
def _deep_recursion(limit: int = _LAMBDIFY_RECURSION_LIMIT):
    previous = sys.getrecursionlimit()
    sys.setrecursionlimit(max(previous, limit))
    try:
        yield
    finally:
        sys.setrecursionlimit(previous)


def _make_callable(expr, args) -> Callable:
    """Lambdify a scalar/Matrix expression into a broadcasting numpy function.

    ``sympy.lambdify`` on a Matrix returns constant entries as Python scalars,
    which breaks shapes when the inputs are time-series arrays.  Flattening to a
    list and broadcasting afterwards keeps a uniform ``(*batch, *shape)`` result.
    """
    if isinstance(expr, sp.MatrixBase):
        shape = expr.shape if expr.shape[1] != 1 else (expr.shape[0],)
        flat = list(expr)
        with _deep_recursion():
            fn = sp.lambdify(args, flat, modules="numpy", cse=True)

        def wrapped(*call_args):
            parts = fn(*call_args)
            parts = np.broadcast_arrays(*[np.asarray(p, dtype=float) for p in parts])
            batch = parts[0].shape
            return np.stack(parts, axis=-1).reshape(*batch, *shape)

        return wrapped

    with _deep_recursion():
        fn = sp.lambdify(args, expr, modules="numpy", cse=True)

    def wrapped_scalar(*call_args):
        return np.asarray(fn(*call_args), dtype=float)

    return wrapped_scalar


@dataclass
class Dynamics:
    """Numeric rigid-body dynamics for a planar ``n``-link chain.

    All methods accept either single states of shape ``(n,)`` or batched
    trajectories of shape ``(..., n)``; the batch dimensions are broadcast.
    """

    n_links: int
    link_lengths: np.ndarray
    gravity: float
    _fns: dict[str, Callable]
    _expr: dict[str, object]

    # -- internals ------------------------------------------------------
    def _call(self, key: str, q, dq=None, ddq=None, sgn=None, pi=None):
        n = self.n_links
        q = np.asarray(q, dtype=float)
        zeros = np.zeros_like(q)
        dq = zeros if dq is None else np.broadcast_to(np.asarray(dq, float), q.shape)
        ddq = zeros if ddq is None else np.broadcast_to(np.asarray(ddq, float), q.shape)
        sgn = np.sign(dq) if sgn is None else np.broadcast_to(np.asarray(sgn, float), q.shape)
        pi_arr = np.zeros(N_PARAMS_PER_LINK * n) if pi is None else np.asarray(pi, float)

        args = (
            [q[..., i] for i in range(n)]
            + [dq[..., i] for i in range(n)]
            + [ddq[..., i] for i in range(n)]
            + [sgn[..., i] for i in range(n)]
            + [pi_arr[..., i] for i in range(N_PARAMS_PER_LINK * n)]
            + [self.link_lengths[i] for i in range(n)]
            + [self.gravity]
        )
        return self._fns[key](*args)

    # -- public API -----------------------------------------------------
    def mass_matrix(self, q, pi) -> np.ndarray:
        """``M(q)``, shape ``(..., n, n)``."""
        return self._call("M", q, pi=pi)

    def coriolis(self, q, dq, pi) -> np.ndarray:
        """``C(q, dq)`` factored so that ``Mdot - 2C`` is skew-symmetric.

        Only derived for chains up to :data:`MAX_CORIOLIS_LINKS` links; the
        equations of motion themselves never need it (see
        :meth:`inverse_dynamics`), it exists for the passivity check.
        """
        if "C" not in self._fns:
            raise NotImplementedError(
                f"Coriolis matrix is only derived for n_links <= {MAX_CORIOLIS_LINKS}; "
                f"this chain has {self.n_links}. Use inverse_dynamics() instead."
            )
        return self._call("C", q, dq=dq, pi=pi)

    def gravity_torque(self, q, pi) -> np.ndarray:
        """``g(q)``, shape ``(..., n)``."""
        return self._call("g", q, pi=pi)

    def regressor(self, q, dq, ddq, sgn=None) -> np.ndarray:
        """``Y(q, dq, ddq)`` with ``tau = Y @ pi``, shape ``(..., n, 5n)``.

        Depends only on measured motion, never on the unknown parameters.
        """
        return self._call("Y", q, dq=dq, ddq=ddq, sgn=sgn)

    def inverse_dynamics(self, q, dq, ddq, pi, sgn=None) -> np.ndarray:
        """Torque required to produce ``ddq``, shape ``(..., n)``."""
        return self._call("tau", q, dq=dq, ddq=ddq, sgn=sgn, pi=pi)

    def forward_dynamics(self, q, dq, tau, pi, sgn=None) -> np.ndarray:
        """Acceleration produced by ``tau``, shape ``(..., n)``."""
        q = np.asarray(q, float)
        dq = np.asarray(dq, float)
        tau = np.asarray(tau, float)
        # bias = C qd + g + friction  ==  inverse dynamics at zero acceleration
        bias = self.inverse_dynamics(q, dq, np.zeros_like(q), pi, sgn=sgn)
        mass = self.mass_matrix(q, pi)
        return np.linalg.solve(mass, (tau - bias)[..., None])[..., 0]

    def kinetic_energy(self, q, dq, pi) -> np.ndarray:
        return self._call("T", q, dq=dq, pi=pi)

    def potential_energy(self, q, pi) -> np.ndarray:
        return self._call("V", q, pi=pi)

    def total_energy(self, q, dq, pi) -> np.ndarray:
        return self.kinetic_energy(q, dq, pi) + self.potential_energy(q, pi)

    @property
    def n_params(self) -> int:
        return N_PARAMS_PER_LINK * self.n_links

    @property
    def param_names(self) -> list[str]:
        return param_names(self.n_links)


def get_dynamics(
    n_links: int,
    link_lengths: Sequence[float] | None = None,
    gravity: float = 9.81,
    use_cache: bool = True,
) -> Dynamics:
    """Build (or load from cache) the dynamics of a planar ``n_links`` chain.

    The symbolic derivation keeps link lengths and ``g`` as free symbols, so one
    cached derivation per ``n_links`` serves any geometry.
    """
    if n_links < 1:
        raise ValueError(f"n_links must be >= 1, got {n_links}")

    expr = _load_or_derive(n_links, use_cache=use_cache)
    s = expr["symbols"]
    args = s["q"] + s["dq"] + s["ddq"] + s["sgn"] + s["pi"] + s["l"] + [expr["gravity_symbol"]]

    keys = [k for k in ("M", "C", "g", "tau", "Y", "T", "V") if expr.get(k) is not None]
    fns = {key: _make_callable(expr[key], args) for key in keys}

    lengths = np.ones(n_links) if link_lengths is None else np.asarray(link_lengths, dtype=float)
    if lengths.shape != (n_links,):
        raise ValueError(f"link_lengths must have shape ({n_links},), got {lengths.shape}")

    return Dynamics(
        n_links=n_links,
        link_lengths=lengths,
        gravity=float(gravity),
        _fns=fns,
        _expr=expr,
    )
