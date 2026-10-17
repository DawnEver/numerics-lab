"""`solve_equilibrated` returns the same solution as the plain factorization, safely.

WHAT THIS FILE CLAIMS, AND WHAT IT DELIBERATELY DOES NOT. Equilibration is an algebraic identity:
with `D = diag(1/sqrt(|diag A|))`, `(D A D) y = D b` and `x = D y` is the same `x` in exact
arithmetic. These tests pin that identity and the edge case that could break it. They do NOT claim
equilibration is more accurate in general, because that was MEASURED AND REFUTED here on
2026-09-06: on a synthetic SPD system whose rows were scaled across nine decades -- the shape a
stiffness matrix was assumed to have -- the equilibrated solve came back slightly WORSE than the
unscaled one (componentwise 4.6e-7 vs 1.0e-7 at a 5% Dirichlet-row fraction). A test asserting
general superiority would have been a declaration the code does not enforce.

WHERE THE IMPROVEMENT IS REAL, it is measured on the real operator and lives with the case:
`tests/integration/physics/test_rotor_stress_case.py`'s LINEARITY PIN, a ratio in which every
discretization error cancels by construction, so a miss on it is arithmetic and nothing else.
Unscaled it read 2.25 at `n_radial` 8/16/32 and drifted at 64 (2.2338 recorded 2026-08-14,
2.2514 re-measured 2026-09-06 -- itself not reproducible to three digits, which is what a
conditioning floor looks like). Equilibrated, measured 2026-09-06 at 8/16/32/64/96:

    2.250000000  2.250000001  2.250000008  2.250000000  2.250000032

with the hoop residual still converging first order (0.158046 / 0.092939 / 0.052081 / 0.028146 /
0.019077). The drift is gone at every mesh, and the mesh dependence of the DEFECT is why a
synthetic fixture could not stand in for the operator that has it.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from numerics_lab.linear import solve_equilibrated


def _spd(n: int, scale: np.ndarray) -> sp.csc_matrix:
    base = sp.diags([np.full(n, 2.0), np.full(n - 1, -1.0), np.full(n - 1, -1.0)], [0, 1, -1])
    return (sp.diags(scale) @ base @ sp.diags(scale)).tocsc()


def test_it_agrees_with_the_plain_factorization_on_a_well_scaled_system() -> None:
    """THE IDENTITY. Where conditioning is not in question, the two agree to round-off.

    To round-off and not to the bit since 2026-09-26: the ``direct`` backend under the equilibrated
    solve is faer's factorisation when the native wheel is present, a different pivot/ordering
    from the SciPy ``spsolve`` it is compared against.
    """
    n = 60
    a = _spd(n, np.ones(n))
    x = np.random.default_rng(7).standard_normal(n)
    b = np.asarray(a @ x)
    assert np.abs(solve_equilibrated(a, b) - x).max() < 1e-12
    ref = np.asarray(spsolve(a, b))
    assert np.abs(solve_equilibrated(a, b) - ref).max() <= 1e-12 * np.abs(ref).max()


def test_it_agrees_with_the_plain_factorization_on_a_badly_scaled_one_too() -> None:
    """THE FLOOR, and the honest form of it.

    The two solves must land on the SAME answer even when the operator's rows span the decades a
    real `K`'s do -- that is the identity holding where it matters. The bar is loose (1e-4
    relative) precisely because this is the regime where both solves lose digits; tightening it
    would be asserting the accuracy claim this file's docstring says was refuted.
    """
    rng = np.random.default_rng(20260906)
    n = 400
    scale = np.full(n, np.sqrt(2e11))
    scale[rng.choice(n, n // 20, replace=False)] = 1.0  # folded Dirichlet rows sit at unit scale
    a = _spd(n, scale)
    x = rng.standard_normal(n)
    b = np.asarray(a @ x)
    plain, equilibrated = np.asarray(spsolve(a, b)), solve_equilibrated(a, b)
    assert np.abs(plain - x).max() / np.abs(x).max() < 1e-4, 'the fixture stopped being solvable'
    assert np.abs(equilibrated - plain).max() / np.abs(x).max() < 1e-4


def test_a_zero_diagonal_row_does_not_divide_by_zero() -> None:
    """A row with no diagonal entry has no scale to take out; it passes through, not blows up."""
    a = sp.csc_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
    assert np.allclose(solve_equilibrated(a, np.array([2.0, 3.0])), [3.0, 2.0])


def test_a_negative_diagonal_is_scaled_by_its_MAGNITUDE() -> None:
    """`sqrt` of a raw negative diagonal is `nan`; the scaling must take `|diag|`.

    An indefinite operator (a saddle-point / KKT system is the one this repo assembles) has
    negative diagonal entries by construction, so this is not a hypothetical row.
    """
    a = sp.csc_matrix(np.array([[2.0, 1.0], [1.0, -3.0]]))
    x = np.array([0.5, -1.25])
    got = solve_equilibrated(a, np.asarray(a @ x))
    assert np.all(np.isfinite(got))
    assert np.allclose(got, x)
