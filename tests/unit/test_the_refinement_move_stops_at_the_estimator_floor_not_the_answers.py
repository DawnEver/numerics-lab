"""The refinement move had a FLOOR that belonged to the instrument, and it refused good answers.

``reject_unsolved_direct_system``'s verdict is ``||d|| / ||x||`` for ``d = A^-1 (b - A x)`` -- "the
relative amount a second solve would move the answer". ``repair_direct_solution`` then takes up to
four such steps and re-judges, on the argument that a merely ill-conditioned system's moves
COLLAPSE while a singular one's do not.

That argument is right about the answer and was wrong about the measurement. ``b - A x`` for an
``x`` that already solves the system to round-off is pure cancellation: computed in working
precision it carries an error of order ``eps * ||A|| ||x||``, so the move cannot fall below
``cond(A) * eps`` no matter how many steps are taken. Past ``cond ~ 1e10`` that floor crosses the
1e-6 bar, and every solve above it was refused as "singular / rank-deficient" while holding an
answer good to six digits.

FOUND 2026-09-16 on `case_c --scope full --set solve.chart=cartesian_3d --select fea,tet` at
3.0 mm: an n=207104 H(curl) magnetostatic solve refused at refinement = **1.635e-06**, i.e. 1.6x
the bar -- against the committed singular plant's 4.7 and the well-posed sweep's 1e-13. A system
that is genuinely rank-deficient does not land at 1.6x. The repair had run its four steps and the
sequence had not collapsed; it had STAGNATED, which is what a floor does and what a singular
system's moves also do, and that collision is what made the verdict unreadable.

THE BAR IS NOT TOUCHED BY ANY OF THIS, and the last test in this file is what says so.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from numerics_lab.linear import (
    LinearSolveError,
    refinement_residual,
    repair_direct_solution,
    solve_linear_system,
)

# MEASURED 2026-09-17 on this tree via `runner.py measure --price --with-heavy` (`-n0`, per-test
# timeout DISABLED so no reading is censored by the kill being priced): 4 tests, 4 passed, 0.352 s
# of CALL time for the whole file and 0.368 s over all phases, worst single test 0.240 s. The work
# is a handful of small sparse factorisations built to a DECLARED condition number -- no mesher, no
# vendor, no OCC import -- so the cost is set by `splu` on matrices sized in this file, not by
# anything that can grow under it.
# 600 s is therefore a HANG DETECTOR at ~1700x the measured file, never a performance assertion. It
# is the low end of the band a ceiling may occupy here: above `PER_TEST_TIMEOUT_S['measure']` = 300,
# so it grants MORE than an ordinary run rather than becoming the only thing that can kill its
# worker under `timeout_method = "thread"`, and below `HEAVY_PER_TEST_TIMEOUT_S` = 3000.
# THE READING WAS TAKEN CONTENDED (another agent held the box with a 3-D downstream case run), which is
# stated because a busy-box timing is not a property of the file. It can only INFLATE the number,
# and the claim made from it -- that 600 s is three orders above the work -- only strengthens.
pytestmark = pytest.mark.timeout(600)

#: The conditioning at which the working-precision floor (``cond * eps``) crosses the 1e-6 bar.
#: Chosen as a RATIO to the bar rather than as a value: at 1e10 the floor is ~2e-6, just over, which
#: is the regime the production refusal was measured in -- not a caricature three decades past it.
_COND = 1.0e10


def _ill_conditioned_system(n: int = 400) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """A symmetric system of exactly :data:`_COND` and its EXACT solution.

    Built from an orthogonal ``Q`` and a prescribed log-spaced spectrum, so the conditioning is a
    declared input rather than a property of some mesh that might drift. Full rank by
    construction -- every singular value is strictly positive -- which is the whole point: any
    refusal of this system is a false one.
    """
    rng = np.random.default_rng(1)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    spectrum = np.logspace(0.0, -np.log10(_COND), n)
    matrix = sp.csr_matrix(q @ np.diag(spectrum) @ q.T)
    exact = rng.standard_normal(n)
    return matrix, matrix @ exact, exact


def _move_sequence(matrix, rhs, x, *, compensated: bool, steps: int = 5) -> list[float]:
    """The relative refinement moves, taken with either residual, through ONE factorization."""
    factor = splu(sp.csc_matrix(matrix))
    current = np.asarray(x, dtype=float)
    moves: list[float] = []
    for _ in range(steps):
        residual = refinement_residual(matrix, rhs, current) if compensated else rhs - matrix @ current
        assert residual is not None
        step = factor.solve(residual)
        moves.append(float(np.linalg.norm(step)) / float(np.linalg.norm(current)))
        current = current + step
    return moves


def test_the_working_precision_move_stagnates_at_the_floor_and_the_compensated_one_collapses() -> None:
    """The plant: same system, same factor, same starting vector -- only the residual differs.

    The FIRST move agrees between the two to within a factor of two, and it must: both are honest
    estimates of the error really present in ``x``, and that error is genuinely ~``cond * eps``.
    What separates them is everything after it. This is the assertion that would red if
    :func:`refinement_residual` silently degraded to a working-precision subtraction.
    """
    matrix, rhs, _ = _ill_conditioned_system()
    x = splu(sp.csc_matrix(matrix)).solve(rhs)

    plain = _move_sequence(matrix, rhs, x, compensated=False)
    exact = _move_sequence(matrix, rhs, x, compensated=True)

    assert plain[0] > 1.0e-8, 'FLOOR: a plant whose first move were already tiny would prove nothing'
    assert min(plain) > 1.0e-8, f'the working-precision sequence must STAGNATE at its floor; got {plain}'
    assert exact[-1] < 1.0e-14, f'the compensated sequence must collapse to round-off; got {exact}'
    assert plain[-1] / exact[-1] > 1.0e6, 'the separation is the finding, not either number alone'


def test_the_compensated_residual_is_exact_where_the_plain_one_has_cancelled() -> None:
    """``b - A x`` measured against a reference computed in exact rational arithmetic.

    The claim :func:`refinement_residual` makes is not "smaller" -- smaller could be a bug that
    returns zeros. It is CORRECTLY ROUNDED, so it is judged against an exact evaluation of the same
    expression, and the working-precision subtraction is shown missing that mark by orders of
    magnitude on the same inputs.
    """
    matrix, rhs, _ = _ill_conditioned_system(n=120)
    x = splu(sp.csc_matrix(matrix)).solve(rhs)

    dense = matrix.toarray()
    truth = np.array([
        float(Fraction(rhs[i]) - sum((Fraction(dense[i, j]) * Fraction(x[j]) for j in range(len(x))), Fraction(0)))
        for i in range(len(rhs))
    ])
    scale = float(np.linalg.norm(truth))
    assert scale > 0.0, 'FLOOR: an exactly-zero residual would make both errors trivially zero'

    compensated = refinement_residual(matrix, rhs, x)
    assert compensated is not None
    plain = rhs - dense @ x

    err_compensated = float(np.linalg.norm(compensated - truth)) / scale
    err_plain = float(np.linalg.norm(plain - truth)) / scale
    assert err_compensated < 1.0e-14, f'correctly rounded, measured {err_compensated:.3e} against exact arithmetic'
    assert err_plain / max(err_compensated, 1.0e-300) > 1.0e3, (
        f'and the working-precision subtraction must be visibly worse on the same inputs '
        f'(plain {err_plain:.3e} vs compensated {err_compensated:.3e})'
    )


def test_a_complex_operator_is_declared_unjudged_rather_than_silently_plain() -> None:
    """``None``, not a plain residual wearing this function's name.

    The error-free transformation is stated for binary64 reals. Returning the working-precision
    answer under a name that promises double precision is the declaration-that-lies shape, so the
    fallback is the caller's and is visible at the call site.
    """
    matrix, rhs, _ = _ill_conditioned_system(n=40)
    x = splu(sp.csc_matrix(matrix)).solve(rhs)
    assert refinement_residual(matrix.astype(complex), rhs.astype(complex), x.astype(complex)) is None
    assert refinement_residual(matrix[:10], rhs[:10], x) is None, 'a non-square matrix has no refinement to take'


def test_a_genuinely_singular_system_is_still_refused_and_the_bar_is_unmoved() -> None:
    """The ratchet's other side: nothing above was allowed to make a singular system pass.

    A rank-deficient operator with an inconsistent right-hand side has no solution at all, so no
    residual however exact can produce a collapsing move sequence. It must still raise, and the
    repair must still decline to repair it.
    """
    rng = np.random.default_rng(7)
    n = 60
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    spectrum = np.logspace(0.0, -6.0, n)
    spectrum[-1] = 0.0  # exactly rank-deficient
    matrix = sp.csr_matrix(q @ np.diag(spectrum) @ q.T)
    rhs = q[:, -1] + 1.0e-3 * rng.standard_normal(n)  # a component squarely in the null direction

    repaired, moves = repair_direct_solution(matrix, rhs, np.zeros(n))
    assert not moves or moves[-1] > 1.0e-6, f'a singular system may not be repaired to the bar; moves {moves}'
    assert repaired is not None

    with pytest.raises(LinearSolveError):
        solve_linear_system(sp.csc_matrix(matrix), rhs)
