"""A `FactorCache` HIT returned back-substitution output unjudged, so inf/NaN escaped silently.

WHAT WAS ALREADY SETTLED, AND IS NOT RE-ARGUED HERE. `factor_reuse.py`'s own docstring prices the
expensive guard and REFUSES it on a hit, with the measurement: over seven `cases/network/circuit`
nodes, 595,898 solves of which 211,388 (35%) were hits, and 95% of `class_e`'s. Judging a hit by
`reject_unsolved_direct_system` means calling `build_matrix()` for it -- reinstating on a third of
every circuit march exactly the work the cache exists to remove. That is not a guard, it is a
deletion of the feature, and this file does not propose it.

WHAT WAS NOT SETTLED. The guard has two halves and only the expensive one needs the matrix.
Finiteness needs only the answer. It existed all along as `gauss.linear._check_finite` -- private
to that module and reached from exactly one line -- so the cheap half was simply unavailable here,
and a hit returned inf/NaN as an ordinary result. `invariant.md`: an unsupported combo RAISES,
never silently wrong.

THE PRICE IS A RATIO, WHICH IS WHY IT IS AFFORDABLE. `O(n)` against a back-substitution the caller
has already paid at `O(n^2)` dense / `O(nnz)` sparse. An escape hatch needs a ceiling, not just a
reason (`taste.md`), and the ceiling here is that the check can never dominate the work it rides on.

WHAT THIS DOES NOT CLAIM. A stale-but-FINITE factor -- a matrix that goes singular without moving
its key -- is untouched, and stays the key's problem on the problem's side of the seam. That is the
module's NAMED HOLE and this file closes the inf/NaN disguise only, which is the one that was
returning rather than raising.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import lu_factor, lu_solve

from numerics_lab.linear import LinearSolveError
from numerics_lab.time_integrators.factor_reuse import FactorCache

_KEY = ('scalar-jacobian', 1)

#: A 1-DOF problem whose jacobian is a plain number -- the ONE storage the expensive guard skips
#: outright (`matrix @ x` is not a matvec for a scalar), so it is also the only path on which a
#: non-finite answer had nothing at all standing in front of it. The module docstring said so in
#: as many words: "a divide by an exactly-zero scalar already yields inf/nan the caller sees".
#: 1e-200 is finite and its reciprocal is not representable, so an rhs of 1e200 overflows on the
#: back-substitution alone -- no singular matrix and no stale factor required.
_TINY = 1e-200


def _plant():
    """The scalar jacobian, as `build_matrix` sees it."""
    return np.asarray(_TINY)


def test_the_dense_path_is_not_where_this_bites():
    """SCOPE, stated so the next reader does not plant the easy case and think it covered.

    `scipy.linalg.lu_solve` calls `asarray_chkfinite` on the right-hand side, so a non-finite rhs
    on the DENSE path is refused by scipy before this seam is reached -- as a `ValueError`, not a
    `LinearSolveError`. That is why the plant below is the scalar path instead: it is the storage
    with no upstream check of its own.
    """
    with pytest.raises(ValueError, match='infs or NaNs'):
        lu_solve(lu_factor(np.array([[2.0, 0.0], [0.0, 4.0]])), np.array([np.inf, 1.0]))


def test_the_plant_reaches_the_hit_path_and_produces_a_nonfinite_answer():
    """THE FLOOR. Finding nothing is vacuous, so the plant is measured before it is relied on.

    The back-substitution alone must overflow, with the factor itself perfectly finite -- otherwise
    the assertion below would be about a singular matrix the MISS path already refuses.
    """
    assert np.isfinite(_TINY), 'the factor must be finite; a singular one is the other test'
    with np.errstate(over='ignore'):
        assert not np.isfinite(np.array([1e200]) / _TINY).all(), (
            'the plant no longer overflows, so the refusal below would prove nothing'
        )


def test_a_hit_that_back_substitutes_to_nonfinite_raises_instead_of_returning():
    """THE DISCRIMINATING ASSERTION: the second solve is a HIT and must still refuse inf/NaN.

    `build_matrix` is called exactly once, so the refusal cannot have come from the miss path with
    its matrix in hand -- the finiteness half is the only thing that can have fired.
    """
    cache = FactorCache()
    calls = []

    def build():
        calls.append(1)
        return _plant()

    assert cache.solve(_KEY, build, np.array([1.0])) == pytest.approx([1e200], rel=1e-12, abs=0.0)
    assert len(calls) == 1, 'the first solve must be the miss that fills the cache'

    with np.errstate(over='ignore'), pytest.raises(LinearSolveError, match='non-finite'):
        cache.solve(_KEY, build, np.array([1e200]))
    assert len(calls) == 1, (
        'the refusal must come from the HIT path -- if `build_matrix` ran, this was a miss and the '
        'test proves nothing about a hit'
    )


def test_a_healthy_hit_still_returns_and_still_pays_no_build():
    """THE CONVERSE. Without it, refusing every hit would satisfy the assertion above.

    A hit must remain a hit: the right answer, and `build_matrix` uncalled.
    """
    cache = FactorCache()
    calls = []

    def build():
        calls.append(1)
        return np.array([[2.0, 0.0], [0.0, 4.0]])

    first = cache.solve(_KEY, build, np.array([2.0, 4.0]))
    second = cache.solve(_KEY, build, np.array([4.0, 8.0]))
    assert first == pytest.approx([1.0, 1.0], abs=0.0)
    assert second == pytest.approx([2.0, 2.0], abs=0.0)
    assert len(calls) == 1, 'a hit must not pay the jacobian build -- that is the whole feature'
    assert cache.factorizations == 1
    assert cache.solves == 2


def test_the_first_solve_of_a_singular_march_is_still_refused():
    """The MISS half, unchanged: the cache starts empty, so a march that is singular from the
    start is refused with its matrix in hand before any hit can back-substitute against it.
    """
    cache = FactorCache()
    with pytest.raises(LinearSolveError):
        cache.solve(('singular', 1), lambda: np.array([[1.0, 1.0], [1.0, 1.0]]), np.array([1.0, 2.0]))
