"""A refusal past the repair ceiling must say WHICH of the two things it is.

`_DIRECT_REPAIRABLE_MAX = 1e-3` is a ceiling on the REPAIR, not a diagnosis: above it
`repair_direct_solution` returns the vector bit-identical and the guard refuses with
"singular / rank-deficient". But two different systems land there --

* one whose refinement moves COLLAPSE (a starting vector this system's conditioning spoiled,
  merely further out than the ceiling the repair was calibrated for -- the ceiling was measured
  on an n=864 KKT system and a solve is not obliged to stay that small), and
* one whose moves STALL (a genuine near-null direction: refinement cannot converge because the
  system has no unique answer to converge to),

and the refusal said the SECOND about both. That is a declaration that lies, and it is the
declaration a reader follows when deciding whether to look for a missing gauge.

THE VERDICT IS UNCHANGED BY THIS FILE AND THAT IS ASSERTED: both plants still raise. The
trajectory is EVIDENCE attached to a refusal, never a route to accepting one, so no bar moves.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from motronics.gauss.linear import (
    _DIRECT_REPAIRABLE_MAX,
    LinearSolveError,
    refinement_move_trajectory,
    reject_unsolved_direct_system,
)

GOLDEN = Path(__file__).parent / 'goldens' / 'lipo_inductor_m6_singular_157.npz'


def _lipo_m6_singular():
    data = np.load(GOLDEN)
    shape = tuple(int(v) for v in data['shape'])
    return sp.csr_matrix((data['data'], data['indices'], data['indptr']), shape=shape), data['rhs']


def _solvable_system(n: int = 40):
    """A non-singular SPD system: refinement from ANY finite start converges to its one answer."""
    main = np.full(n, 2.0)
    off = np.full(n - 1, -1.0)
    matrix = sp.csr_matrix(sp.diags([off, main, off], offsets=[-1, 0, 1]))
    rhs = np.ones(n)
    return matrix, rhs


def test_a_solvable_system_past_the_ceiling_has_a_COLLAPSING_trajectory() -> None:
    """THE FLOOR: the plant is solvable, the start is past the ceiling, and refinement converges."""
    matrix, rhs = _solvable_system()
    exact = sp.linalg.spsolve(sp.csc_matrix(matrix), rhs)
    x = exact * 1.05  # 5% off: two decades past `_DIRECT_REPAIRABLE_MAX`, and repairable

    moves = refinement_move_trajectory(matrix, rhs, x)

    assert moves, 'a diagnostic that measures nothing cannot separate anything'
    assert moves[0] > _DIRECT_REPAIRABLE_MAX
    assert moves[-1] < moves[0] * 1e-6


def test_a_singular_system_has_a_STALLING_trajectory() -> None:
    """The other half: the production singular plant, where the moves do NOT collapse."""
    matrix, rhs = _lipo_m6_singular()
    x = sp.linalg.spsolve(sp.csc_matrix(matrix), rhs)

    moves = refinement_move_trajectory(matrix, rhs, np.nan_to_num(np.asarray(x).ravel()))

    assert moves
    assert moves[-1] > moves[0] * 1e-6, 'a singular system has no unique answer to converge to'


@pytest.mark.parametrize('plant', ['solvable', 'singular'])
def test_the_trajectory_rides_on_a_REFUSAL_and_never_replaces_one(plant: str) -> None:
    """Both plants still raise. The evidence is added to the message; the verdict does not move."""
    if plant == 'singular':
        matrix, rhs = _lipo_m6_singular()
        x = np.nan_to_num(np.asarray(sp.linalg.spsolve(sp.csc_matrix(matrix), rhs)).ravel())
    else:
        matrix, rhs = _solvable_system()
        x = sp.linalg.spsolve(sp.csc_matrix(matrix), rhs) * 1.05

    with pytest.raises(LinearSolveError) as excinfo:
        reject_unsolved_direct_system(matrix, rhs, x, 'direct')

    message = str(excinfo.value)
    assert 'refinement trajectory' in message
    assert ('COLLAPSES' in message) or ('STALLS' in message)
