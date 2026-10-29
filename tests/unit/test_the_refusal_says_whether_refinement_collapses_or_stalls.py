"""A refusal must say WHICH of two independent gates refused it, and neither gate subsumes the other.

The repair ceiling is a ceiling, not a diagnosis: above it `repair_direct_solution` returns the
vector bit-identical and the guard refuses with "singular / rank-deficient". But two different
systems land there --

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

from numerics_lab.linear import (
    _REPAIRABLE_CONDITION_FACTOR,
    LINEAR_SOLVER_REGISTRY,
    LinearSolveError,
    condition_estimate,
    move_is_this_systems_conditioning,
    refinement_move_trajectory,
    reject_unsolved_direct_system,
    solve_linear_system,
)

GOLDEN = Path(__file__).parent / 'goldens' / 'case_i_m6_singular_157.npz'


def _inductor_m6_singular():
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
    x = exact * 1.05  # 5% off: two decades past the retired absolute ceiling, and repairable

    moves = refinement_move_trajectory(matrix, rhs, x)

    assert moves, 'a diagnostic that measures nothing cannot separate anything'
    assert moves[0] > 1.0e-3, 'the plant must sit past the absolute ceiling this design replaced'
    assert moves[-1] < moves[0] * 1e-6


def test_a_singular_system_has_a_STALLING_trajectory() -> None:
    """The other half: the production singular plant, where the moves do NOT collapse."""
    matrix, rhs = _inductor_m6_singular()
    x = sp.linalg.spsolve(sp.csc_matrix(matrix), rhs)

    moves = refinement_move_trajectory(matrix, rhs, np.nan_to_num(np.asarray(x).ravel()))

    assert moves
    assert moves[-1] > moves[0] * 1e-6, 'a singular system has no unique answer to converge to'


@pytest.mark.parametrize('plant', ['solvable', 'singular'])
def test_the_trajectory_rides_on_a_REFUSAL_and_never_replaces_one(plant: str) -> None:
    """Both plants still raise. The evidence is added to the message; the verdict does not move."""
    if plant == 'singular':
        matrix, rhs = _inductor_m6_singular()
        x = np.nan_to_num(np.asarray(sp.linalg.spsolve(sp.csc_matrix(matrix), rhs)).ravel())
    else:
        matrix, rhs = _solvable_system()
        x = sp.linalg.spsolve(sp.csc_matrix(matrix), rhs) * 1.05

    with pytest.raises(LinearSolveError) as excinfo:
        reject_unsolved_direct_system(matrix, rhs, x, 'direct')

    message = str(excinfo.value)
    assert 'refinement trajectory' in message
    assert ('COLLAPSES' in message) or ('STALLS' in message)


def _spd(n: int = 8):
    """A well-conditioned SPD system -- the plant `test_high_residual_direct_solve_is_rejected` uses."""
    rng = np.random.default_rng(0)
    m = rng.standard_normal((n, n))
    return sp.csr_matrix(m @ m.T + n * np.eye(n)), rng.standard_normal(n)


def test_a_broken_backend_COLLAPSES_TOO_which_is_why_the_trajectory_cannot_decide_alone() -> None:
    """THE REFUTATION THAT SHAPED THE DESIGN, planted so it cannot be re-derived away.

    Refinement drives ANY finite start to the answer of a SOLVABLE system, so `np.ones_like(rhs)`
    on a well-posed SPD system collapses exactly like a conditioning-spoiled answer does. A
    criterion made only of "does the trajectory collapse" would therefore have repaired garbage
    into the right answer and silently flipped the test that exists to catch it. Measured here so
    the next reader sees the collapse rather than trusting the claim.
    """
    matrix, rhs = _spd()
    moves = refinement_move_trajectory(matrix, rhs, np.ones_like(rhs))

    assert moves[-1] <= moves[0] * 1.0e-6, 'the broken-backend plant COLLAPSES -- that is the whole point'
    assert moves[0] > 1.0e-3


def test_the_two_gates_are_INDEPENDENT_and_the_magnitude_one_refuses_the_collapsing_garbage() -> None:
    """The ratchet: passing one gate is not passing the decision.

    The broken-backend plant passes the TRAJECTORY gate (it collapses, above) and fails the
    MAGNITUDE one, because `np.ones_like(rhs)` sits ~5.9e+14 times its system's own `cond * eps`.
    A system on which both gates agreed would not need two.
    """
    matrix, rhs = _spd()
    moves = refinement_move_trajectory(matrix, rhs, np.ones_like(rhs))
    cond = condition_estimate(matrix)

    assert cond is not None
    ratio = moves[0] / (cond * float(np.finfo(float).eps))
    assert ratio > _REPAIRABLE_CONDITION_FACTOR * 1.0e6, f'the plant must be far past the gate, got {ratio:.3e}'
    assert move_is_this_systems_conditioning(matrix, moves[0]) is False


def test_the_broken_backend_plant_is_STILL_REFUSED_and_says_so_by_magnitude() -> None:
    """CONDITION (2), asserted where it can actually regress: nothing refused today is accepted.

    This is `test_high_residual_direct_solve_is_rejected`'s plant driven through the whole
    `solve_linear_system` path, which is the path the conditioning gate changed. It must still
    raise, and the message must blame the MAGNITUDE gate rather than a rank deficiency -- the
    system is well posed and telling its reader to go looking for a missing gauge would be the
    declaration that lies.
    """
    matrix, rhs = _spd()

    # `**_kwargs` IS THE FIX, NOT A RENAME: the registry calls this substitution by keyword for
    # the optional hooks, so declaring those names would make them a contract the caller could
    # break. No names remain to break. See `test_suppression_ratchet.py`'s docstring -- this is
    # the shape it names as the only legitimate way past that ratchet.
    def _finite_garbage(_m, b, **_kwargs):
        del _kwargs  # accepted and deliberately ignored: this vector is wrong by construction
        return np.ones_like(b)

    original = LINEAR_SOLVER_REGISTRY['direct']
    LINEAR_SOLVER_REGISTRY['direct'] = _finite_garbage
    try:
        with pytest.raises(LinearSolveError) as excinfo:
            solve_linear_system(sp.csc_matrix(matrix), rhs, linear_solver='direct')
    finally:
        LINEAR_SOLVER_REGISTRY['direct'] = original

    message = str(excinfo.value)
    assert 'MAGNITUDE' in message
    assert 'STALLS' not in message


def test_an_undecidable_condition_estimate_DECLINES_the_repair_rather_than_allowing_it() -> None:
    """Unmeasured is not a pass. With no scale to judge against, the gate must not open."""
    matrix, _rhs = _spd()

    assert move_is_this_systems_conditioning(matrix[:4], 1.0e-3) is None
