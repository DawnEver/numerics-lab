"""A solve that stops at ``max_iter`` must report the increment, not only the energy error.

THE TWO STATES THE OLD LOG COULD NOT TELL APART. `_has_converged` is a conjunction: the energy has
settled AND the iterate has stopped moving. The observer was handed only the first half, so a run
that capped at ``max_iter`` logged ``Energy Error Percentage: 0.00%`` in BOTH of these cases:

* the iterate is at a genuine fixed point and the increment bar is simply stricter than the energy
  bar -- the solve is finished and the cap is a waste;
* the energy is sitting on a plateau the iterate is still wandering across -- the solve is NOT
  finished, and the plateau is exactly the failure mode `_has_converged`'s docstring records
  (a saturating-BH case that read 0.00% for 20+ iterations at ~14x the physical energy).

MEASURED 2026-08-27. One heavy shard capped 104 solves at 100/100 while logging 0.00% energy error.
Reading that log, the obvious conclusion was "converged, spinning" -- and it was the wrong one: the
alpha trace showed a step damped to ~0.1, which cannot reach the increment bar in the iterations it
has. Loosening the criterion on that misreading would have re-admitted the plateau defect.

So the increment is now measured beside the energy error, and these two tests pin that the pair can
DISCRIMINATE -- one case per state, because a hook that reported a constant would pass either alone.
"""

from __future__ import annotations

import numpy as np

from numerics_lab.nonlinear import AssembledState, get_strategy

_RTOL = 0.01  # percent -- the production default
_INCREMENT_BAR = _RTOL / 100.0  # the fraction the second half of the rule compares against


class _Observer:
    """Records BOTH halves of the convergence rule, exactly as the production log now does."""

    is_linear = False

    def __init__(self) -> None:
        self.seen: list[float] = []
        self.rows: list[tuple[float, float]] = []

    def initial_guess(self) -> np.ndarray:
        return np.zeros(1)

    def _next(self, x: float) -> float:
        raise NotImplementedError

    def assemble(self, x, /) -> AssembledState:
        self.seen.append(float(x[0]))
        # Energy is CONSTANT in both systems: that is the point. It is the flat reading that made
        # the two states look identical, so holding it fixed isolates the increment as the only
        # thing that can distinguish them.
        return AssembledState(system_matrix=None, total_energy=500.0)

    def solve_linear(self, state, /) -> np.ndarray:
        return np.array([self._next(self.seen[-1])])

    def residual(self, x, state, /) -> np.ndarray:
        return np.asarray(x) - 10.0

    def residual_norm(self, x, state, /) -> float:
        return float(abs(float(np.asarray(x)[0]) - 10.0))

    def tangent(self, x, state, /):
        return None

    def solve_system(self, matrix, rhs, /):
        raise NotImplementedError

    def solve_secant(self, state, rhs, /):
        return np.asarray(rhs, dtype=float)

    def on_iteration(self, i, state, err, alpha, increment, /) -> None:
        self.rows.append((float(err), float(increment)))

    def scale_source(self, lam, /):
        raise NotImplementedError


class _StillWandering(_Observer):
    """Flat energy, iterate oscillating between 0 and 20 forever -- a plateau, NOT a fixed point."""

    def _next(self, x: float) -> float:
        return 20.0 - x


class _AtItsFixedPoint(_Observer):
    """Flat energy and the iterate already AT ``x* = 10`` -- genuinely finished."""

    def _next(self, x: float) -> float:
        return 10.0


def test_a_wandering_iterate_reports_an_increment_above_the_bar():
    """The defect state. Energy says 0.00%; only the increment can say "not converged"."""
    system = _StillWandering()
    get_strategy('picard-damped').solve(system, max_iter=12, rtol_energy_pct=_RTOL)

    # From the SECOND observation on: the first has no previous energy to difference against, so
    # `_energy_error` returns inf there by construction. That is the absence of a reading, not a
    # reading of "still moving", and judging the fixture on it would judge the wrong thing.
    energy_errors = [err for err, _inc in system.rows[1:]]
    worst_increment = max(inc for _err, inc in system.rows[1:])
    assert max(energy_errors) < _RTOL, (
        f'the fixture no longer reproduces the flat-energy reading it exists to test: {energy_errors[:4]}'
    )
    assert worst_increment > _INCREMENT_BAR, (
        f'a solve whose iterate is oscillating over a 20-wide interval reported a largest increment '
        f'of {worst_increment:.2e}, at or under the {_INCREMENT_BAR:.0e} bar. The energy error is '
        f'0.00% here, so if the increment does not separate these two states nothing does.'
    )


def test_a_settled_iterate_reports_an_increment_under_the_bar():
    """The converse, so the increment cannot be a constant that merely happens to look alarming.

    Without this arm, reporting a fixed large number would satisfy the test above and the pair
    would still be unable to discriminate -- which is the whole claim being made.
    """
    system = _AtItsFixedPoint()
    get_strategy('picard-damped').solve(system, max_iter=12, rtol_energy_pct=_RTOL)

    settled = [inc for _err, inc in system.rows[1:]]
    assert settled, 'the solve stopped before producing a second observation to judge'
    assert min(settled) < _INCREMENT_BAR, (
        f'an iterate sitting exactly on its fixed point never reported an increment under the '
        f'{_INCREMENT_BAR:.0e} bar (smallest {min(settled):.2e}); the number being logged is then '
        f'not a fixed-point measure and cannot be read as one'
    )
