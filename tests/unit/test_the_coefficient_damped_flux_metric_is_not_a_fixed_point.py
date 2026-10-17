"""`coefficient-damped` is the FEA default, and its second metric does NOT see the iterate move.

THE NAMED HOLE THIS CLOSES. `test_a_flat_energy_plateau_is_not_convergence.py` covers every
registered strategy except two, and says so out loud rather than filtering silently. Its note on
the second exclusion is honest and unproven, in its own words:

    `coefficient-damped` therefore still carries its own two-metric energy-AND-flux rule, NOT
    `_has_converged`. That is a NAMED hole in this file's scope, not a claim that it is covered:
    its flux metric is `max|x|`, iterate-derived and so not vulnerable in the same way, but
    nothing here proves that.

`taste.md`: "A scope claim needs its own test -- one that PLANTS the thing in the region and calls
the REAL guard." This is that test, and it REFUTES the hopeful half of the note. `max|x|` is a
single order statistic of the iterate, so it is flat over every step that redistributes the field
without moving its largest component -- which is exactly what a saturating region does while the
rest of the domain is still relaxing. Being "iterate-derived" buys nothing on its own; only a
metric that sees ALL of the iterate move can refuse a plateau, which is what `increment_ratio`
in `_has_converged` is.

WHY IT WAS EXCLUDED, AND WHY THAT IS NOT A REASON TO LEAVE IT UNCOVERED. The exclusion is real:
`coefficient-damped` under-relaxes the RELUCTIVITY inside the assembly kernel, so it needs `raw_state` /
`fold_raw` and REFUSES a plain `NonlinearSystem` with a `TypeError` -- it cannot be driven by the
sibling file's synthetic scalar map. That makes it undrivable THERE, not undrivable. A system
carrying the wider interface is thirty lines, and it is below.

THE PLANT. `x = (a, b)` with `a` pinned at 10 -- the saturated component, already at its fixed
point -- while `b` relaxes 0 -> 5 by halves. Then across every step:

* `total_energy` is held exactly constant, so the energy metric reads 0.00%;
* `max|x|` is 10 throughout, so the flux metric reads 0.00% too;
* `b` is still 2.5 away from its fixed point, so the iterate is nowhere near converged.

Both of the strategy's metrics agree the solve is finished, and both are wrong. The alpha
adaptation cannot rescue it either: alpha only ratchets on an energy-delta SIGN FLIP, and a
constant energy has delta exactly 0, so alpha sits at 1.0 and the mixing is the identity.

THE CONVERSE IS TESTED TOO, and it is the half that keeps this from being satisfied by a strategy
that simply never converges: on a system that genuinely reaches its fixed point, `coefficient-damped` must
still report `converged=True`.
"""

from __future__ import annotations

import numpy as np
import pytest

from numerics_lab.nonlinear import AssembledState, get_strategy

_A_PINNED = 10.0
_B_STAR = 5.0
_PLATEAU_ENERGY = 100.0


class _RedistributingSystem:
    """A two-component map whose energy AND `max|x|` are both constant while `b` still moves.

    Carries the wider `raw_state` / `fold_raw` interface `coefficient-damped` demands. The "raw" arrays are
    a 2x2 identity in COO plus a right-hand side holding `g(x)`, so `solve_linear` returns exactly
    `g(x)` and the iteration map is visible in one place. That shape is what makes the strategy's
    reluctivity mixing the IDENTITY here (alpha stays 1.0), which is the point: nothing about the
    damping is being tested, only the convergence rule.
    """

    is_linear = False

    def __init__(self, *, converging: bool) -> None:
        #: When False the energy is pinned flat -- the plant. When True it tracks `b`, so the
        #: energy metric is honest and the converse assertion has something to converge on.
        self._converging = converging

    def _g(self, x: np.ndarray) -> np.ndarray:
        """Halve `b`'s distance to its fixed point; `a` is already at its own and never moves."""
        return np.array([_A_PINNED, x[1] + 0.5 * (_B_STAR - x[1])], dtype=float)

    def _energy(self, x: np.ndarray) -> float:
        return _PLATEAU_ENERGY if not self._converging else _PLATEAU_ENERGY + float(x[1])

    def initial_guess(self) -> np.ndarray:
        return np.array([_A_PINNED, 0.0], dtype=float)

    def raw_state(self, x: np.ndarray):
        """`(s_row, s_col, s_val, t_mat, total_energy)` for a 2x2 identity system with rhs `g(x)`."""
        s_row = np.array([0, 1], dtype=int)
        s_col = np.array([0, 1], dtype=int)
        s_val = np.array([1.0, 1.0], dtype=float)
        return s_row, s_col, s_val, self._g(x), self._energy(x)

    def fold_raw(self, s_row, s_col, s_val, t_mat, total_energy) -> AssembledState:
        """No boundary conditions to fold; the mixed arrays pass straight through as the state."""
        return AssembledState(system_matrix=(s_row, s_col, s_val, t_mat), total_energy=total_energy)

    def solve_linear(self, state: AssembledState) -> np.ndarray:
        _row, _col, val, rhs = state.system_matrix
        return np.asarray(rhs, dtype=float) / np.asarray(val, dtype=float)

    def assemble(self, x: np.ndarray) -> AssembledState:
        return self.fold_raw(*self.raw_state(x))

    def on_iteration(self, *_args, **_kwargs) -> None:
        return None


def test_the_plant_really_is_flat_on_both_metrics():
    """THE FLOOR. Without this the refutation below could pass on a plant that never plateaued.

    Finding nothing is vacuous rather than green, so the plant is measured before it is used: two
    consecutive iterates must agree to 0.00% on BOTH of the strategy's metrics while the iterate
    itself is still far from its fixed point.
    """
    system = _RedistributingSystem(converging=False)
    x = system.initial_guess()
    for _ in range(2):
        x = system.solve_linear(system.assemble(x))
    x_next = system.solve_linear(system.assemble(x))
    assert system._energy(x) == system._energy(x_next), 'the energy metric must read exactly flat'
    assert max(abs(x)) == max(abs(x_next)) == _A_PINNED, 'the flux metric must read exactly flat'
    assert abs(x[1] - _B_STAR) > 1.0, 'the iterate must still be far from its fixed point'


def test_coefficient_damped_refuses_a_plateau_its_two_metrics_both_call_converged():
    """THE DISCRIMINATING ASSERTION, and what it refutes is a comment, not a guess.

    Both declared metrics read 0.00% while `b` is 2.5 from its fixed point. A rule that consults
    the whole iterate refuses this; the energy/flux pair cannot, because neither of its two numbers
    is a function of anything but the largest component and a scalar the system chose.
    """
    system = _RedistributingSystem(converging=False)
    # THREE passes, and the cap is a MEASUREMENT rather than a guess. `b` halves its distance
    # each step, so the distance left at the cap is `_B_STAR / 2**max_iter` -- measured
    # 2026-09-06: 1.250 / 0.625 / 0.312 / 0.156 at caps 2 / 3 / 4 / 5. Both metrics are already
    # flat by pass 2, so three is the smallest cap that leaves the iterate unmistakably in
    # flight; six (the first draft) leaves 0.078 and the floor below stops discriminating.
    result = get_strategy('coefficient-damped').solve(system, max_iter=3, rtol_energy_pct=0.01)
    assert abs(result.x[1] - _B_STAR) == pytest.approx(_B_STAR / 2**3, abs=1e-12), (
        'the plant is built so the iterate is still moving when both metrics go flat, and the '
        'distance left is EXACTLY `_B_STAR / 2**max_iter` by construction. A different value '
        'means the iteration map changed, so the assertion below would no longer be about a '
        'plateau at all'
    )
    assert not result.converged, (
        'coefficient-damped reported convergence at a point that is not a fixed point. Both of its metrics '
        'are blind here: `total_energy` is flat by construction, and `max|x|` cannot see a '
        'redistribution that leaves the largest component alone. The remedy is the one the other '
        'eleven strategies already take -- AND the increment into the rule, with `_StagnationWatch` '
        'as the escape for a genuinely stationary increment.'
    )


def test_coefficient_damped_still_converges_when_the_solve_genuinely_settles():
    """THE CONVERSE, without which "never report converged" would satisfy this file.

    Same map, honest energy: the iterate reaches its fixed point and BOTH metrics settle with it,
    so the strategy must still say so.
    """
    system = _RedistributingSystem(converging=True)
    result = get_strategy('coefficient-damped').solve(system, max_iter=60, rtol_energy_pct=0.01)
    assert result.converged, 'a genuinely converged solve must still report convergence'
    assert result.x == pytest.approx(np.array([_A_PINNED, _B_STAR]), rel=1e-3, abs=1e-6)
