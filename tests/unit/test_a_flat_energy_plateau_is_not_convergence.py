"""A flat ``total_energy`` is not a fixed point, and EVERY strategy must say so.

WHAT THE CODE ALREADY KNEW. `DampedPicardStrategy`'s own docstring records the failure mode, with
a measurement:

    a heavily-damped iterate can sit on a near-flat plateau of ``total_energy`` that is NOT the
    true Picard fixed point (measured: a saturating-BH case read 0.00% energy change for 20+
    iterations at a spurious energy level ~14x the physical one ...)

WHY THAT PROTECTION DOES NOT COVER THE CASE IT DESCRIBES. It is guarded by `if not ratcheted`, so
the extra increment test runs ONLY on a solve whose alpha happened to be damped at some earlier,
unrelated moment. The same docstring says so out loud -- "A run that never ratchets (today's
common case) never evaluates this extra test". And of the twelve strategies in this module, ELEVEN
never had it at all: the plateau is a property of the ENERGY CRITERION, not of damped Picard, but
the protection was written into the class where the bug was found rather than into the criterion.

So a strategy can return `converged=True` at a point that is not a solution, and the caller cannot
tell -- the shape AGENTS.md names as the dominant defect class.

WHAT IS PINNED. On a system that is CONSTRUCTED to hold its energy exactly constant while the
iterate is still far from the fixed point, no strategy may report convergence. The energy staircase
here is synthetic on purpose: it isolates the criterion from any physics, so a strategy that passes
is passing because it checks the ITERATE, not because this particular BH curve was kind to it.

The converse test matters as much: on a genuinely converged system every strategy must STILL report
convergence, or "never report converged" would pass this file.
"""

from __future__ import annotations

import numpy as np
import pytest

from numerics_lab.nonlinear import ITERATION_REGISTRY, AssembledState, NewtonTangentUnavailable, get_strategy

_X_STAR = 10.0
_PLATEAU_ENERGY = 100.0

# EVERY registered strategy, derived from the registry rather than listed: a strategy added later
# is covered without anyone remembering to add it here, which is the whole point -- the protection
# used to exist in exactly one class because it was written where the bug was found.
# Two exclusions, BY NAME and with their reasons -- never a silent filter, which is how a
# registry-derived list quietly stops covering anything:
#   `continuation` wraps another strategy and is driven through `scale_source`;
#   `coefficient-damped` under-relaxes the RELUCTIVITY inside the assembly kernel, so it needs a wider
#     interface than the `NonlinearSystem` Protocol (`raw_state` / `fold_raw`) and REFUSES a system
#     that lacks it with a TypeError -- it cannot be driven by a synthetic scalar map at all.
# `coefficient-damped` therefore still carries its own two-metric energy-AND-flux rule, NOT `_has_converged`.
# That is a NAMED hole in this file's scope, not a claim that it is covered: its flux metric is
# `max|x|`, iterate-derived and so not vulnerable in the same way, but nothing here proves that.
#   `newton` / `newton-linesearch` (the ONE Newton, two names) REFUSE a system with no tangent since
#     2026-09-25 -- this fixture offers none, so they are held to the same two properties on
#     `_TangentPlateauSystem` below instead, and their refusal is pinned too.
_NEWTON = ('newton', 'newton-linesearch', 'newton-held-factor')
_UNDRIVABLE = {'continuation', 'coefficient-damped', *_NEWTON}
_STRATEGIES = sorted(set(ITERATION_REGISTRY) - _UNDRIVABLE)


class _PlateauSystem:
    """A contraction whose energy is FLAT over the stretch the iterate is still crossing.

    ``g(x) = x + 0.5 * (x* - x)`` converges to ``x* = 10`` from ``x0 = 0`` as 0, 5, 7.5, 8.75 ...
    Energy is a staircase pinned at a single value until the iterate is nearly there, so two
    CONSECUTIVE iterates report a 0.00% energy change while the iterate is still 5.0 away from the
    solution. Nothing about this is exotic: it is the measured saturating-BH plateau, reduced to
    the one property that produces the wrong answer.
    """

    is_linear = False

    def __init__(self) -> None:
        self.seen: list[float] = []

    def initial_guess(self) -> np.ndarray:
        return np.zeros(1)

    def _energy(self, x: float) -> float:
        return _PLATEAU_ENERGY if x < 9.0 else _PLATEAU_ENERGY + (x - 9.0)

    def assemble(self, x, /) -> AssembledState:
        self.seen.append(float(x[0]))
        return AssembledState(system_matrix=None, total_energy=self._energy(float(x[0])))

    def solve_linear(self, state, /) -> np.ndarray:
        x = self.seen[-1]
        return np.array([x + 0.5 * (_X_STAR - x)])

    def residual(self, x, state, /) -> np.ndarray:
        return np.asarray(x) - _X_STAR

    def residual_norm(self, x, state, /) -> float:
        return float(abs(float(np.asarray(x)[0]) - _X_STAR))

    def tangent(self, x, state, /):
        return None

    def solve_system(self, matrix, rhs, /):
        # No tangent is offered (`tangent` returns None): Newton refuses the system, modified-newton
        # takes the Picard substitution, and the quasi-Newton family solves the SECANT system below.
        return np.asarray(rhs, dtype=float) / float(matrix)

    def solve_secant(self, state, rhs, /):
        # H0 = I: the secant operator for this scalar map. Enough for Broyden / BFGS / L-BFGS to
        # take real steps, which is what makes them exercise the convergence test at all.
        return np.asarray(rhs, dtype=float)

    def on_iteration(self, i, state, err, alpha, increment, /) -> None:
        return None

    def scale_source(self, lam, /):
        raise NotImplementedError


class _ConvergedSystem(_PlateauSystem):
    """Already AT the fixed point: the converse. Every strategy must still say converged."""

    def initial_guess(self) -> np.ndarray:
        return np.array([_X_STAR])


@pytest.mark.parametrize('name', _STRATEGIES)
def test_a_constant_energy_is_not_reported_as_convergence(name: str):
    """The discriminating assertion. Measured RED before the fix on EVERY strategy here."""
    system = _PlateauSystem()
    result = get_strategy(name).solve(system, max_iter=3)
    x = float(result.x[0])
    assert not (result.converged and abs(x - _X_STAR) > 1e-3), (
        f'{name} reported converged=True at x={x:.4f}, which is {abs(x - _X_STAR):.4f} away from '
        f'the fixed point x*={_X_STAR}. It read a flat total_energy as a solution -- the plateau '
        f"this module's own docstring measured at 14x the physical energy level"
    )


@pytest.mark.parametrize('name', _STRATEGIES)
def test_a_genuine_fixed_point_is_still_reported_as_converged(name: str):
    """The converse. Without it, a strategy that NEVER converges would pass the test above."""
    result = get_strategy(name).solve(_ConvergedSystem(), max_iter=10)
    assert result.converged, (
        f'{name} did not converge on a system started exactly AT its fixed point, so the guard '
        f'above proves nothing -- it would pass on a strategy that never converges at all'
    )
    assert abs(float(result.x[0]) - _X_STAR) < 1e-9


class _TangentPlateauSystem(_PlateauSystem):
    """The same flat-energy plateau WITH the Jacobian of its residual (``r = x - x*``, ``J = 1``).

    What Newton needs to run at all. Its convergence is judged on the RESIDUAL, so the flat energy
    cannot fool it by construction -- and this file is where that is shown rather than assumed.
    """

    def __init__(self, x0: float = 0.0) -> None:
        super().__init__()
        self._x0 = x0

    def initial_guess(self) -> np.ndarray:
        return np.array([self._x0])

    def tangent(self, x, state, /):
        return 1.0


@pytest.mark.parametrize('name', _NEWTON)
def test_newton_refuses_a_system_with_no_tangent(name: str):
    """UNSUPPORTED-RAISES: the Picard step it used to take ran another algorithm under its name."""
    with pytest.raises(NewtonTangentUnavailable, match='_PlateauSystem'):
        get_strategy(name).solve(_PlateauSystem(), max_iter=3)


@pytest.mark.parametrize('name', _NEWTON)
def test_newton_on_the_plateau_lands_on_the_root_and_says_so(name: str):
    """Both properties at once: no converged=True short of x*, and converged at it."""
    result = get_strategy(name).solve(_TangentPlateauSystem(), max_iter=3)
    x = float(result.x[0])
    assert not (result.converged and abs(x - _X_STAR) > 1e-3), f'{name} converged at {x}'
    assert result.converged, f'{name} did not converge on a linear residual'
    assert abs(x - _X_STAR) < 1e-9
    at_root = get_strategy(name).solve(_TangentPlateauSystem(_X_STAR), max_iter=10)
    assert at_root.converged
    assert abs(float(at_root.x[0]) - _X_STAR) < 1e-9
