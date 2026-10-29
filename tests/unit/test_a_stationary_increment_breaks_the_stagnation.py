"""A STATIONARY increment below a plateau the energy already accepts IS convergence -- break on it.

THE MEASURED DEFECT (2026-09-01). On `case_b` the newton energy error
passes `rtol_energy_pct` by iteration ~12, but the INCREMENT pins at ~3.57e-4 -- the stationary
limit of the hanging-fold update -- so `_has_converged`'s increment floor (`rtol/100` = 1e-4) is
never met and the solver burns the whole 100-step cap: ~85 wasted iterations, ~123 s per solve.
The floor is systematically 2-4.5x tighter than the stationary increment on EVERY measured case
(case_e, case_a, case_b): the fast break's increment half is unreachable on these meshes, not
barely missed.

THE FIX (the spec). KEEP the fast break exactly as it is -- `err < rtol AND increment <
rtol/100` fires first whenever it can -- and ADD a stagnation break beside it: the energy error
has been under `rtol` for K CONSECUTIVE steps AND the increment is stationary over the same
window (max/min <= 2). K = 25 because the documented false-plateau hazard (module docstring at
`gauss/nonlinear/__init__.py`, measured on saturating B-H) is a 20+ iteration 0.00% energy plateau
-- the window must OUTLAST it, and on that hazard the increment WANDERS, so the stationarity half
is what refuses it. K = 25 fires on case_b newton at ~iteration 39. The energy half is NOT weakened:
stagnation still requires err < rtol on every step of the window.

WHAT IS PINNED HERE.
  (a) an iterate creeping at a constant 3.5e-4 increment under a flat-passing energy must converge
      at ~iteration K + first-pass, NOT at the cap;
  (b) the documented false plateau -- energy 0.00% for the whole run, increment WANDERING by more
      than the 2x band -- must NOT stagnation-break;
  (b2) an increment CREEPING UP inside the 2x band must not stagnation-break either. The band test
      alone cannot see a trend: a 0.7%-per-step rise stays within max/min <= 2 for a hundred
      iterations while diverging the whole time. MEASURED 2026-09-06 via ``downstream run
cases/case_a --select mec,quad`` -- 1.91e-3 -> 2.05e-3 over ten steps, band
      ratio 1.07, and the break fired, returning ``converged = True`` at 20x the increment bar.
      STATIONARY was the word the docstring used and a BAND was all it tested;
  (c) a system already at its fixed point still converges in a handful of iterations, long before
      the window can fill: the fast break still fires first.

WHY ONLY TWO STRATEGIES ARE DRIVEN. The test controls the increment by controlling the STEP the
strategy takes, which is only true for the strategies whose step IS the Picard map's (`picard`,
`picard-damped` at alpha=1). `newton` was the third until 2026-09-25, driven through the Picard
step it took on a `None` tangent; it now REFUSES a tangent-less system
(`test_a_flat_energy_plateau_is_not_convergence::test_newton_refuses_a_system_with_no_tangent`), and
it judges the residual rather than the energy, so the case_b newton creep this file measured is no
longer reachable through it. Anderson / the quasi-Newton family mix their own steps from history, so a synthetic
map cannot pin their increment -- the same reason `_PlateauSystem` next door drives them only for
the NOT-converged direction. `coefficient-damped` and `continuation` are undrivable here for the reasons
`test_a_flat_energy_plateau_is_not_convergence` names.
"""

from __future__ import annotations

import numpy as np
import pytest

from numerics_lab.nonlinear import AssembledState, get_strategy
from numerics_lab.nonlinear._seam import _STAGNATION_WINDOW as _K

_DRIVABLE = ['picard', 'picard-damped']
_ENERGY = 100.0


class _CreepingSystem:
    """A contraction toward 0 whose RELATIVE step is pinned at ``step`` and whose energy is flat.

    ``g(x) = x * (1 - step)`` gives ``increment_ratio(x, g(x)) == step`` exactly, for every
    strategy whose step is the map's own. The energy never changes, so after the first pass
    (``err = inf``) the energy error is 0.00% -- under any ``rtol`` -- forever. With
    ``step = 3.5e-4`` this is the measured case_b newton tail reduced to its two load-bearing
    numbers: energy passes, increment sits 3.5x above the fast break's floor.
    """

    is_linear = False

    def __init__(self, steps) -> None:
        # ``steps``: one ratio, or a cycled sequence of them (the wandering plateau).
        self._steps = tuple(np.atleast_1d(steps).astype(float))
        self._x = 0.0
        self._n = 0

    def initial_guess(self) -> np.ndarray:
        return np.array([1.0])

    def assemble(self, x, /) -> AssembledState:
        self._x = float(np.asarray(x)[0])
        return AssembledState(system_matrix=None, total_energy=_ENERGY)

    def solve_linear(self, state, /) -> np.ndarray:
        step = self._steps[self._n % len(self._steps)]
        self._n += 1
        return np.array([self._x * (1.0 - step)])

    def residual(self, x, state, /) -> np.ndarray:
        return np.asarray(x)

    def residual_norm(self, x, state, /) -> float:
        return float(abs(float(np.asarray(x)[0])))

    def tangent(self, x, state, /):
        return None  # no Jacobian: a Newton strategy refuses this system

    def solve_secant(self, state, rhs, /):
        return np.asarray(rhs, dtype=float)

    def on_iteration(self, i, state, err, alpha, increment, /) -> None:
        return None


class _CreepingUpSystem(_CreepingSystem):
    """Energy equally flat; the increment RISES geometrically, slowly enough to stay in the band.

    ``1.007 ** 24 = 1.18``, so a full 25-step window never leaves the 2x band -- the arrangement
    that made the measured case_a tail read as stationary. The ramp is long enough that it
    never cycles inside the run, so every window the watch sees is genuinely rising.
    """

    def __init__(self) -> None:
        super().__init__([1.5e-3 * 1.007**k for k in range(400)])


class _AtFixedPointSystem(_CreepingSystem):
    """Already AT the fixed point: ``g(x) == x``, increment exactly 0 from the first step."""

    def __init__(self) -> None:
        super().__init__([0.0])


@pytest.mark.parametrize('name', _DRIVABLE)
def test_a_stationary_increment_under_a_passing_energy_converges_before_the_cap(name: str):
    """(a) The case_b newton shape: converged by ~iteration K+2, not capped at 100."""
    window = _K  # the implementation's own K, so this test reads the constant, not a copy of it
    result = get_strategy(name).solve(_CreepingSystem([3.5e-4]), max_iter=100)
    assert result.converged, (
        f'{name} ran a stationary 3.5e-4-increment tail to the cap: energy passed rtol from the '
        f'second step and the increment never moved (max/min = 1), yet {result.iterations}/100 '
        f'iterations ran -- the measured ~85-wasted-iteration case_b burn'
    )
    assert result.iterations <= window + 2, (
        f'{name} took {result.iterations} iterations to stagnation-break; the window is {window} '
        f'steps plus the first (err=inf) pass and the delayed-break pass, so anything later means '
        f'the window is not what the spec fixed'
    )


@pytest.mark.parametrize('name', _DRIVABLE)
def test_a_wandering_increment_on_a_flat_energy_never_stagnation_breaks(name: str):
    """(b) The documented false-plateau hazard: energy 0.00% for 20+ iterations, increment
    WANDERING past the 2x band. The stagnation break must refuse it -- the energy half of the
    rule alone would have fired here, which is exactly the hazard the increment half exists for.
    """
    cap = _K + 20  # the window fills and refills several times over; a miss is a real refusal
    result = get_strategy(name).solve(_CreepingSystem([2.0e-4, 8.0e-4]), max_iter=cap)
    assert not result.converged, (
        f'{name} reported convergence on a flat-energy plateau whose increment wandered 4x across '
        f'the window -- the saturating-BH false plateau, which must run to its cap or escape, '
        f'never be called converged'
    )
    assert result.iterations == cap


@pytest.mark.parametrize('name', _DRIVABLE)
def test_the_fast_break_still_fires_first(name: str):
    """(c) A system already at its fixed point converges in a few iterations -- long before the
    stagnation window can fill -- so the added break has not displaced or weakened the fast one.
    """
    result = get_strategy(name).solve(_AtFixedPointSystem(), max_iter=100)
    assert result.converged
    assert result.iterations <= 5, (
        f'{name} took {result.iterations} iterations on a system started AT its fixed point; the '
        f'fast break (err < rtol AND increment < rtol/100) should have fired almost immediately'
    )


@pytest.mark.parametrize('name', _DRIVABLE)
def test_a_creeping_rise_inside_the_band_never_stagnation_breaks(name: str):
    """(b2) A trend the band cannot see. The break says STATIONARY; a monotone rise is not.

    The FLOOR for this addition is test (a) above, unchanged and still green: an increment pinned
    at a constant 3.5e-4 still breaks at ~K+2. A trend test that also refused (a) would have
    deleted the capability rather than sharpened it.
    """
    cap = _K + 20  # several full windows, every one of them rising
    result = get_strategy(name).solve(_CreepingUpSystem(), max_iter=cap)
    assert not result.converged, (
        f'{name} reported convergence on an increment that rose every single step -- it stayed '
        f'inside the 2x band the whole time, which is why the band alone is not stationarity'
    )
    assert result.iterations == cap
