"""Alpha may only GROW while the iterate is contracting -- the energy alone cannot license it.

MEASURED 2026-09-06 on the `[sparse]` leg (MEC reluctance loop, `picard-damped`), reproduced through
``downstream run cases/case_a --select mec,quad --max-num-iter 600``::

    Iteration: 117/600, ... Energy Error Percentage: 0.00%, alpha: 0.137, increment: 2.03e-04
    Iteration: 118/600, ... Energy Error Percentage: 0.00%, alpha: 0.137, increment: 1.95e-04
    Iteration: 119/600, ... Energy Error Percentage: 0.00%, alpha: 0.205, increment: 1.89e-04
    Iteration: 120/600, ... Energy Error Percentage: 0.00%, alpha: 0.205, increment: 3.70e-04
    Iteration: 121/600, ... Energy Error Percentage: 0.00%, alpha: 0.205, increment: 7.59e-04
    Iteration: 122/600, ... Energy Error Percentage: 0.00%, alpha: 0.308, increment: 1.40e-03
    Iteration: 123/600, ... Energy Error Percentage: 0.01%, alpha: 0.154, increment: 3.91e-03

The iterate was CONTRACTING toward the 1e-4 increment bar (2.03e-4 -> 1.95e-4 -> 1.89e-4) when the
recovery leg grew alpha, and the bigger step blew the increment out ~20x. The energy error is at
its own noise floor throughout, so `meaningful` is False on every one of those steps and NOTHING
opposed the growth: both shrink ratchets are deliberately deaf below rtol, and the recovery leg was
listening to that same deaf channel. The cycle repeats until a blow-out big enough to move the
ENERGY finally re-arms a shrink. At the production cap of 100 the run stops mid-blow-out and
reports `converged = False` on numbers that were, twenty iterations earlier, nearly converged.

Same class as the SDM/reduced-order defect fixed in 9507db6a6 -- energy reads converged while the
increment does not fall -- with a different mechanism: there the MAP changed shape each step; here
the STEP SCHEDULE does, and the solve inflicts it on itself.

THE FIX IS ONE REGISTRY, `_IncrementTrend`, consulted by both damped Picard legs instead of each
spelling half of it. Gating the ENERGY shrinks on `meaningful` stays exactly as it is and is right
and measured (below tolerance the energy delta's sign is floating-point noise, and a noise-driven
shrink is what pinned alpha at 0.108 on case_a). What changes is that the INCREMENT now
speaks on that same plateau, in both directions: one non-contracting step refuses a recovery, and
`_INCREMENT_RISE_STEPS` consecutive rises force a shrink.

BLOCKING THE GROWTH ALONE WAS MEASURED TO BE HALF A FIX, and that measurement is why the shrink is
here. With only the refusal, the same CLI run parked at alpha 0.150 with the increment creeping UP
(1.91e-3 -> 2.05e-3 over ten steps) -- no longer cycling, still not contracting, and nothing left
that could move alpha. Worse, `_StagnationWatch` then read that 1.07 band ratio as STATIONARY and
returned `converged = True` at 20x the increment bar, turning an honest refusal into a false
success. Both halves landed together for that reason; the watch's own trend test is pinned in
`test_a_stationary_increment_breaks_the_stagnation.py` case (b2).

MEASURED END TO END on the same command, `[sparse]` / `mec:quad` / case_a: 100/100 with
`converged = False` became 50/100 with `converged = True` at increment 7.58e-05, total energy
1.7065e+02 -> 1.7064e+02 (the same fixed point, now actually reached). case_a: 73 -> 42 iterations,
1.5489e+02 unchanged to five figures.

The tests below are a discrimination set, not one assertion. The plateau case pins the refusal; the
contracting case pins the FLOOR, because a gate that never let alpha grow would pass the first
alone and would re-introduce the crawl `_ALPHA_CALM_STEPS` exists to remove.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from numerics_lab.nonlinear import AssembledState, get_strategy

pytestmark = pytest.mark.unit

# Three energies that force ONE shrink (err 50% -> 60%, both far above rtol), then the plateau --
# the shape measured above. After the shrink alpha is 0.5 with alpha_cap 1.0, so the recovery leg
# is armed and the only open question is what licenses it.
_ENERGIES = (100.0, 200.0, 500.0)
_PLATEAU = 500.0

#: The plateau is not EXACTLY flat: a geometrically decaying jitter reproduces the measured
#: reading -- `0.00%` at two decimals, well under rtol, hence `meaningful` is False -- while the
#: energy still falls strictly each step, so nothing on the energy side refuses a recovery.
_PLATEAU_JITTER = 1e-6


def _plateau_energy(k: int) -> float:
    return _PLATEAU * (1.0 + _PLATEAU_JITTER * 0.5**k)


class _ScriptedIncrement:
    """A system whose energy plateaus and whose INCREMENT follows a script.

    ``solve_linear`` returns ``x * (1 + s)``, so ``increment_ratio(x, x_solve)`` is exactly ``|s|``
    whatever alpha did to ``x``. The increment is DRIVEN rather than observed, which is what lets
    one fixture isolate a wandering plateau from a contracting one on identical energies.
    """

    is_linear = False

    def __init__(self, script: tuple[float, ...]) -> None:
        self._script = script
        self._assembled = 0
        self._solved = 0
        self._x = np.ones(1)
        self.alphas: list[float] = []
        self.increments: list[float] = []

    def initial_guess(self) -> np.ndarray:
        return np.ones(1)

    def assemble(self, x, /) -> AssembledState:
        # The strategies hand `solve_linear` only the assembled state, so the current iterate is
        # captured here -- `assemble(x)` runs immediately before it every step, in every strategy.
        self._x = np.asarray(x, dtype=float)
        i = self._assembled
        self._assembled += 1
        energy = _ENERGIES[i] if i < len(_ENERGIES) else _plateau_energy(i - len(_ENERGIES))
        return AssembledState(system_matrix=None, total_energy=energy, payload=None)

    def solve_linear(self, state, /) -> np.ndarray:
        s = self._script[min(self._solved, len(self._script) - 1)]
        self._solved += 1
        return self._x * (1.0 + s)

    def residual(self, x, state, /) -> np.ndarray:
        return np.zeros(1)

    def residual_norm(self, x, state, /) -> float:
        return 0.0

    def tangent(self, x, state, /):
        return None

    def on_iteration(self, i, state, err, alpha, increment, /) -> None:
        self.alphas.append(float(alpha))
        self.increments.append(float(increment))


class _Plateau(_ScriptedIncrement):
    """Energy flat, increment GROWING -- the measured `[sparse]` blow-out."""

    def __init__(self) -> None:
        super().__init__((
            2.0e-2,
            2.0e-2,
            2.0e-2,
            2.2e-2,
            2.6e-2,
            3.4e-2,
            5.0e-2,
            8.0e-2,
            1.3e-1,
            2.0e-1,
            3.0e-1,
            4.5e-1,
        ))


class _Contracting(_ScriptedIncrement):
    """Energy equally flat, increment FALLING -- the smooth tail alpha recovery exists for."""

    def __init__(self) -> None:
        super().__init__((
            2.0e-2,
            2.0e-2,
            2.0e-2,
            1.8e-2,
            1.6e-2,
            1.4e-2,
            1.2e-2,
            1.0e-2,
            9.0e-3,
            8.0e-3,
            7.0e-3,
            6.0e-3,
        ))


def _alphas(system: _ScriptedIncrement, strategy: str) -> list[float]:
    get_strategy(strategy).solve(system, max_iter=12)
    return system.alphas


def test_a_growing_increment_refuses_the_alpha_recovery_on_an_energy_plateau() -> None:
    """The measured defect: the energy says nothing, the iterate says it is getting WORSE, alpha grows."""
    alphas = _alphas(_Plateau(), 'picard-damped')
    assert min(alphas) < 1.0, f'the fixture never armed the recovery leg: {alphas}'
    # The script's increment starts rising at index 3, so index 4 is the first step that could act
    # on it. From there alpha may only fall: never a recovery, and -- once the rise has run for
    # `_INCREMENT_RISE_STEPS` -- a damping.
    tail = alphas[4:]
    assert all(b <= a + 1e-12 for a, b in itertools.pairwise(tail)), f'alpha grew while the increment grew: {alphas}'
    assert tail[-1] < tail[0], f'a sustained rise in the increment never damped alpha: {alphas}'


def test_a_contracting_increment_still_recovers_alpha() -> None:
    """THE FLOOR. Identical plateau energy; a gate that refused this would re-introduce the crawl."""
    alphas = _alphas(_Contracting(), 'picard-damped')
    tail = alphas[3:]
    assert max(tail) > min(tail), f'alpha never recovered on a contracting march: {alphas}'
