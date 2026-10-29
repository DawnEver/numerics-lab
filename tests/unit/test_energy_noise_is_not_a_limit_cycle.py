"""Alpha may not be ratcheted down by the energy's own floating-point noise.

THE MECHANISM, MEASURED. `DampedPicardStrategy` damps alpha on an energy-delta SIGN FLIP, which
exists to catch a period-2 limit cycle (a sharp BH knee overshooting every iteration): the plain
`err > prev_err` rule cannot see one, because a limit cycle has a roughly CONSTANT relative error.

But the trigger had no amplitude threshold. Once the energy settles, `delta` is the difference of
two nearly equal floats and its sign alternates at random -- so the ratchet fires on NOISE, every
step, forever. Measured on case_a mec:quad through the production CLI: the energy error
reached 0.00% early, alpha was driven to `1.0 x 0.5 x 0.6**3 = 0.108` and pinned there, and every
one of the 8 rotor positions then spent 44-61 iterations at ~3 s crawling to the increment bar at a
tenth of the step size. From outside the process that is indistinguishable from a hang.

It also silently disabled the recovery leg added alongside it: recovery requires a run of
UNDISTURBED steps, and a noise-driven ratchet means there is never one.

WHAT IS PINNED. A limit cycle is by definition an oscillation that does NOT decay, so both ratchets
now require the relative energy change to be at or above the convergence tolerance. Below it there
is nothing to stabilise. The converse is pinned too, in the second test: a REAL oscillation, whose
amplitude stays far above tolerance, must still damp alpha -- otherwise "never ratchet" would pass
the first test and the limit-cycle protection would be gone.
"""

from __future__ import annotations

import numpy as np

from numerics_lab.nonlinear import AssembledState, get_strategy

_RTOL = 0.01  # percent -- the production default


class _RecordingSystem:
    """Base: drives a contraction toward ``x* = 10`` and records the alpha it was driven with."""

    is_linear = False
    _ENERGY = 500.0

    def __init__(self) -> None:
        self.alphas: list[float] = []
        self.seen: list[float] = []

    def initial_guess(self) -> np.ndarray:
        return np.zeros(1)

    def _energy(self, k: int) -> float:
        raise NotImplementedError

    def assemble(self, x, /) -> AssembledState:
        self.seen.append(float(x[0]))
        return AssembledState(system_matrix=None, total_energy=self._energy(len(self.seen) - 1))

    def solve_linear(self, state, /) -> np.ndarray:
        return np.array([self.seen[-1] + 0.5 * (10.0 - self.seen[-1])])

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
        self.alphas.append(float(alpha))

    def scale_source(self, lam, /):
        raise NotImplementedError


class _NoisyButSettledSystem(_RecordingSystem):
    """Energy constant to within 1e-9 percent, its delta ALTERNATING in sign every step.

    That is the floating-point noise floor, reproduced deterministically so the test does not
    depend on real rounding. `1e-11` against `500.0` is a relative change of 2e-12 -- ten orders of
    magnitude below the 0.01% tolerance, so no reading of "the energy is still moving" survives it.
    """

    def _energy(self, k: int) -> float:
        return self._ENERGY + (1e-11 if k % 2 else -1e-11)


class _GenuineLimitCycleSystem(_RecordingSystem):
    """A REAL period-2 oscillation: amplitude 20% of the energy, far above any tolerance."""

    def _energy(self, k: int) -> float:
        return self._ENERGY * (1.2 if k % 2 else 0.8)


def test_energy_noise_does_not_drive_alpha_to_the_floor():
    """The discriminating assertion: measured RED before the amplitude threshold (alpha -> 0.1)."""
    system = _NoisyButSettledSystem()
    get_strategy('picard-damped').solve(system, max_iter=25, rtol_energy_pct=_RTOL)
    worst = min(system.alphas)
    assert worst == 1.0, (
        f'alpha was ratcheted to {worst} by an energy oscillation of relative size 2e-12, twelve '
        f'orders of magnitude inside the {_RTOL}% tolerance. That is the float noise floor, not a '
        f'limit cycle, and damping for it costs ~10x the iterations at a tenth of the step size'
    )


def test_a_real_limit_cycle_still_damps_alpha():
    """The converse. Without it, deleting the sign-flip ratchet entirely would pass the test above."""
    system = _GenuineLimitCycleSystem()
    get_strategy('picard-damped').solve(system, max_iter=25, rtol_energy_pct=_RTOL)
    worst = min(system.alphas)
    assert worst < 1.0, (
        'a period-2 energy oscillation at 20% amplitude did not damp alpha at all -- the limit-cycle '
        'protection the sign-flip ratchet exists for is gone, not merely threshold-gated'
    )
