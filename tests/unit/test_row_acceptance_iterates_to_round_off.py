"""Per-row acceptance iterates each row to its own round-off floor, not to a fixed 1e-10 bar.

A fixed ``ADMISSION_BAR`` of each row's scale is LOOSER than the arithmetic on a well-scaled system:
Newton's quadratic last step lands at round-off, and stopping one step earlier left a varistor's
recovered exponent at rel 2.9e-11 where the global-norm march had reached 2e-15 (motronics,
2026-11-06). The rule now: every row within its bar, and then Newton is allowed to finish -- it
stops at the row's round-off floor, or when its correction no longer reduces the scaled residual.
``row_tol`` LOOSENS the bar for cost; ``row_tol=ADMISSION_BAR`` is the old fixed bar, the planted
control every round-off claim here is measured against.
"""

from __future__ import annotations

import numpy as np
import pytest

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.linear.resolution import ADMISSION_BAR
from numerics_lab.time_integrators import integrate_transient

_STEP = TimeWindow(t0=0.0, t1=1e-3, dt=1e-3, integrator=IntegratorKind.BDF1)
_ROOT = np.log(5.0)


class _Exponential:
    """``exp(u) - 5 = 0``: a well-scaled nonlinear (algebraic) step whose root is ``ln 5``."""

    def residual(self, u, u_dot, u_ddot, t):
        return np.array([np.exp(u[0]) - 5.0])

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.array([[np.exp(u[0]) * weights.alpha0.real]])


def _exp_march(**kw):
    return integrate_transient(_Exponential(), _STEP, np.array([0.0]), max_newton=50, row_acceptance=True, **kw)


def test_a_well_scaled_nonlinear_step_reaches_round_off() -> None:
    solution = _exp_march()
    assert solution.converged
    assert abs(solution.states[-1][0] - _ROOT) <= 4e-15 * _ROOT


def test_the_fixed_admission_bar_stops_short_of_round_off() -> None:
    # The planted control: the same step at the old fixed bar is measurably off round-off.
    solution = _exp_march(row_tol=ADMISSION_BAR)
    assert solution.converged
    assert abs(solution.states[-1][0] - _ROOT) > 1e-13 * _ROOT


def test_a_loosened_row_tol_stops_earlier() -> None:
    assert _exp_march(row_tol=1e-4).newton_iterations < _exp_march().newton_iterations


class _Linear:
    def residual(self, u, u_dot, u_ddot, t):
        return np.array([u_dot[0] + 3.0 * u[0] - 7.0 * u[1], 2.0 * u[1] - 3.0])

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        a0, a1 = weights.alpha0.real, weights.alpha1.real
        return np.array([[a1 + 3.0 * a0, -7.0 * a0], [0.0, 2.0 * a0]])


def test_a_well_scaled_linear_step_reaches_round_off() -> None:
    solution = integrate_transient(_Linear(), _STEP, np.array([1.0, 0.0]), max_newton=10, row_acceptance=True)
    expected = (1.0 + 1e-3 * 10.5) / (1.0 + 3e-3)
    assert solution.converged
    assert solution.states[-1] == pytest.approx([expected, 1.5], rel=1e-14, abs=1e-15)


_WEBER = 1e-6


class _MixedUnits:
    """A 400 V row of scale ~4e8 beside a microweber linkage row coupled to it at 1e-12."""

    def residual(self, u, u_dot, u_ddot, t):
        return np.array([
            u_dot[0] + 1e6 * (u[0] - 400.0),
            u[1] + 1e6 * u[1] ** 3 - _WEBER + 1e-12 * (u[0] - 400.0),
        ])

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        a0, a1 = weights.alpha0.real, weights.alpha1.real
        return np.array([[a1 + 1e6 * a0, 0.0], [1e-12 * a0, (1.0 + 3e6 * u[1] ** 2) * a0]])


def _weber_row(state) -> float:
    return abs(state[1] + 1e6 * state[1] ** 3 - _WEBER + 1e-12 * (state[0] - 400.0))


def test_a_mixed_unit_weber_row_is_driven_to_its_own_round_off() -> None:
    def march(**kw):
        return integrate_transient(
            _MixedUnits(), _STEP, np.array([0.0, 1e-5]), max_newton=50, row_acceptance=True, **kw
        )

    solution, loose = march(), march(row_tol=1e-3)
    bar = 64 * np.finfo(float).eps * _WEBER  # the weber row's own round-off bar, scale ~1e-6 Wb
    assert solution.converged
    assert _weber_row(solution.states[-1]) <= bar
    # Planted control: a loosened bar leaves the same row measurably above it.
    assert _weber_row(loose.states[-1]) > bar


class _NoiseFloor:
    """``u - 2`` plus a 1e-12 rounding-like term: Newton plateaus ABOVE round-off, inside the bar.

    No correction can reduce it, so only the stagnation verdict ends the step before the ceiling.
    """

    def residual(self, u, u_dot, u_ddot, t):
        return np.array([u[0] - 2.0 + 1e-12 * np.sin(1e12 * u[0])])

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.array([[weights.alpha0.real]])


def test_a_stagnating_newton_stops_within_its_bar_instead_of_looping() -> None:
    solution = integrate_transient(_NoiseFloor(), _STEP, np.array([3.0]), max_newton=500, row_acceptance=True)
    final = abs(_NoiseFloor().residual(solution.states[-1], None, None, 0.0)[0])
    assert solution.converged
    assert final > 64 * np.finfo(float).eps * 4.0  # above round-off: the floor did not end it
    assert final <= ADMISSION_BAR * 4.0  # within the bar it was accepted at
    assert solution.newton_iterations <= 5


def test_gmin_continuation_composes_with_row_acceptance() -> None:
    # Three plain iterations cannot reach exp(u) = 5 from u = 0; the continuation finishes the step.
    solution = integrate_transient(
        _Exponential(),
        _STEP,
        np.array([0.0]),
        max_newton=3,
        row_acceptance=True,
        gmin_stages=(1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12),
    )
    assert solution.converged
    assert abs(solution.states[-1][0] - _ROOT) <= 4e-15 * _ROOT
