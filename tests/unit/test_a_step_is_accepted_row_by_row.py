"""A per-row step acceptance cannot let a large row carry a small one (ruling 2026-10-14).

Two decoupled rows: a stiff 1e6-scale decay and an algebraic ``(u1 - c)^3 = 0`` whose Newton
converges only linearly. The norm bar ``tol_rel * |r_0|`` is set by the large row, so it accepts the
small row far from its root -- the planted control. The per-row bar holds every row at
``ADMISSION_BAR`` of its own scale.
"""

from __future__ import annotations

import numpy as np
import pytest

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import integrate_transient

_C = 2.0


class _TwoScales:
    def residual(self, u, u_dot, u_ddot, t):
        return np.array([u_dot[0] + 1e6 * u[0] - 1e6, (u[1] - _C) ** 3])

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        a0, a1 = weights.alpha0.real, weights.alpha1.real
        return np.diag([1e6 * a0 + a1, 3.0 * (u[1] - _C) ** 2 * a0])


_STUDY = TimeWindow(t0=0.0, t1=2e-6, dt=1e-6, integrator=IntegratorKind.BDF1)


def _march(**bars):
    return integrate_transient(_TwoScales(), _STUDY, np.array([0.0, _C + 1.0]), max_newton=200, backtrack_cap=0, **bars)


def test_the_norm_bar_accepts_the_small_row_far_from_its_root() -> None:
    solution = _march(tol_rel=1e-12)
    assert abs(solution.states[1][1] - _C) > 1e-3


def test_the_row_bar_holds_every_row_at_its_own_scale() -> None:
    solution = _march(row_acceptance=True)
    assert solution.converged
    assert abs(solution.states[1][1] - _C) < 1e-9


def test_the_row_bar_is_the_only_bar() -> None:
    with pytest.raises(ValueError, match='only bar'):
        _march(row_acceptance=True, tol_rel=1e-12)


class _SlowLinkage:
    """``lambda' = 1e-3``: a large stored state whose rate is small, over a short step.

    BDF builds the rate from the state, ``lambda' = alpha1 (lambda - origin) + h``, so the row can
    resolve no better than ``eps alpha1 |lambda|`` (2.2e-9 here) however small the rate it carries --
    the case_a zero-vector step (2026-10-15): a linkage row at a 1.85 us step held to 1.2e-13.
    """

    def residual(self, u, u_dot, u_ddot, t):
        return np.array([u_dot[0] - 1e-3])

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.array([[weights.alpha1.real]])


def test_a_rate_row_is_judged_at_the_scale_its_rate_is_built_from() -> None:
    study = TimeWindow(t0=0.0, t1=3e-7, dt=1e-7, integrator=IntegratorKind.BDF1)
    solution = integrate_transient(_SlowLinkage(), study, np.array([1.0]), max_newton=20, row_acceptance=True)
    assert solution.converged
    assert solution.states[-1][0] == pytest.approx(1.0 + 3e-10, abs=1e-14)
