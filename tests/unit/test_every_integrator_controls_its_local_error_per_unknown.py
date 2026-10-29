"""Local-truncation-error step control, for every integrator, with per-unknown tolerances.

``integrate_transient(step_control=StepTolerance(rtol, atol))`` sizes each step from a
predictor-corrector estimate of the scheme's own local error, unknown by unknown. The claims:

* the controller works for EVERY registered integrator: the true error tracks the tolerance
  (tightening it 100x shrinks the error) while the step count stays far below a grid that would
  reach the same error everywhere;
* the tolerance is PER UNKNOWN: a vector ``atol`` that is tight on one unknown takes more steps
  than the same scalar loosened on all -- a scalar path could not express it;
* the planted control: without ``step_control`` the march is the plain ``study.dt`` grid;
* an undeclared discontinuity cannot be resolved at ``h_min`` and RAISES with the state reached.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import StepTolerance, integrate_transient
from numerics_lab.time_integrators._integrators import get_integrator


@pytest.mark.parametrize('kind', [IntegratorKind.BDF1, IntegratorKind.TRAPEZOID, IntegratorKind.TR_BDF2])
def test_first_order_rate_does_not_cancel_a_large_constant_state(kind) -> None:
    integrator = get_integrator(kind)
    previous = np.array([6.0])
    state = integrator.start(previous, 0.0, np.zeros(1))
    form = integrator.prepare(state, 1e-5, 1e-5)
    current = np.nextafter(previous, np.inf)
    expected = form.weights.alpha1.real * (current - previous)
    np.testing.assert_array_equal(form.u_dot(current), expected)


@pytest.mark.parametrize('next_dt', [1e-5, 2e-5])
def test_bdf2_history_keeps_one_ulp_rate_on_uniform_and_variable_spacing(next_dt) -> None:
    integrator = get_integrator(IntegratorKind.BDF2)
    previous = np.array([6.0])
    state = integrator.start(previous, 0.0)
    form = integrator.prepare(state, 1e-5, 1e-5)
    state = integrator.advance(state, previous.copy(), 1e-5, 1e-5, form)
    form = integrator.prepare(state, next_dt, 1e-5 + next_dt)
    current = np.nextafter(previous, np.inf)
    np.testing.assert_array_equal(form.u_dot(current), form.weights.alpha1.real * (current - previous))


class _Decay:
    """``u_i' = -lam_i u_i`` (first order) -- a fast and a slow mode, decoupled."""

    lam = np.array([40.0, 1.0])

    def residual(self, u, u_dot, u_ddot, t):
        return u_dot + self.lam * u

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.diag(weights.alpha0.real * self.lam + weights.alpha1.real)


class _Oscillator:
    """``u'' + w^2 u = 0`` (second order), ``u(0) = 1``, ``u'(0) = 0``."""

    w2 = 4.0

    def residual(self, u, u_dot, u_ddot, t):
        return u_ddot + self.w2 * u

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.atleast_2d(weights.alpha0.real * self.w2 + weights.alpha2.real)


_FIRST_ORDER = [IntegratorKind.BDF1, IntegratorKind.BDF2, IntegratorKind.TRAPEZOID, IntegratorKind.TR_BDF2]
_SECOND_ORDER = [IntegratorKind.NEWMARK, IntegratorKind.GENERALIZED_ALPHA]


def _run(kind, rtol, atol=1e-12, dt=0.5, t1=4.0):
    if kind in _SECOND_ORDER:
        sol = integrate_transient(
            _Oscillator(),
            TimeWindow(t0=0.0, t1=t1, dt=dt, integrator=kind),
            np.array([1.0]),
            step_control=StepTolerance(rtol=rtol, atol=atol),
        )
        exact = [np.array([math.cos(2.0 * t)]) for t in sol.times]
    else:
        sol = integrate_transient(
            _Decay(),
            TimeWindow(t0=0.0, t1=t1, dt=dt, integrator=kind),
            np.ones(2),
            step_control=StepTolerance(rtol=rtol, atol=atol),
        )
        exact = [np.exp(-_Decay.lam * t) for t in sol.times]
    err = max(float(np.max(np.abs(u - e))) for u, e in zip(sol.states, exact, strict=True))
    return sol, err


@pytest.mark.parametrize('kind', _FIRST_ORDER + _SECOND_ORDER)
def test_the_error_follows_the_tolerance_for_every_integrator(kind) -> None:
    loose, err_loose = _run(kind, 1e-3)
    tight, err_tight = _run(kind, 1e-5)
    report = f'{kind.value}: rtol 1e-3 -> {len(loose.times)} pts err {err_loose:.2e}; rtol 1e-5 -> {len(tight.times)} pts err {err_tight:.2e}'
    print(report)  # noqa: T201
    assert err_tight < 0.2 * err_loose
    assert len(tight.times) > len(loose.times)
    assert err_tight < 5e-3


@pytest.mark.parametrize('kind', _FIRST_ORDER)
def test_the_step_grows_on_the_smooth_tail(kind) -> None:
    sol, _err = _run(kind, 1e-4)
    steps = np.diff(sol.times)
    assert steps[-1] > 20.0 * steps[1]


def test_a_per_unknown_atol_is_obeyed_unknown_by_unknown() -> None:
    """Tight only on the slow mode's absolute floor: more steps than loose everywhere, fewer than tight everywhere."""
    loose, _ = _run(IntegratorKind.BDF2, rtol=1e-9, atol=1e-3)
    mixed, _ = _run(IntegratorKind.BDF2, rtol=1e-9, atol=np.array([1e-3, 1e-6]))
    tight, _ = _run(IntegratorKind.BDF2, rtol=1e-9, atol=1e-6)
    assert len(loose.times) < len(mixed.times) <= len(tight.times)


def test_planted_control_without_step_control_the_march_is_the_dt_grid() -> None:
    sol = integrate_transient(_Decay(), TimeWindow(t0=0.0, t1=4.0, dt=0.5, integrator=IntegratorKind.BDF2), np.ones(2))
    np.testing.assert_allclose(sol.times, np.arange(0.0, 4.01, 0.5))


def test_event_location_tolerance_cannot_prevent_lte_step_reduction() -> None:
    class CountedDecay(_Decay):
        calls = 0

        def residual(self, u, u_dot, u_ddot, t):
            self.calls += 1
            if self.calls > 5000:
                msg = 'step rejection made no progress'
                raise RuntimeError(msg)
            return super().residual(u, u_dot, u_ddot, t)

    sol = integrate_transient(
        CountedDecay(),
        TimeWindow(t0=0.0, t1=0.01, dt=0.01),
        np.ones(2),
        time_tol=0.01,
        step_control=StepTolerance(rtol=1e-5, atol=1e-10),
    )
    assert sol.times[-1] == pytest.approx(0.01)
    np.testing.assert_allclose(sol.states[-1], np.exp(-_Decay.lam * 0.01), atol=1e-3, rtol=0)


class _UndeclaredStep:
    """``u' = s(t)`` with ``s`` jumping 0 -> 1e3 at t = 0.5 -- no breakpoint declares it."""

    def residual(self, u, u_dot, u_ddot, t):
        return u_dot - (1.0e3 if t > 0.5 else 0.0) * np.ones_like(u) - np.sin(t)

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.atleast_2d(weights.alpha1.real)


def test_an_unresolvable_step_raises_at_h_min_with_the_state_reached() -> None:
    with pytest.raises(RuntimeError, match='still .* over its local-error tolerance at h_min'):
        integrate_transient(
            _UndeclaredStep(),
            TimeWindow(t0=0.0, t1=1.0, dt=0.1, integrator=IntegratorKind.BDF2),
            np.zeros(1),
            step_control=StepTolerance(rtol=1e-8, atol=1e-10, h_min=1e-3),
        )
