"""Periodic shooting closes the complete BDF numerical history."""

import numpy as np
import pytest

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import integrate_transient
from numerics_lab.time_integrators.periodic_march import shoot_transient_cycle


class DrivenDecay:
    def residual(self, u, u_dot, u_ddot, t):
        return u_dot + u + 0.1 * u**3 - np.cos(2 * np.pi * t)

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.diag(weights.alpha0.real * (1 + 0.3 * u**2) + weights.alpha1.real)


def test_nonlinear_shooting_closes_the_accepted_cycle():
    result = shoot_transient_cycle(
        DrivenDecay(),
        TimeWindow(0.0, 2.0, 0.02, IntegratorKind.BDF1),
        np.zeros(1),
        period=1.0,
        max_newton=20,
        step_tol=1e-10,
    )
    assert result.residual_norm < 1e-7
    assert result.cycle_evaluations > 1


def test_bdf2_closes_both_history_states_without_bootstrapping():
    study = TimeWindow(0.0, 2.0, 0.02)
    result = shoot_transient_cycle(DrivenDecay(), study, np.zeros(1), period=1.0, step_tol=1e-10)
    assert len(result.initial_history) == 2
    march = integrate_transient(
        DrivenDecay(), study, result.state, initial_history=result.initial_history, tol=1e-10, max_newton=20
    )
    np.testing.assert_allclose(march.states[49], result.initial_history[0], atol=1e-7)
    np.testing.assert_allclose(march.states[50], result.state, atol=1e-7)
    np.testing.assert_allclose(march.states[99], march.states[49], atol=1e-7)


def test_period_must_have_an_exact_step_boundary():
    with pytest.raises(ValueError, match='whole number'):
        shoot_transient_cycle(DrivenDecay(), TimeWindow(0.0, 2.0, 0.03, IntegratorKind.BDF1), np.zeros(1), period=1.0)


def test_unconverged_trial_cycle_cannot_produce_a_steady_seed():
    with pytest.raises(RuntimeError, match='trial cycle'):
        shoot_transient_cycle(
            DrivenDecay(),
            TimeWindow(0.0, 2.0, 0.02, IntegratorKind.BDF1),
            np.zeros(1),
            period=1.0,
            max_newton=1,
            step_tol=1e-16,
        )


@pytest.mark.parametrize(
    'history',
    [(np.zeros(1),), (np.zeros(2), np.zeros(1)), (np.zeros(1), np.ones(1)), (np.array([np.nan]), np.zeros(1))],
)
def test_invalid_bdf2_seed_history_is_never_silently_reset(history):
    with pytest.raises(ValueError, match='exactly two finite BDF2'):
        integrate_transient(DrivenDecay(), TimeWindow(0.0, 1.0, 0.02), np.zeros(1), initial_history=history)


def test_antiperiodic_bdf2_shooting_closes_both_signed_history_states():
    study = TimeWindow(0.0, 1.0, 0.02)

    # Linear affine odd-symmetric law is the native supported subset.
    class LinearDecay:
        def residual(self, u, v, a, t):
            return v + u - np.cos(2 * np.pi * t)

        def jacobian(self, u, v, a, t, weights):
            return np.diag(np.full(u.size, weights.alpha0.real + weights.alpha1.real))

    result = shoot_transient_cycle(LinearDecay(), study, np.zeros(1), period=0.5, sign=-1)
    march = integrate_transient(
        LinearDecay(), study, result.state, initial_history=result.initial_history, max_newton=20, tol=1e-10
    )
    np.testing.assert_allclose(march.states[24], -result.initial_history[0], atol=1e-7)
    np.testing.assert_allclose(march.states[25], -result.state, atol=1e-7)
    np.testing.assert_allclose(march.states[50], result.state, atol=1e-7)
