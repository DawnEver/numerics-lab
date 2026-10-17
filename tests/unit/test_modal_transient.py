"""Modal superposition reconstructs the physical second-order trajectory."""

import numpy as np
import pytest

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.modal_transient import integrate_modal_transient
from numerics_lab.time_integrators import integrate_transient


def test_full_basis_preserves_nonproportional_damping_and_dynamic_force():
    k, m, c = (
        np.array([[6.0, -2.0], [-2.0, 8.0]]),
        np.array([[2.0, 0.3], [0.3, 1.0]]),
        np.array([[0.8, 0.2], [0.2, 0.3]]),
    )

    def force(t):
        return np.array([np.cos(3 * t + 0.2), np.sin(2 * t)])

    class DirectProblem:
        def residual(self, u, u_dot, u_ddot, t):
            return k @ u + c @ u_dot + m @ u_ddot - force(t)

        def jacobian(self, u, u_dot, u_ddot, t, weights):
            return weights.alpha0.real * k + weights.alpha1.real * c + weights.alpha2.real * m

    study = TimeWindow(0.3, 0.8, 0.001, IntegratorKind.NEWMARK)
    u0, v0 = np.array([0.2, -0.1]), np.array([0.1, 0.3])
    a0 = np.linalg.solve(m, force(study.t0) - k @ u0 - c @ v0)
    direct = integrate_transient(DirectProblem(), study, u0, v0=v0, a0=a0)
    modal = integrate_modal_transient(k, m, c, force, study, n_modes=2, u0=u0, v0=v0, a0=a0)
    np.testing.assert_allclose(modal.solution.states, direct.states, rtol=1e-8, atol=1e-10)


def test_callable_force_keeps_absolute_time_and_phase():
    study = TimeWindow(0.3, 0.8, 0.001, IntegratorKind.NEWMARK)
    amplitude, omega, phase = 0.5, 2.0, 0.4
    result = integrate_modal_transient(
        [[18.0]],
        [[2.0]],
        [[0.0]],
        lambda t: np.array([5 * np.cos(omega * t + phase)]),
        study,
        n_modes=1,
        u0=[amplitude * np.cos(omega * study.t0 + phase)],
        v0=[-omega * amplitude * np.sin(omega * study.t0 + phase)],
    )
    expected = amplitude * np.cos(omega * np.array(result.solution.times) + phase)
    np.testing.assert_allclose(np.array(result.solution.states)[:, 0], expected, atol=2e-6)


def test_dynamic_truncation_error_includes_load_after_initial_time():
    result = integrate_modal_transient(
        np.diag([4.0, 9.0]),
        np.eye(2),
        np.zeros((2, 2)),
        lambda t: np.array([0.0, t]),
        TimeWindow(0.0, 0.1, 0.01, IntegratorKind.NEWMARK),
        n_modes=1,
    )
    assert result.load_projection_error == pytest.approx(1.0)


def test_nonfinite_callable_load_is_refused_at_evaluation_time():
    with pytest.raises(ValueError, match='finite vector'):
        integrate_modal_transient(
            [[4.0]],
            [[1.0]],
            [[0.0]],
            lambda t: np.array([0.0 if t == 0 else np.nan]),
            TimeWindow(0.0, 0.1, 0.01, IntegratorKind.NEWMARK),
            n_modes=1,
        )


def test_all_modes_reproduce_a_decoupled_oscillator():
    study = TimeWindow(0.0, 0.5, 0.0005, IntegratorKind.NEWMARK)
    result = integrate_modal_transient(
        np.diag([4.0, 18.0]), np.diag([1.0, 2.0]), np.zeros((2, 2)), np.zeros(2), study, n_modes=2, u0=np.ones(2)
    )
    np.testing.assert_allclose(result.solution.states[-1], [np.cos(1.0), np.cos(1.5)], atol=1e-6)
    assert result.initial_projection_error < 1e-12


def test_truncated_modes_report_projection_loss():
    result = integrate_modal_transient(
        np.diag([4.0, 18.0, 48.0]),
        np.diag([1.0, 2.0, 3.0]),
        np.zeros((3, 3)),
        np.zeros(3),
        TimeWindow(0.0, 0.1, 0.001, IntegratorKind.NEWMARK),
        n_modes=1,
        u0=np.array([1.0, 1.0, 0.0]),
    )
    assert result.initial_projection_error > 0.5
    assert abs(result.solution.states[-1][1]) < 1e-12


def test_modal_damping_ratio_is_applied_in_mass_normalized_coordinates():
    result = integrate_modal_transient(
        np.array([[4.0]]),
        np.array([[1.0]]),
        np.zeros((1, 1)),
        np.zeros(1),
        TimeWindow(0.0, 0.5, 0.001, IntegratorKind.NEWMARK),
        n_modes=1,
        u0=np.ones(1),
        damping_ratios=(1.0,),
    )
    assert result.solution.states[-1][0] == pytest.approx(2.0 / np.e, abs=1e-6)


def test_damping_definition_is_not_silently_double_counted():
    with pytest.raises(ValueError, match='cannot combine'):
        integrate_modal_transient(
            np.eye(1),
            np.eye(1),
            np.eye(1),
            np.zeros(1),
            TimeWindow(0.0, 1.0, 0.01, IntegratorKind.NEWMARK),
            n_modes=1,
            damping_ratios=(0.1,),
        )


def test_modal_mass_must_be_positive_on_discarded_modes_too():
    with pytest.raises(ValueError, match='positive definite mass'):
        integrate_modal_transient(
            np.eye(3),
            np.diag([1.0, 2.0, -1.0]),
            np.zeros((3, 3)),
            np.zeros(3),
            TimeWindow(0.0, 1.0, 0.01, IntegratorKind.NEWMARK),
            n_modes=1,
        )


def test_velocity_truncation_is_measured_when_displacement_is_zero():
    result = integrate_modal_transient(
        np.diag([4.0, 9.0, 16.0]),
        np.eye(3),
        np.zeros((3, 3)),
        np.zeros(3),
        TimeWindow(0.0, 0.1, 0.001, IntegratorKind.NEWMARK),
        n_modes=1,
        v0=np.array([0.0, 1.0, 0.0]),
    )
    assert result.initial_projection_error > 0.99
