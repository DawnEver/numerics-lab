"""Uniform BDF2 EEC corrects its boundary moment, preserving both history states."""

from types import SimpleNamespace

import numpy as np
import pytest
from tests.unit.test_antiperiodic_eec import Affine

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import integrate_transient


def _phasor(dt):
    z = np.exp(-2j * np.pi * dt)
    return 1 / (Affine.stiffness + Affine.mass * (3 - 4 * z + z * z) / (2 * dt))


@pytest.mark.parametrize('sign', [1, -1])
def test_true_bdf2_periodic_orbit_remains_a_fixed_point_after_eec(sign):
    dt, period = 0.05, 1.0 if sign == 1 else 0.5
    amplitude = _phasor(dt)
    u0 = np.array([amplitude.real])
    result = integrate_transient(
        Affine(),
        TimeWindow(0.0, 2 * period + dt, dt),
        u0,
        initial_history=(np.array([(amplitude * np.exp(-2j * np.pi * dt)).real]), u0),
        eec_period=period,
        eec_sign=sign,
        eec_affine_first_order=True,
    )
    exact = np.real(amplitude * np.exp(2j * np.pi * np.asarray(result.times)))
    np.testing.assert_allclose(np.asarray(result.states)[:, 0], exact, atol=1e-13)


@pytest.mark.parametrize('sign', [1, -1])
def test_augmented_error_matches_the_independent_boundary_moment_map(sign):
    dt, period = 0.05, 1.0 if sign == 1 else 0.5
    n = round(period / dt)
    amplitude = _phasor(dt)
    initial_error = np.array([0.7, 1.0])  # previous and current error, independently perturbed
    history = (
        np.array([(amplitude * np.exp(-2j * np.pi * dt)).real + initial_error[0]]),
        np.array([amplitude.real + initial_error[1]]),
    )
    result = integrate_transient(
        Affine(),
        TimeWindow(0.0, 2 * period, dt),
        history[-1],
        initial_history=history,
        eec_period=period,
        eec_sign=sign,
        eec_affine_first_order=True,
    )
    r = dt * Affine.stiffness / Affine.mass
    step = np.array([[0.0, 1.0], [-1 / (3 + 2 * r), 4 / (3 + 2 * r)]])
    cycle = np.linalg.matrix_power(step, n)
    boundary = np.array([-0.5, 1.5])
    correction = cycle + np.outer(np.ones(2), boundary @ (sign * cycle - np.eye(2))) / (n * r + 1 - sign)
    expected = cycle @ correction @ initial_error
    actual = np.array([result.states[-2][0], result.states[-1][0]])
    reference = np.real(amplitude * np.exp(2j * np.pi * np.array(result.times[-2:])))
    np.testing.assert_allclose(actual - reference, expected, atol=1e-13)
    # The recorded first endpoint is still the physical, uncorrected solve.
    assert result.states[n][0] - np.real(amplitude * np.exp(2j * np.pi * period)) == pytest.approx(
        (cycle @ initial_error)[1]
    )


@pytest.mark.parametrize('sign', [1, -1])
def test_three_physical_cycles_give_tenfold_acceleration_against_the_exact_augmented_map(sign):
    dt, period = 0.05, 1.0 if sign == 1 else 0.5
    n = round(period / dt)
    amplitude = _phasor(dt)
    error = np.array([0.7, 1.0])
    history = (
        np.array([(amplitude * np.exp(-2j * np.pi * dt)).real + error[0]]),
        np.array([amplitude.real + error[1]]),
    )
    result = integrate_transient(
        Affine(),
        TimeWindow(0.0, 3 * period, dt),
        history[-1],
        initial_history=history,
        eec_period=period,
        eec_sign=sign,
        eec_affine_first_order=True,
    )
    r = dt * Affine.stiffness / Affine.mass
    cycle = np.linalg.matrix_power(np.array([[0.0, 1.0], [-1 / (3 + 2 * r), 4 / (3 + 2 * r)]]), n)
    correction = cycle + np.outer(np.ones(2), np.array([-0.5, 1.5]) @ (sign * cycle - np.eye(2))) / (n * r + 1 - sign)
    expected = cycle @ correction @ correction @ error
    reference = np.real(amplitude * np.exp(2j * np.pi * np.array(result.times[-2:])))
    np.testing.assert_allclose(np.asarray(result.states[-2:])[:, 0] - reference, expected, atol=1e-13)
    assert np.linalg.norm(expected) < np.linalg.norm(np.linalg.matrix_power(cycle, 3) @ error) * 0.1


@pytest.mark.parametrize('sign', [1, -1])
def test_bootstrap_cycle_uses_the_exact_mixed_be_bdf2_boundary_moment(sign):
    dt, period = 0.05, 1.0 if sign == 1 else 0.5
    n = round(period / dt)
    study = TimeWindow(0.0, period + dt, dt)
    plain = integrate_transient(Affine(), study, np.ones(1))
    corrected = integrate_transient(
        Affine(), study, np.ones(1), eec_period=period, eec_sign=sign, eec_affine_first_order=True
    )
    start = 0.5 * (plain.states[0] + plain.states[1])
    end = 1.5 * plain.states[n] - 0.5 * plain.states[n - 1]
    p = Affine.mass * (sign * end - start) / (period * Affine.stiffness + (1 - sign) * Affine.mass)
    expected = (
        Affine.mass * (4 * (plain.states[n] + p) - (plain.states[n - 1] + p)) / (2 * dt)
        + np.cos(2 * np.pi * (period + dt))
    ) / (Affine.stiffness + 3 * Affine.mass / (2 * dt))
    np.testing.assert_allclose(corrected.states[-1], expected, atol=1e-13)


def test_bdf2_eec_refuses_nonuniform_step_tables():
    with pytest.raises(ValueError, match='uniform BDF2'):
        integrate_transient(
            Affine(),
            TimeWindow(0.0, 1.0, 0.05),
            np.ones(1),
            step_times=(0.0, 0.1, 0.3, 0.5, 0.75, 1.0),
            eec_period=0.5,
            eec_affine_first_order=True,
        )


def test_bdf2_eec_refuses_an_unaligned_final_endpoint():
    with pytest.raises(ValueError, match='uniform BDF2'):
        integrate_transient(
            Affine(), TimeWindow(0.0, 0.53, 0.05), np.ones(1), eec_period=0.5, eec_affine_first_order=True
        )


@pytest.mark.parametrize('period', [True, 1e-12])
def test_eec_admission_requires_a_real_interval_with_positive_integer_ticks(period, monkeypatch):
    def forbid_allocation(*_args, **_kwargs):
        msg = 'invalid EEC period reached correction-grid allocation'
        raise AssertionError(msg)

    monkeypatch.setattr(np, 'arange', forbid_allocation)
    with pytest.raises(ValueError, match='positive|integer'):
        integrate_transient(
            Affine(), TimeWindow(0.0, 2.0, 1.0), np.ones(1), eec_period=period, eec_affine_first_order=True
        )


def test_eec_admission_rejects_a_nonadvancing_uniform_clock_before_solving():
    study = SimpleNamespace(t0=1e16, t1=1e16 + 4.0, dt=0.1, integrator=IntegratorKind.BDF2)
    with pytest.raises(ValueError, match='advanc|representable'):
        integrate_transient(Affine(), study, np.ones(1), eec_period=1.0, eec_affine_first_order=True)
