"""Half-period affine EEC includes the anti-periodic boundary mass term."""

import numpy as np
import pytest

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import integrate_transient
from numerics_lab.time_integrators.periodic import explicit_error_correction


class Affine:
    mass, stiffness = 2.0, 0.4

    def residual(self, u, v, a, t):
        return self.mass * v + self.stiffness * u - np.cos(2 * np.pi * t)

    def jacobian(self, u, v, a, t, weights):
        return np.eye(1) * (self.stiffness * weights.alpha0.real + self.mass * weights.alpha1.real)


def test_half_period_correction_has_the_extra_boundary_mass():
    current, previous, half = np.array([0.8]), np.array([1.2]), 0.5
    corrected = explicit_error_correction(Affine(), current, previous, half, half, solve=np.linalg.solve, sign=-1)
    expected = current - Affine.mass * (current + previous) / (half * Affine.stiffness + 2 * Affine.mass)
    np.testing.assert_allclose(corrected, expected, atol=1e-15)


def test_analytic_discrete_antiperiodic_state_is_a_correction_fixed_point():
    dt, half = 0.05, 0.5
    phasor = 1 / (Affine.stiffness + Affine.mass * (1 - np.exp(-2j * np.pi * dt)) / dt)
    previous = np.array([phasor.real])
    corrected = explicit_error_correction(Affine(), -previous, previous, half, half, solve=np.linalg.solve, sign=-1)
    np.testing.assert_allclose(corrected, -previous, atol=1e-15)


def test_half_period_jump_is_not_reported_as_a_physical_endpoint():
    study = TimeWindow(0.0, 0.55, 0.05, IntegratorKind.BDF1)
    plain = integrate_transient(Affine(), study, np.ones(1), max_newton=20)
    corrected = integrate_transient(
        Affine(), study, np.ones(1), max_newton=20, eec_period=0.5, eec_sign=-1, eec_affine_first_order=True
    )
    np.testing.assert_allclose(corrected.states[10], plain.states[10], atol=1e-14)
    seed = plain.states[10] - Affine.mass * (plain.states[10] + 1) / (0.5 * Affine.stiffness + 2 * Affine.mass)
    expected_next = (Affine.mass / 0.05 * seed + np.cos(2 * np.pi * 0.55)) / (Affine.mass / 0.05 + Affine.stiffness)
    np.testing.assert_allclose(corrected.states[11], expected_next, atol=1e-14)
    assert corrected.corrected_times == [0.5]


def test_half_period_bdf2_eec_accepts_a_uniform_history_projection():
    result = integrate_transient(
        Affine(), TimeWindow(0.0, 1.0, 0.05), np.ones(1), eec_period=0.5, eec_sign=-1, eec_affine_first_order=True
    )
    assert result.converged


def test_forced_antiperiodic_march_accelerates_the_actual_discrete_solution():
    dt = 0.05
    study = TimeWindow(0.0, 1.5, dt, IntegratorKind.BDF1)
    phasor = 1 / (Affine.stiffness + Affine.mass * (1 - np.exp(-2j * np.pi * dt)) / dt)
    seed = np.array([phasor.real + 1.0])
    plain = integrate_transient(Affine(), study, seed, max_newton=20)
    corrected = integrate_transient(
        Affine(), study, seed, max_newton=20, eec_period=0.5, eec_sign=-1, eec_affine_first_order=True
    )
    exact = np.real(phasor * np.exp(2j * np.pi * study.t1))
    assert abs(corrected.states[-1][0] - exact) < abs(plain.states[-1][0] - exact) * 1e-4
    # Endpoint correction remains at its advanced clock phase. Its error multiplier
    # is ((1+a)*q**n-1)/(2+a), not a same-phase restart with a negated endpoint.
    a = 0.5 * Affine.stiffness / Affine.mass
    q = Affine.mass / (Affine.mass + dt * Affine.stiffness)
    reduction = ((1 + a) * q**10 - 1) / (2 + a)
    np.testing.assert_allclose(corrected.states[-1][0] - exact, q**10 * reduction**2, atol=1e-13)


@pytest.mark.parametrize('sign', [True, 0, 1.5, -1.0])
def test_correction_helper_rejects_noninteger_or_invalid_signs(sign):
    with pytest.raises(ValueError, match='explicit integer'):
        explicit_error_correction(Affine(), np.ones(1), np.ones(1), 0.5, 0.5, solve=np.linalg.solve, sign=sign)


@pytest.mark.parametrize('period', [0.0, -1.0, np.nan, np.inf])
def test_correction_helper_rejects_invalid_intervals(period):
    with pytest.raises(ValueError, match='positive finite scalar'):
        explicit_error_correction(Affine(), np.ones(1), np.ones(1), period, 0.5, solve=np.linalg.solve, sign=-1)
