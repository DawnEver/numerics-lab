"""Nonlinear coarse EEC uses the sum of actual accepted-state tangents."""

import numpy as np
import pytest

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import integrate_transient


class Cubic:
    mass, stiffness, cubic = 2.0, 0.4, 0.2

    def residual(self, u, v, a, t):
        return self.mass * v + self.stiffness * u + self.cubic * u**3 - np.cos(2 * np.pi * t)

    def jacobian(self, u, v, a, t, weights):
        return np.eye(1) * (
            weights.alpha1.real * self.mass + weights.alpha0.real * (self.stiffness + 3 * self.cubic * u[0] ** 2)
        )


def test_nonlinear_eec_matches_the_independent_integrated_tangent_and_next_physical_step():
    dt, period = 0.05, 1.0
    study = TimeWindow(0.0, period + dt, dt, IntegratorKind.BDF1)
    plain = integrate_transient(Cubic(), study, np.array([0.7]), max_newton=30)
    result = integrate_transient(
        Cubic(), study, np.array([0.7]), max_newton=30, eec_period=period, eec_nonlinear_first_order=True
    )
    values = np.asarray(plain.states)[1:21, 0]
    integrated_tangent = dt * np.sum(Cubic.stiffness + 3 * Cubic.cubic * values**2)
    p = Cubic.mass * (plain.states[20][0] - 0.7) / integrated_tangent
    seed = plain.states[20][0] + p
    roots = np.roots([
        Cubic.cubic,
        0.0,
        Cubic.stiffness + Cubic.mass / dt,
        -Cubic.mass * seed / dt - np.cos(2 * np.pi * (period + dt)),
    ])
    expected = next(root.real for root in roots if abs(root.imag) < 1e-12)
    np.testing.assert_allclose(result.states[:21], plain.states[:21], atol=1e-13)
    assert result.states[-1][0] == pytest.approx(expected, abs=1e-12)
    assert result.corrected_times == [period]
    assert result.converged


@pytest.mark.parametrize(
    'controls',
    [
        {'adaptive': True},
        {'eec_sign': -1},
    ],
)
def test_nonlinear_eec_explicitly_refuses_unsupported_algorithm_state(controls):
    with pytest.raises(ValueError, match='nonlinear EEC'):
        integrate_transient(
            Cubic(),
            TimeWindow(0.0, 2.0, 0.05, IntegratorKind.BDF1),
            np.ones(1),
            eec_period=1.0,
            eec_nonlinear_first_order=True,
            **controls,
        )
