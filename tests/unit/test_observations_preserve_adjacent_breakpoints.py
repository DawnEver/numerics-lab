"""Distinct observation instants must never displace a real source transition."""

import numpy as np
import pytest

from numerics_lab.formulation import TimeWindow
from numerics_lab.time_integrators import integrate_transient


class _Ramp:
    def residual(self, u, v, a, t):
        return v - 1.0

    def jacobian(self, u, v, a, t, weights):
        return np.array([[weights.alpha1]])


@pytest.mark.parametrize('direction', [-np.inf, np.inf])
def test_adjacent_observation_preserves_exact_breakpoint_callback_and_provenance(direction):
    edge = 0.25
    observation = np.nextafter(edge, direction)
    callbacks = []
    result = integrate_transient(
        _Ramp(),
        TimeWindow(t0=0.0, t1=0.5, dt=0.5, integrator='bdf1'),
        np.zeros(1),
        time_breakpoints=(edge,),
        observation_times=(observation, 0.5),
        on_breakpoint=lambda t, _u: callbacks.append(t),
    )
    np.testing.assert_array_equal(result.times, [0.0, *sorted((edge, observation)), 0.5])
    assert callbacks == [edge]
    assert result.residual_times[result.times.index(edge) - 1] == np.nextafter(edge, -np.inf)
    assert result.residual_times[result.times.index(observation) - 1] == observation
