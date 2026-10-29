"""A landed source edge records the actual residual time, not a nearest-time guess."""

import numpy as np
import pytest

from numerics_lab.formulation import TimeWindow
from numerics_lab.time_integrators import _step, integrate_transient
from numerics_lab.time_integrators.initial import consistent_initial_condition
from numerics_lab.time_integrators.march import StepNotConverged
from numerics_lab.time_integrators.multirate import SampledTask, integrate_sampled


class _Jump:
    def residual(self, u, v, a, t):
        return v - (1.0 if t < 0.5 else 2.0)

    def jacobian(self, u, v, a, t, weights):
        return np.array([[weights.alpha1]])


def test_source_edge_records_left_residual_time_and_exact_clock():
    result = integrate_transient(
        _Jump(),
        TimeWindow(t0=0.0, t1=1.0, dt=0.25, integrator='bdf1'),
        np.zeros(1),
        time_breakpoints=(0.5,),
    )
    assert result.times == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert result.residual_times == [0.25, np.nextafter(0.5, -np.inf), 0.75, 1.0]
    np.testing.assert_allclose(np.ravel(result.states), [0.0, 0.25, 0.5, 1.0, 1.5], atol=1e-14)


class _JumpDae:
    def residual(self, u, v, a, t):
        return np.array([v[0] - u[1], u[1] - (1.0 if t < 0.5 else 2.0)])

    def jacobian(self, u, v, a, t, weights):
        return np.array([[weights.alpha1, -weights.alpha0], [0.0, weights.alpha0]])


def test_source_transition_preserves_dynamic_state_and_records_preprojection_evidence():
    problem = _JumpDae()
    transitions = []

    def reinitialize(t, u):
        before = u.copy()
        u[:] = consistent_initial_condition(problem, u, t, tol=1e-14).u
        transitions.append((t, before, u.copy()))

    result = integrate_transient(
        problem,
        TimeWindow(t0=0.0, t1=1.0, dt=0.25, integrator='bdf1'),
        np.array([0.0, 1.0]),
        time_breakpoints=(0.5,),
        on_breakpoint=reinitialize,
    )
    np.testing.assert_allclose(result.states, [[0, 1], [0.25, 1], [0.5, 2], [1, 2], [1.5, 2]], atol=1e-14)
    assert len(transitions) == 1 and transitions[0][0] == 0.5
    np.testing.assert_array_equal(result.rate_state_overrides[1], [0.5, 1])
    np.testing.assert_array_equal(result.state_rates[1], [1, 0])
    assert result.residual_times[1] == np.nextafter(0.5, -np.inf)


def test_sampled_composition_preserves_each_segments_residual_time_evidence():
    result = integrate_sampled(
        _Jump(),
        TimeWindow(t0=0.0, t1=1.0, dt=0.125, integrator='bdf1'),
        np.zeros(1),
        (SampledTask(0.25, lambda _t, _u: None),),
    )
    assert result.residual_times == result.times[1:]
    assert len(result.residual_times) == len(result.state_rates)


def test_a_rejected_source_edge_cannot_advance_its_physical_handler(monkeypatch):
    original = _step._solve_step
    transitions = []

    def reject(*args, **kwargs):
        u, _ok, iterations, residual, floor = original(*args, **kwargs)
        return u, False, iterations, residual, floor

    monkeypatch.setattr(_step, '_solve_step', reject)
    with pytest.raises(StepNotConverged) as caught:
        integrate_transient(
            _Jump(),
            TimeWindow(t0=0.0, t1=1.0, dt=0.5, integrator='bdf1'),
            np.zeros(1),
            time_breakpoints=(0.5,),
            on_breakpoint=lambda t, u: transitions.append((t, u.copy())),
        )
    assert transitions == []
    assert caught.value.t_reached == 0.0
    np.testing.assert_array_equal(caught.value.u_reached, [0.0])


def test_a_nonfinite_source_transition_is_refused_instead_of_returned():
    def invalid(_t, u):
        u[:] = np.nan

    with pytest.raises(ValueError, match='breakpoint handler.*non-finite'):
        integrate_transient(
            _Jump(),
            TimeWindow(t0=0.0, t1=1.0, dt=0.5, integrator='bdf1'),
            np.zeros(1),
            time_breakpoints=(0.5,),
            on_breakpoint=invalid,
        )
