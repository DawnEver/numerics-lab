"""Derivative readback records accepted StepForms, including restart history."""

import numpy as np
import pytest
from tests.unit._helpers import _RampWithToggle
from tests.unit.test_time_integrators import DecayProblem

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import integrate_transient


@pytest.mark.parametrize('adaptive', [False, True])
def test_bdf2_rates_match_the_actual_equation_on_nonuniform_accepted_steps(adaptive):
    solution = integrate_transient(
        DecayProblem(2.0),
        TimeWindow(t0=0.0, t1=0.37, dt=0.1, integrator=IntegratorKind.BDF2),
        np.array([1.0]),
        adaptive=adaptive,
    )
    assert solution.converged
    assert len(solution.state_rates) == len(solution.states) - 1
    assert not np.allclose(np.diff(solution.times), 0.1)
    np.testing.assert_allclose(solution.state_rates, -2 * np.asarray(solution.states[1:]), atol=1e-12)
    assert sum(rate.nbytes for rate in solution.state_rates) == sum(state.nbytes for state in solution.states[1:])


def test_event_probes_are_not_published_and_the_rate_keeps_its_restart():
    problem = _RampWithToggle(level=0.37)
    solution = integrate_transient(
        problem,
        TimeWindow(t0=0.0, t1=1.0, dt=0.2, integrator=IntegratorKind.BDF2),
        np.array([0.0]),
        event_fns=[problem.event_fn],
        on_event=lambda _k, _t, _u: problem.toggle(),
    )
    assert solution.converged and problem.toggles == 1
    assert len(solution.state_rates) == len(solution.states) - 1
    event_time = solution.events[0][1]
    expected = np.where(np.asarray(solution.times[1:]) <= event_time, 1.0, -1.0)
    np.testing.assert_allclose(np.asarray(solution.state_rates)[:, 0], expected, atol=1e-12)
