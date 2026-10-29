"""Requested continuous observations are true solved endpoints, never fake source edges."""

import numpy as np
import pytest
from scipy.linalg import expm
from tests.unit.test_trapezoid_and_tr_bdf2_are_second_order_on_rc_rl_rlc import _Linear

from numerics_lab.formulation import TimeWindow
from numerics_lab.time_integrators import integrate_transient


def test_offset_nominal_observations_are_exact_hits_without_ulp_sized_cells():
    grid = np.linspace(0.1, 0.12, 401)
    solution = integrate_transient(
        _Linear(1, 1, lambda _t: np.zeros(1)),
        TimeWindow(t0=grid[0], t1=grid[-1], dt=5e-5, integrator='trapezoid'),
        np.ones(1),
        v0=-np.ones(1),
        observation_times=grid[1:],
        tol=1e-13,
    )
    np.testing.assert_array_equal(solution.times, grid)
    assert np.min(np.diff(solution.times)) > 4e-5


def test_distinct_requested_final_neighbors_are_not_discarded_by_end_tolerance():
    grid = np.array([0.0, np.nextafter(0.2, 0.0), 0.2])
    solution = integrate_transient(
        _Linear(1, 0, lambda _t: np.ones(1)),
        TimeWindow(t0=0.0, t1=0.2, dt=0.2, integrator='trapezoid'),
        np.zeros(1),
        v0=np.ones(1),
        observation_times=grid[1:],
    )
    np.testing.assert_array_equal(solution.times, grid)


def test_nonuniform_observations_preserve_trapezoid_history_and_match_expm():
    a = np.array([[-2.0, 1.0], [-1.0, -3.0]])
    problem = _Linear(np.eye(2), -a, lambda _t: np.zeros(2))
    u0 = np.array([1.0, -0.4])
    grid = np.r_[0.0, np.cumsum(np.tile([0.001, 0.002, 0.003], 20))]
    solution = integrate_transient(
        problem,
        TimeWindow(t0=0, t1=grid[-1], dt=0.1, integrator='trapezoid'),
        u0,
        v0=a @ u0,
        observation_times=grid[1:],
        tol=1e-13,
    )
    np.testing.assert_array_equal(solution.times, grid)
    expected = u0.copy()
    for k, h in enumerate(np.diff(grid), 1):
        expected = np.linalg.solve(np.eye(2) - h / 2 * a, (np.eye(2) + h / 2 * a) @ expected)
        np.testing.assert_allclose(solution.states[k], expected, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(solution.states[k], expm(a * grid[k]) @ u0, atol=1e-6, rtol=0)


def test_observation_on_a_real_breakpoint_still_restarts_discontinuous_history():
    problem = _Linear(1, 0, lambda t: np.array([1 if t < 0.1 else 3]))
    solution = integrate_transient(
        problem,
        TimeWindow(t0=0, t1=0.2, dt=0.2, integrator='trapezoid'),
        np.zeros(1),
        v0=np.ones(1),
        observation_times=(0.1, 0.2),
        time_breakpoints=(0.1,),
    )
    np.testing.assert_array_equal(solution.times, [0, 0.1, 0.2])
    np.testing.assert_allclose(np.asarray(solution.states).ravel(), [0, 0.1, 0.4], atol=1e-12)


def test_continuous_observations_do_not_suppress_a_state_event_restart():
    drive = [1]
    fired = []
    problem = _Linear(1, 0, lambda _t: np.array(drive))

    def event(_index, instant, _state):
        fired.append(instant)
        drive[0] = 3

    solution = integrate_transient(
        problem,
        TimeWindow(t0=0, t1=0.2, dt=0.2, integrator='trapezoid'),
        np.zeros(1),
        v0=np.ones(1),
        observation_times=(0.1, 0.2),
        event_fns=(lambda state, _t: state[0] - 0.1,),
        on_event=event,
    )
    assert len(fired) == 1
    assert 0.1 in solution.times and 0.2 in solution.times
    assert solution.states[-1][0] == pytest.approx(0.4, abs=1e-6)


@pytest.mark.parametrize('grid', [(0,), (0.3,), (0.1, 0.1), (0.2, 0.1), (np.nan,), (np.inf,)])
def test_invalid_observation_times_are_named_refusals(grid):
    with pytest.raises(ValueError, match='observation_times'):
        integrate_transient(
            _Linear(1, 1, lambda _t: np.zeros(1)),
            TimeWindow(t0=0, t1=0.2, dt=0.1),
            np.ones(1),
            observation_times=grid,
        )
