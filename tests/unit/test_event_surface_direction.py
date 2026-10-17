"""A located crossing has an explicit orientation, independent of physics."""

import numpy as np
import pytest
from tests.unit._helpers import _RampWithToggle

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import EventSurface, integrate_transient


@pytest.mark.parametrize(('direction', 'slope'), [(1, -1), (-1, 1)])
def test_initial_zero_leaving_into_stay_region_does_not_fire(direction, slope):
    problem = _RampWithToggle(level=0)
    problem.slope = slope
    solution = integrate_transient(
        problem,
        TimeWindow(0, 1, 0.3, IntegratorKind.BDF2),
        np.array([0.0]),
        event_fns=(EventSurface(lambda state, _time: float(state[0]), direction),),
    )
    assert solution.converged
    assert solution.events == []
    assert solution.states[-1][0] == pytest.approx(slope)


def test_declared_rising_surface_still_locates_the_actual_later_crossing():
    problem = _RampWithToggle(level=0.5)
    solution = integrate_transient(
        problem,
        TimeWindow(0, 1, 0.3, IntegratorKind.BDF2),
        np.array([0.0]),
        event_fns=(EventSurface(problem.event_fn, 1),),
        on_event=lambda _index, _time, _state: problem.toggle(),
    )
    assert len(solution.events) == 1
    assert solution.events[0][1] == pytest.approx(0.5, abs=0.3e-6)


@pytest.mark.parametrize('direction', [True, 2, 0.5])
def test_event_orientation_is_explicit_not_coerced(direction):
    with pytest.raises(ValueError, match=r'explicit -1/0/\+1'):
        EventSurface(lambda _state, _time: 0, direction)
