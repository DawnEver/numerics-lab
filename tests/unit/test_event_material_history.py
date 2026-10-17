"""Only canonical eventful landings advance constitutive memory."""

import numpy as np
import pytest

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import EventSurface, integrate_transient


class _HistoryRamp:
    def __init__(self) -> None:
        self.accepted = []

    def residual(self, state, rate, acceleration, time):
        return np.asarray(rate) - 1

    def jacobian(self, state, rate, acceleration, time, weights):
        return np.array([[weights.alpha1.real]])

    def commit_history(self, state, time):
        self.accepted.append((time, float(state[0])))


def test_event_probes_never_commit_and_handler_reads_the_actual_landing_history():
    problem = _HistoryRamp()
    handled = []

    def handle(_index, time, state):
        handled.append((time, float(state[0]), problem.accepted[-1]))

    solution = integrate_transient(
        problem,
        TimeWindow(0, 1, 0.3, IntegratorKind.BDF2),
        np.array([0.0]),
        event_fns=(EventSurface(lambda state, _time: float(state[0]) - 0.5, 1),),
        on_event=handle,
    )
    assert solution.converged
    assert problem.accepted == [
        (time, float(state[0])) for time, state in zip(solution.times[1:], solution.states[1:], strict=True)
    ]
    assert len(handled) == 1
    time, value, committed = handled[0]
    assert time == pytest.approx(0.5, abs=0.3e-6)
    assert committed == (time, value)
