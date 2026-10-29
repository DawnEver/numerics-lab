"""Portable locator clocks preserve the genuine unfinished parent cell."""

import json

import numpy as np
import pytest

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import EventSurface, StepNotConverged, _step, integrate_transient
from numerics_lab.time_integrators.checkpoint import TransientCheckpoint


class _StoredRamp:
    def __init__(self) -> None:
        self.slope = 1.0

    def residual(self, state, rate, acceleration, time):
        return np.asarray(rate) - self.slope

    def jacobian(self, state, rate, acceleration, time, weights):
        return np.array([[weights.alpha1.real]])

    def commit_history(self, state, time):
        pass

    def checkpoint_history(self):
        return {'slope': self.slope}

    def restore_checkpoint_history(self, data):
        self.slope = data['slope']

    def observe(self, state, time):
        return float(state[0]) - 0.5 if self.slope > 0 else -1.0

    def handle(self, index, time, state):
        self.slope = -1.0


def test_exact_clock_landing_after_state_event_is_committed_before_checkpoint():
    problem = _StoredRamp()
    observed = []
    captured = []

    def landing(time, state):
        assert not state.flags.writeable
        observed.append(time)
        if time == 0.75:
            problem.slope = -2.0
            return (1,)
        return ()

    whole = integrate_transient(
        problem,
        TimeWindow(0, 1, 0.3, IntegratorKind.BDF2),
        np.zeros(1),
        event_fns=(EventSurface(problem.observe, 1),),
        on_event=problem.handle,
        next_breakpoint=lambda time: 0.75 if time < 0.75 else None,
        on_landing=landing,
        checkpoint_key='event-and-clock',
        on_checkpoint=captured.append,
    )
    assert 0.75 in whole.times
    assert observed == whole.times[1:]
    assert len(observed) == len(set(observed))
    checkpoint = next(item for item in captured if item.time == 0.75)
    assert checkpoint.material_history == {'slope': -2.0}
    assert len(checkpoint.states) == 1
    assert 0.75 in whole.accepted_discontinuities
    assert whole.converged


@pytest.mark.parametrize('edge', [0.15, 0.55])
def test_fresh_json_resume_after_event_keeps_right_history_and_unfinished_cell(edge):
    original = _StoredRamp()
    captured = []
    whole = integrate_transient(
        original,
        TimeWindow(0, 1, 0.3, IntegratorKind.BDF2),
        np.array([0.0]),
        event_fns=(EventSurface(original.observe, 1),),
        on_event=original.handle,
        time_breakpoints=(edge,),
        checkpoint_key='declared-ramp',
        on_checkpoint=captured.append,
    )
    event_time = whole.events[0][1]
    checkpoint = next(item for item in captured if item.time == event_time)
    checkpoint = TransientCheckpoint.from_mapping(json.loads(json.dumps(checkpoint.to_mapping())))
    assert checkpoint.event_state.pending_target > event_time
    assert checkpoint.states == ((float(whole.states[whole.times.index(event_time)][0]),),)
    fresh = _StoredRamp()
    resumed = integrate_transient(
        fresh,
        TimeWindow(event_time, 1, 0.3, IntegratorKind.BDF2),
        np.array(checkpoint.states[-1]),
        event_fns=(EventSurface(fresh.observe, 1),),
        on_event=fresh.handle,
        time_breakpoints=(edge,) if edge > event_time else (),
        checkpoint_key='declared-ramp',
        restart=checkpoint,
    )
    start = whole.times.index(event_time)
    assert resumed.times == whole.times[start:]
    np.testing.assert_array_equal(resumed.states, whole.states[start:])
    assert resumed.events == []
    assert resumed.checkpoint.to_mapping() == whole.checkpoint.to_mapping()


def test_exhausted_step_cannot_be_persisted_as_accepted_locator_history():
    class Unreachable:
        def residual(self, state, rate, acceleration, time):
            return np.ones(1)

        def jacobian(self, state, rate, acceleration, time, weights):
            return np.eye(1)

    captured = []
    with pytest.raises(StepNotConverged):
        integrate_transient(
            Unreachable(),
            TimeWindow(0, 1, 0.1, IntegratorKind.BDF1),
            np.array([0.0]),
            checkpoint_key='unreachable',
            on_checkpoint=captured.append,
        )
    assert captured == []


@pytest.mark.parametrize('controls', [{'h_min': float('nan')}, {'h_min': 0}, {'adaptive': 1}, {'refine': -1}])
def test_live_locator_and_persisted_cursor_share_the_same_strict_step_policy(controls):
    with pytest.raises(ValueError, match='event checkpoint'):
        integrate_transient(
            _StoredRamp(),
            TimeWindow(0, 1, 0.1, IntegratorKind.BDF1),
            np.array([0.0]),
            **controls,
        )


@pytest.mark.parametrize('strategy', ['backtrack', 'directional'])
def test_actual_event_step_and_checkpoint_preserve_the_declared_history_line_search(monkeypatch, strategy):
    seen = []
    original = _step._solve_step

    def observe(*args, **kwargs):
        seen.append(kwargs['line_search'])
        return original(*args, **kwargs)

    monkeypatch.setattr(_step, '_solve_step', observe)
    solution = integrate_transient(
        _StoredRamp(),
        TimeWindow(0, 0.2, 0.1, IntegratorKind.BDF2),
        np.array([0.0]),
        line_search=strategy,
        checkpoint_key='declared-ramp',
    )
    assert seen and all(value == strategy for value in seen)
    assert solution.checkpoint.event_state.line_search == strategy
