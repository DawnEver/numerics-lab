"""Several event observers query one physical trial, not several field solves."""

from collections import Counter
from importlib import import_module

import numpy as np
import pytest
from tests.unit._helpers import _RampWithToggle

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import _step as _step_mod


def test_rollback_retracts_only_discarded_residual_side_snapshots():
    events = import_module('numerics_lab.time_integrators.march')
    state = events.IntegratorState(us=[np.array([2.0])], t=1.0, v=None, a=None)
    solved = events.TransientSolution(
        times=[0.0, 1.0, 2.0],
        states=[np.array([0.0]), np.array([2.0]), np.array([4.0])],
        state_rates=[np.array([1.0]), np.array([2.0])],
        rate_state_overrides={0: np.array([1.0]), 1: np.array([3.0])},
    )
    checkpoint = events._Checkpoint(state=state, n_points=2)
    events._rollback(checkpoint, solved)
    assert solved.times == [0.0, 1.0]
    assert solved.rate_state_overrides is not None
    assert tuple(solved.rate_state_overrides) == (0,)
    np.testing.assert_array_equal(solved.rate_state_overrides[0], [1.0])


@pytest.mark.parametrize('kind', [IntegratorKind.BDF1, IntegratorKind.BDF2])
def test_a_mutating_handler_retains_the_original_residual_side_once(kind):
    events = import_module('numerics_lab.time_integrators.march')
    problem = _RampWithToggle(0.3)
    before = []

    def handler(index, _time, state):
        if index == 0:
            before.append(state.copy())
            problem.toggle()
        state[0] += index + 1

    solved = events.integrate_transient(
        problem,
        TimeWindow(t0=0.0, t1=0.5, dt=0.4, integrator=kind),
        np.zeros(1),
        max_newton=2,
        event_fns=(problem.event_fn, problem.event_fn),
        on_event=handler,
    )
    assert solved.rate_state_overrides is not None
    assert len(solved.rate_state_overrides) == 1
    index, residual_state = next(iter(solved.rate_state_overrides.items()))
    np.testing.assert_array_equal(residual_state, before[0])
    np.testing.assert_array_equal(solved.states[index + 1], before[0] + 3.0)
    assert not residual_state.flags.writeable
    assert solved.state_rates is not None
    np.testing.assert_allclose(solved.state_rates[index], [1.0], atol=1e-12, rtol=0.0)


@pytest.mark.parametrize('kind', [IntegratorKind.BDF1, IntegratorKind.BDF2])
@pytest.mark.parametrize('cache_bytes', [0, 64 * 1024**2])
def test_coincident_observers_reuse_exact_trial_but_preserve_states_rates_and_callbacks(monkeypatch, kind, cache_bytes):
    events = import_module('numerics_lab.time_integrators.march')
    study = TimeWindow(t0=0.0, t1=1.0, dt=0.4, integrator=kind)

    def march(count):
        problem = _RampWithToggle(0.3)
        callbacks = []

        def handler(index, time, _state):
            callbacks.append((index, time))
            if index == 0:
                problem.toggle()

        solved = events.integrate_transient(
            problem,
            study,
            np.zeros(1),
            max_newton=2,
            event_fns=tuple(problem.event_fn for _ in range(count)),
            on_event=handler,
        )
        return solved, callbacks

    reference, _callbacks = march(1)
    original_step, original_locator = _step_mod._solve_step, events._locate_bracket
    inside_probe = False
    requests, solves = [], []

    def locate(lo, g_lo, hi, g_hi, probe, **kwargs):
        def observed(time):
            nonlocal inside_probe
            inside_probe = True
            requests.append(time)
            try:
                return probe(time)
            finally:
                inside_probe = False

        return original_locator(lo, g_lo, hi, g_hi, observed, **kwargs)

    def solve_step(problem, form, guess, **kwargs):
        if inside_probe:
            solves.append(form.t_eval)
        return original_step(problem, form, guess, **kwargs)

    monkeypatch.setattr(events, '_locate_bracket', locate)
    monkeypatch.setattr(_step_mod, '_solve_step', solve_step)
    monkeypatch.setattr(events, '_EVENT_TRIAL_CACHE_BYTES', cache_bytes)
    actual, callbacks = march(6)
    assert actual.converged
    assert len(callbacks) == 6 and {index for index, _time in callbacks} == set(range(6))
    np.testing.assert_array_equal(actual.times, reference.times)
    np.testing.assert_array_equal(actual.states, reference.states)
    np.testing.assert_array_equal(actual.state_rates, reference.state_rates)
    assert max(Counter(requests).values()) == 6
    assert set(solves) == set(requests)
    expected_solves = 1 if cache_bytes else 6
    assert max(Counter(solves).values()) == expected_solves


def test_a_failed_trial_is_retried_not_exact_memoized(monkeypatch):
    events = import_module('numerics_lab.time_integrators.march')
    problem = _RampWithToggle(0.3)
    study = TimeWindow(t0=0.0, t1=0.4, dt=0.4, integrator=IntegratorKind.BDF1)
    original_step, original_locator = _step_mod._solve_step, events._locate_bracket
    inside_probe = False
    failed_time = None
    attempts = Counter()

    def locate(lo, g_lo, hi, g_hi, probe, **kwargs):
        def observed(time):
            nonlocal inside_probe
            inside_probe = True
            try:
                return probe(time)
            finally:
                inside_probe = False

        return original_locator(lo, g_lo, hi, g_hi, observed, **kwargs)

    def solve_step(*args, **kwargs):
        nonlocal failed_time
        result = original_step(*args, **kwargs)
        if inside_probe:
            time = args[1].t_eval
            attempts[time] += 1
            if failed_time is None:
                failed_time = time
                state, _ok, iterations, residual, floor = result
                return state, False, iterations, residual, floor
        return result

    monkeypatch.setattr(events, '_locate_bracket', locate)
    monkeypatch.setattr(_step_mod, '_solve_step', solve_step)
    solved = events.integrate_transient(
        problem,
        study,
        np.zeros(1),
        max_newton=2,
        event_fns=(problem.event_fn, problem.event_fn),
        on_event=lambda *_args: problem.toggle(),
    )
    assert solved.converged
    assert attempts[failed_time] == 2
    assert failed_time in solved.nonconverged_probe_times


@pytest.mark.parametrize('kind', [IntegratorKind.BDF1, IntegratorKind.BDF2])
def test_same_time_after_a_physical_handler_has_a_new_checkpoint_and_rate(monkeypatch, kind):
    events = import_module('numerics_lab.time_integrators.march')
    problem = _RampWithToggle(0.25)
    study = TimeWindow(t0=0.0, t1=1.0, dt=1.0, integrator=kind)
    original_locator = events._locate_bracket
    half_time_states = []
    handled = []

    def locate(lo, g_lo, hi, g_hi, probe, **kwargs):
        if lo < 0.5 < hi:
            probe(0.5)
        return original_locator(lo, g_lo, hi, g_hi, probe, **kwargs)

    def observe(state, time, root):
        if time == 0.5:
            half_time_states.append(float(state[0]))
        return time - root

    def handle(index, time, _state):
        handled.append(time)
        problem.slope = 2.0 if index == 0 else 3.0

    monkeypatch.setattr(events, '_locate_bracket', locate)
    solved = events.integrate_transient(
        problem,
        study,
        np.zeros(1),
        max_newton=2,
        event_fns=(lambda state, time: observe(state, time, 0.25), lambda state, time: observe(state, time, 0.75)),
        on_event=handle,
    )
    assert solved.converged
    first, second = handled
    np.testing.assert_allclose(half_time_states, [0.5, 0.5, 1.0 - first, 1.0 - first], rtol=0.0, atol=1.0e-14)
    at_second = first + 2.0 * (second - first)
    expected = [
        time if time <= first else first + 2.0 * (time - first) if time <= second else at_second + 3.0 * (time - second)
        for time in solved.times
    ]
    np.testing.assert_allclose(np.asarray(solved.states)[:, 0], expected, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(solved.state_rates)[:, 0], [1.0, 2.0, 3.0], atol=1.0e-12)
