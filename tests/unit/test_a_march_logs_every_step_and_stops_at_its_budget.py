"""The march's structured JSONL step/event log and its wall/memory budget abort.

Planted controls: a march WITHOUT a log writes nothing to the stream, and a march with an
unreachable budget finishes -- so the record and the abort are the capability's, not the stream's
or the clock's.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import BudgetExceeded, RunBudget, StepTolerance, integrate_transient


class _Decay:
    def residual(self, u, u_dot, u_ddot, t):
        return u_dot + 20.0 * u

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.atleast_2d(20.0 * weights.alpha0.real + weights.alpha1.real)


class _Ramp:
    """``u' = 1`` with an event at ``u = 0.25``."""

    def residual(self, u, u_dot, u_ddot, t):
        return u_dot - 1.0

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.atleast_2d(weights.alpha1.real)


_STUDY = TimeWindow(t0=0.0, t1=0.5, dt=0.1, integrator=IntegratorKind.BDF2)


def _records(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_every_accepted_step_rejection_and_event_is_one_otel_record() -> None:
    stream = io.StringIO()
    sol = integrate_transient(
        _Decay(), _STUDY, np.ones(1), step_control=StepTolerance(rtol=1e-4, atol=1e-8), step_log=stream
    )
    records = _records(stream)
    assert {'timestamp', 'severity_text', 'body', 'attributes'} <= set(records[0])
    steps = [r for r in records if r['body'] == 'step']
    assert len(steps) == len(sol.times) - 1
    assert [r['attributes']['march.t'] for r in steps] == sol.times[1:]
    assert any(r['body'] == 'solve' for r in records)

    events = io.StringIO()
    integrate_transient(_Ramp(), _STUDY, np.zeros(1), event_fns=(lambda u, _t: float(u[0]) - 0.25,), step_log=events)
    fired = [r for r in _records(events) if r['body'] == 'event']
    assert len(fired) == 1
    assert fired[0]['attributes']['march.t'] == pytest.approx(0.25, abs=1e-6)


def test_planted_control_no_log_argument_writes_nothing() -> None:
    integrate_transient(_Decay(), _STUDY, np.ones(1))  # no step_log: nothing anywhere to assert but the absence
    stream = io.StringIO()
    integrate_transient(_Decay(), _STUDY, np.ones(1), step_log=None)
    assert stream.getvalue() == ''


def test_a_wall_budget_aborts_with_the_state_reached() -> None:
    with pytest.raises(BudgetExceeded, match='wall') as info:
        integrate_transient(_Decay(), _STUDY, np.ones(1), budget=RunBudget(wall_seconds=0.0))
    assert info.value.steps == 0
    assert info.value.t_reached == pytest.approx(0.0)


def _resident() -> int:
    return 1 << 30


def test_a_memory_budget_aborts_and_a_generous_one_does_not() -> None:
    with pytest.raises(BudgetExceeded, match='resident'):
        integrate_transient(_Decay(), _STUDY, np.ones(1), budget=RunBudget(memory_bytes=1, resident_bytes=_resident))
    generous = RunBudget(wall_seconds=3600.0, memory_bytes=1 << 50, resident_bytes=_resident)
    sol = integrate_transient(_Decay(), _STUDY, np.ones(1), budget=generous)
    assert sol.times[-1] == pytest.approx(0.5)


def test_a_memory_budget_needs_the_callers_probe() -> None:
    # The engine reads no process memory itself (numpy and scipy only), so the caller supplies it.
    with pytest.raises(ValueError, match='resident_bytes'):
        RunBudget(memory_bytes=1 << 30)


def test_a_wall_budget_stops_discarded_trials_at_the_last_accepted_state(monkeypatch) -> None:
    from numerics_lab.time_integrators import march  # noqa: PLC0415

    clock = [0.0]
    monkeypatch.setattr(march.time, 'perf_counter', lambda: clock[0])

    class CostlyDecay(_Decay):
        def residual(self, u, u_dot, u_ddot, t):
            clock[0] += 1.0
            return super().residual(u, u_dot, u_ddot, t)

    with pytest.raises(BudgetExceeded, match='wall') as info:
        integrate_transient(
            CostlyDecay(),
            _STUDY,
            np.ones(1),
            step_control=StepTolerance(rtol=1e-6, atol=1e-12),
            budget=RunBudget(wall_seconds=0.5),
        )
    assert info.value.steps == 0
    assert info.value.t_reached == 0.0
    np.testing.assert_array_equal(info.value.u_reached, np.ones(1))


@pytest.mark.parametrize('seconds', [-1.0, np.nan, np.inf])
def test_an_invalid_wall_budget_is_refused(seconds) -> None:
    with pytest.raises(ValueError, match='wall_seconds'):
        RunBudget(wall_seconds=seconds)


@pytest.mark.parametrize('size', [-1, 0])
def test_an_invalid_memory_budget_is_refused(size) -> None:
    with pytest.raises(ValueError, match='memory_bytes'):
        RunBudget(memory_bytes=size, resident_bytes=_resident)


def test_the_solution_carries_the_newton_work_every_logged_solve_did() -> None:
    """The total a per-step cost is divided by is the log's own per-solve counts summed, probes included."""
    stream = io.StringIO()
    sol = integrate_transient(
        _Decay(), _STUDY, np.ones(1), step_control=StepTolerance(rtol=1e-4, atol=1e-8), step_log=stream
    )
    solves = [r['attributes'] for r in _records(stream) if r['body'] == 'solve']
    counted = sum(int(next(v for k, v in a.items() if k.endswith('iterations'))) for a in solves)
    assert counted >= len(sol.times) - 1
    assert sol.newton_iterations == counted
