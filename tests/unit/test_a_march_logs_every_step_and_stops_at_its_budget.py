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

from motronics.contracts.spec.study_spec import IntegratorKind, TransientStudy
from motronics.gauss.time_integrators import BudgetExceeded, RunBudget, StepTolerance, integrate_transient


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


_STUDY = TransientStudy(t0=0.0, t1=0.5, dt=0.1, integrator=IntegratorKind.BDF2)


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
    assert info.value.steps == 1
    assert info.value.t_reached == pytest.approx(0.1)


def test_a_memory_budget_aborts_and_a_generous_one_does_not() -> None:
    with pytest.raises(BudgetExceeded, match='rss'):
        integrate_transient(_Decay(), _STUDY, np.ones(1), budget=RunBudget(memory_bytes=1))
    sol = integrate_transient(_Decay(), _STUDY, np.ones(1), budget=RunBudget(wall_seconds=3600.0, memory_bytes=1 << 50))
    assert sol.times[-1] == pytest.approx(0.5)
