"""The multirate scheduler: zero-order holds, rate transitions and the auto sample time.

The planted control for the ZOH is the same RC driven by the UNSAMPLED sine: if the hold were not
applied (the task's value leaking continuously into the plant), the two would agree; they differ by
far more than the ZOH answer's own error against its exact piecewise solution.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import (
    BudgetExceeded,
    Hold,
    RateTransition,
    RunBudget,
    SampledTask,
    ScheduledTask,
    StepTolerance,
    base_sample_time,
    integrate_sampled,
    integrate_transient,
    sample_hits,
)

_TAU, _W, _TS = 0.2, 3.0, 0.05


def test_segment_step_limits_resolve_nonuniform_intervals_without_changing_clocks():
    intervals = []
    fired = []

    def limit(start, stop):
        intervals.append((start, stop))
        return (stop - start) / 4

    sol = integrate_sampled(
        _RC(lambda _t: np.ones(1)),
        TimeWindow(t0=0.0, t1=0.125, dt=0.02, integrator=IntegratorKind.BDF1),
        np.zeros(1),
        [ScheduledTask([0.0, 0.025, 0.075], lambda t, _u: fired.append(t))],
        segment_step_limit=limit,
    )
    assert fired == [0.0, 0.025, 0.075]
    assert intervals == [(0.0, 0.025), (0.025, 0.075), (0.075, 0.125)]
    for start, stop in intervals:
        times = np.asarray([t for t in sol.times if start <= t <= stop])
        assert len(times) == 5
        assert np.max(np.diff(times)) <= (stop - start) / 4 + 1e-16


@pytest.mark.parametrize('limit', [0.0, -1.0, float('nan'), float('inf')])
def test_segment_step_limits_refuse_invalid_resolution(limit):
    with pytest.raises(ValueError, match='segment step limit'):
        integrate_sampled(
            _RC(lambda _t: np.ones(1)),
            TimeWindow(t0=0.0, t1=0.1, dt=0.01),
            np.zeros(1),
            [],
            segment_step_limit=lambda _a, _b: limit,
        )


def test_an_explicit_nonuniform_clock_holds_between_its_actual_hits() -> None:
    grid = np.array([0.0, 0.025, 0.075, 0.125])
    hold, fired = Hold(0.0), []

    def update(t, _state):
        fired.append(t)
        hold.value = 1.0 + t

    task = ScheduledTask(grid, update)
    grid[:] = 99.0  # The schedule owns its declaration, not a caller's mutable buffer.
    solution = integrate_sampled(
        _RC(lambda _t: np.array([hold.value])),
        TimeWindow(t0=0.0, t1=0.125, dt=0.01, integrator=IntegratorKind.TR_BDF2),
        np.zeros(1),
        [task],
        step_control=StepTolerance(rtol=1e-12, atol=1e-14),
    )
    assert fired == [0.0, 0.025, 0.075]
    state = 0.0
    for start, stop in zip(fired, [0.025, 0.075, 0.125], strict=True):
        value = 1.0 + start
        state = value + (state - value) * math.exp(-(stop - start) / _TAU)
        index = solution.times.index(stop)
        assert solution.states[index][0] == pytest.approx(state, abs=1e-8, rel=0)


@pytest.mark.parametrize('hits', [[0.1, 0.0], [0.0, 0.0], [float('nan')], [float('inf')], [[0.0]]])
def test_explicit_clocks_refuse_nonfinite_unordered_or_nonscalar_hits(hits) -> None:
    with pytest.raises(ValueError, match='finite.*strictly increasing'):
        ScheduledTask(hits, lambda _t, _state: None)


def test_explicit_and_periodic_tasks_share_priority_and_exact_window() -> None:
    fired = []
    tasks = [
        SampledTask(0.05, lambda t, _u: fired.append(('inner', t))),
        ScheduledTask([0.0, 0.05, 0.1], lambda t, _u: fired.append(('outer', t)), priority=-1),
    ]
    integrate_sampled(
        _RC(lambda _t: np.zeros(1)),
        TimeWindow(t0=0.05, t1=0.1, dt=0.01),
        np.zeros(1),
        tasks,
    )
    assert fired == [('outer', 0.05), ('inner', 0.05)]


class _RC:
    """``tau u' + u = drive(t)``."""

    def __init__(self, drive) -> None:
        self.drive = drive

    def residual(self, u, u_dot, u_ddot, t):
        return _TAU * u_dot + u - self.drive(t)

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.atleast_2d(weights.alpha0.real + _TAU * weights.alpha1.real)


def _zoh_exact(t_end: float) -> float:
    u, t = 0.0, 0.0
    while t < t_end - 1e-12:
        h = math.sin(_W * t)
        step = min(_TS, t_end - t)
        u = h + (u - h) * math.exp(-step / _TAU)
        t += step
    return u


def test_the_plant_sees_the_held_value_and_matches_the_exact_zoh_answer() -> None:
    hold = Hold(0.0)

    def sample(t, _u):
        hold.value = math.sin(_W * t)

    study = TimeWindow(t0=0.0, t1=1.0, dt=_TS, integrator=IntegratorKind.TR_BDF2)
    tol = StepTolerance(rtol=1e-9, atol=1e-11)
    sol = integrate_sampled(
        _RC(lambda _t: np.array([hold.value])), study, np.zeros(1), [SampledTask(_TS, sample)], step_control=tol
    )
    zoh_err = abs(float(sol.states[-1][0]) - _zoh_exact(1.0))
    continuous = integrate_transient(_RC(lambda t: np.array([math.sin(_W * t)])), study, np.zeros(1), step_control=tol)
    leak = abs(float(continuous.states[-1][0]) - _zoh_exact(1.0))
    assert zoh_err < 1e-6
    assert leak > 1e3 * zoh_err  # planted control: the unheld drive is a different answer


def test_an_integer_ratio_rate_transition_reads_the_fast_value_of_its_own_instant() -> None:
    fast_out, seen = Hold(0.0), []
    rt = RateTransition(fast_out, period=3.0)

    def fast(t, _u):
        fast_out.value = t

    def slow(t, _u):
        seen.append((t, rt.target.value))

    tasks = [rt.task(), SampledTask(3.0, slow), SampledTask(1.0, fast)]
    integrate_sampled(_RC(lambda _t: np.zeros(1)), TimeWindow(t0=0.0, t1=9.0, dt=1.0), np.zeros(1), tasks)
    # fastest clock first; same-clock tasks in declaration order (the latch, then its reader)
    assert seen == [(0.0, 0.0), (3.0, 3.0), (6.0, 6.0)]


def test_a_declared_causal_priority_precedes_fastest_clock_order() -> None:
    fired = []
    tasks = [
        SampledTask(1.0, lambda t, _u: fired.append(('inner', t))),
        SampledTask(2.0, lambda t, _u: fired.append(('outer', t)), priority=-1),
    ]
    integrate_sampled(_RC(lambda _t: np.zeros(1)), TimeWindow(t0=0.0, t1=3.0, dt=0.1), np.zeros(1), tasks)
    assert fired == [('outer', 0.0), ('inner', 0.0), ('inner', 1.0), ('outer', 2.0), ('inner', 2.0)]


def test_a_non_integer_ratio_shares_one_grid() -> None:
    fired: dict[float, list[float]] = {2.0: [], 3.0: []}
    tasks = [SampledTask(p, lambda t, _u, p=p: fired[p].append(t)) for p in (2.0, 3.0)]
    sol = integrate_sampled(_RC(lambda _t: np.zeros(1)), TimeWindow(t0=0.0, t1=7.0, dt=1.0), np.zeros(1), tasks)
    assert fired[2.0] == [0.0, 2.0, 4.0, 6.0]
    assert fired[3.0] == [0.0, 3.0, 6.0]
    assert {0.0, 2.0, 3.0, 4.0, 6.0, 7.0} <= set(np.round(sol.times, 12))


def test_float_roundoff_cannot_fire_a_coincident_clock_twice() -> None:
    fast, slow = [], []
    tasks = [
        SampledTask(0.1, lambda t, _u: fast.append(t)),
        SampledTask(0.3, lambda t, _u: slow.append(t)),
    ]
    integrate_sampled(_RC(lambda _t: np.zeros(1)), TimeWindow(t0=0.0, t1=0.7, dt=0.1), np.zeros(1), tasks)
    np.testing.assert_allclose(fast, np.arange(7) * 0.1, atol=1e-15, rtol=0)
    np.testing.assert_allclose(slow, [0.0, 0.3, 0.6], atol=1e-15, rtol=0)


def test_close_but_distinct_clock_offsets_are_not_merged() -> None:
    first, delayed = [], []
    offset = 1e-10
    tasks = [
        SampledTask(0.3, lambda t, _u: first.append(t)),
        SampledTask(0.3, lambda t, _u: delayed.append(t), offset=offset),
    ]
    sol = integrate_sampled(_RC(lambda _t: np.zeros(1)), TimeWindow(t0=0.0, t1=0.4, dt=0.1), np.zeros(1), tasks)
    np.testing.assert_array_equal(first, [0.0, 0.3])
    np.testing.assert_array_equal(delayed, [offset, 0.3 + offset])
    assert offset in sol.times and 0.3 + offset in sol.times


def test_a_clock_does_not_fire_before_its_activation_offset() -> None:
    fired = []
    sol = integrate_sampled(
        _RC(lambda _t: np.zeros(1)),
        TimeWindow(t0=0.0, t1=0.4, dt=0.1),
        np.zeros(1),
        [SampledTask(0.1, lambda t, _u: fired.append(t), offset=0.2)],
    )
    np.testing.assert_allclose(fired, [0.2, 0.3], atol=1e-15, rtol=0)
    assert sol.times[-1] == pytest.approx(0.4)


@pytest.mark.parametrize(
    ('periods', 'base'), [((1.0, 3.0), 1.0), ((2e-4, 3e-4), 1e-4), ((1e-4, 1.5e-4), 5e-5), ((0.05,), 0.05)]
)
def test_the_auto_sample_time_is_the_rational_gcd(periods, base) -> None:
    assert base_sample_time(periods) == pytest.approx(base, rel=1e-12, abs=0.0)


def test_rational_clock_gcd_is_not_limited_to_machine_integer_products() -> None:
    denominators = (999999937, 999999929, 999999893)
    periods = tuple(1.0 / d for d in denominators)
    expected = 1.0 / math.prod(denominators)
    assert base_sample_time(periods) == pytest.approx(expected, rel=1e-12, abs=0.0)


@pytest.mark.parametrize('period', [0.0, -0.1, float('nan'), float('inf')])
def test_auto_sample_time_uses_the_same_valid_clock_contract(period) -> None:
    with pytest.raises(ValueError, match='finite positive'):
        base_sample_time([period])


def test_the_hit_grid_reads_a_near_integer_count_as_an_integer() -> None:
    hits = sample_hits(0.0, 8e-3, 2e-6)
    assert len(hits) == 4000
    assert hits[-1] < 8e-3


def test_clock_hits_obey_the_half_open_interval_without_physical_slack() -> None:
    assert sample_hits(0.1 + 1e-11, 0.2, 0.1).size == 0
    np.testing.assert_array_equal(sample_hits(0.0, 0.1 + 1e-11, 0.1), [0.0, 0.1])
    assert sample_hits(0.2, 0.2, 0.1).size == 0


@pytest.mark.parametrize('offset', [float('nan'), float('inf'), -float('inf')])
def test_a_clock_rejects_a_nonfinite_activation_offset(offset) -> None:
    with pytest.raises(ValueError, match='finite'):
        SampledTask(0.1, lambda _t, _u: None, offset=offset)


@pytest.mark.parametrize('tick', [1, 100_000])
def test_due_does_not_expand_its_window_with_the_tick_count(tick) -> None:
    task = SampledTask(0.1, lambda _t, _u: None)
    hit = tick * task.period
    assert task.due(hit)
    assert task.due(np.nextafter(hit, np.inf))
    assert not task.due(hit + 1e-10)


def test_due_never_fires_a_negative_clock_tick() -> None:
    task = SampledTask(0.1, lambda _t, _u: None, offset=1.0)
    assert not task.due(0.9)
    assert task.due(1.0)


@pytest.mark.parametrize('t', [float('nan'), float('inf'), -float('inf')])
def test_due_rejects_a_nonfinite_query_instant(t) -> None:
    assert not SampledTask(0.1, lambda _t, _u: None).due(t)


@pytest.mark.parametrize('period', [0.0, -0.1, float('nan'), float('inf')])
def test_the_public_hit_grid_rejects_an_invalid_clock_period(period) -> None:
    with pytest.raises(ValueError, match='finite positive'):
        sample_hits(0.0, 1.0, period)


def test_a_nonzero_study_start_cannot_schedule_a_task_in_its_past() -> None:
    start = 0.1 + 1e-11
    fired = []
    sol = integrate_sampled(
        _RC(lambda _t: np.zeros(1)),
        TimeWindow(t0=start, t1=0.3, dt=0.01),
        np.zeros(1),
        [SampledTask(0.1, lambda t, _u: fired.append(t))],
    )
    assert fired == [0.2]
    assert min(sol.times) == start


@pytest.mark.parametrize('kind', [IntegratorKind.NEWMARK, IntegratorKind.GENERALIZED_ALPHA])
def test_sample_hits_preserve_continuous_velocity_and_acceleration(kind) -> None:
    class Oscillator:
        def residual(self, u, u_dot, u_ddot, t):
            return u_ddot + 4.0 * u

        def jacobian(self, u, u_dot, u_ddot, t, weights):
            return np.atleast_2d(weights.alpha2.real + 4.0 * weights.alpha0.real)

    sol = integrate_sampled(
        Oscillator(),
        TimeWindow(t0=0.0, t1=1.0, dt=0.001, integrator=kind),
        np.ones(1),
        [SampledTask(0.1, lambda _t, _u: None)],
        v0=np.zeros(1),
        a0=np.array([-4.0]),
    )
    np.testing.assert_allclose(np.asarray(sol.states)[:, 0], np.cos(2.0 * np.asarray(sol.times)), atol=1e-5, rtol=0)


def test_the_budget_covers_all_segments_and_discrete_task_time(monkeypatch) -> None:
    from numerics_lab.time_integrators import march  # noqa: PLC0415

    clock = [0.0]
    monkeypatch.setattr(march.time, 'perf_counter', lambda: clock[0])

    def costly_sample(_t, _u):
        clock[0] += 0.6

    with pytest.raises(BudgetExceeded, match='wall') as info:
        integrate_sampled(
            _RC(lambda _t: np.zeros(1)),
            TimeWindow(t0=0.0, t1=0.5, dt=0.1),
            np.zeros(1),
            [SampledTask(0.1, costly_sample)],
            budget=RunBudget(wall_seconds=1.0),
        )
    assert info.value.t_reached == pytest.approx(0.1)
    assert info.value.steps == 1


def test_consistent_sample_restart_recomputes_the_rate_after_the_hold_changes(monkeypatch) -> None:
    from numerics_lab.time_integrators import multirate  # noqa: PLC0415

    drive = Hold(0.0)
    rates = []
    original = multirate.integrate_transient

    class Plant:
        def residual(self, u, u_dot, u_ddot, t):
            return 2.0 * u_dot + u - drive.value

        def jacobian(self, u, u_dot, u_ddot, t, weights):
            return np.atleast_2d(2.0 * weights.alpha1 + weights.alpha0)

    def record(problem, study, state, **kwargs):
        rates.append((state.copy(), kwargs['v0'].copy(), drive.value))
        return original(problem, study, state, **kwargs)

    monkeypatch.setattr(multirate, 'integrate_transient', record)
    sol = integrate_sampled(
        Plant(),
        TimeWindow(t0=0.0, t1=0.25, dt=0.01, integrator=IntegratorKind.TR_BDF2),
        np.zeros(1),
        [SampledTask(0.1, lambda t, _u: setattr(drive, 'value', 1.0 + t))],
        consistent_start=True,
    )
    assert len(rates) == 3
    for state, rate, held in rates:
        np.testing.assert_allclose(2.0 * rate + state, held, atol=1e-14, rtol=0)
    assert rates[0][1][0] == 0.5
    assert sol.times[-1] == 0.25


@pytest.mark.parametrize('kind', [IntegratorKind.NEWMARK, IntegratorKind.GENERALIZED_ALPHA])
def test_first_order_initialization_refuses_second_order_before_firing_tasks(kind) -> None:
    fired = []
    with pytest.raises(ValueError, match='first-order'):
        integrate_sampled(
            None,
            TimeWindow(t0=0.0, t1=0.1, dt=0.01, integrator=kind),
            np.zeros(1),
            [SampledTask(0.1, lambda _t, _u: fired.append(True))],
            consistent_start=True,
        )
    assert fired == []


def test_a_consistent_sample_restart_re_solves_the_algebraic_state_a_held_input_moves() -> None:
    """A held input that moves an ALGEBRAIC unknown is re-solved at the hit; the differential one holds.

    ``u1 = held`` is algebraic and ``u0' = u1 - u0`` differential: each hit's restart places
    ``u1`` on the new hold and leaves ``u0`` where the last segment ended, and the caller's array
    is never written.
    """
    held = [0.0]

    class Plant:
        def residual(self, u, u_dot, u_ddot, t):
            return np.array([u_dot[0] + u[0] - u[1], u[1] - held[0]])

        def jacobian(self, u, u_dot, u_ddot, t, weights):
            return np.array([[weights.alpha1 + weights.alpha0, -weights.alpha0], [0.0, weights.alpha0]])

    state = np.zeros(2)
    sol = integrate_sampled(
        Plant(),
        TimeWindow(t0=0.0, t1=0.2, dt=0.01, integrator=IntegratorKind.TR_BDF2),
        state,
        [ScheduledTask([0.0, 0.1], lambda t, _u: held.__setitem__(0, 1.0 + 10.0 * t))],
        consistent_start=True,
    )
    np.testing.assert_array_equal(state, np.zeros(2))
    times = np.asarray(sol.times)
    states = np.asarray(sol.states)
    first, second = (times > 0.0) & (times <= 0.1 + 1e-12), times > 0.1 + 1e-12
    np.testing.assert_allclose(states[first, 1], 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(states[second, 1], 2.0, rtol=0, atol=1e-12)
    end_of_first = 1.0 - math.exp(-0.1)
    np.testing.assert_allclose(states[-1, 0], 2.0 - (2.0 - end_of_first) * math.exp(-0.1), rtol=1e-4)
