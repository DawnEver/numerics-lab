"""Multirate scheduling: sampled tasks with zero-order holds around a continuous march -- physics-blind.

A sampled controller (or any discrete block) runs at its own clock; the continuous problem it acts on
sees its outputs HELD between samples. This module owns the three pieces that makes precise:

* :func:`sample_hits` -- the ONE rule for clock instants in exactly ``[t0, t1)``; candidate tick
  padding handles quotient roundoff without expanding the physical time interval;
* :func:`base_sample_time` -- the AUTO sample time: the largest period every declared clock is an
  integer multiple of (a rational gcd), so integer AND non-integer clock ratios (2:3, 1:1.5) share
  one grid;
* :func:`integrate_sampled` -- fire due tasks by declared causal priority, then fastest clock (a slower task
  reading through a :class:`RateTransition` sees the faster task's value OF THAT INSTANT by default), then march
  the continuous problem to the next hit. Every hit is a discontinuity of the held inputs, so each
  segment restarts the integrator history -- the same rule an event follows.

A task never sees the problem, and the problem never sees a task: they meet in :class:`Hold` values
the caller wires into its residual. What a task COMPUTES is the caller's.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators._forms import IntegratorState, TransientSolution
from numerics_lab.time_integrators._integrators import get_integrator
from numerics_lab.time_integrators.checkpoint import (
    CheckpointHistory,
    ScheduleCheckpointState,
    TransientCheckpoint,
)
from numerics_lab.time_integrators.initial import consistent_initial_condition
from numerics_lab.time_integrators.instants import same_instant
from numerics_lab.time_integrators.march import BudgetExceeded, integrate_transient

__all__ = [
    'Hold',
    'RateTransition',
    'SampledTask',
    'ScheduledTask',
    'base_sample_time',
    'integrate_sampled',
    'sample_hits',
    'schedule',
]

#: Relative slack for "this instant is a clock hit": far below any physical period ratio, far above
#: the round-off of ``offset + k * period``.
_HIT_TOL = 1e-9


def _validate_clock(period: float, offset: float) -> None:
    if not (np.isfinite(period) and period > 0.0):
        msg = f'a sampled clock needs a finite positive period, got {period!r}'
        raise ValueError(msg)
    if not np.isfinite(offset):
        msg = f'a sampled clock needs a finite offset, got {offset!r}'
        raise ValueError(msg)


@dataclass
class Hold:
    """A zero-order hold: the value a task last wrote, read unchanged until the next write."""

    value: object = 0.0


@dataclass
class SampledTask:
    """A discrete block on its own clock: ``update(t, u)`` runs at ``offset + k * period``."""

    period: float
    update: Callable[[float, NDArray[np.floating]], None]
    offset: float = 0.0
    #: Lower priorities run first at coincident hits (e.g. an outer reference before its inner PI).
    #: Within one priority, fastest clock then declaration order remain the rule.
    priority: int = 0

    def __post_init__(self) -> None:
        _validate_clock(self.period, self.offset)

    def due(self, t: float) -> bool:
        if not np.isfinite(t):
            return False
        k = round((t - self.offset) / self.period)
        return k >= 0 and same_instant(t, self.offset + k * self.period)

    @property
    def order_key(self) -> tuple[int, float]:
        return self.priority, self.period

    def hits(self, t0: float, t1: float) -> NDArray[np.float64]:
        return sample_hits(t0, t1, self.period, self.offset)


@dataclass(frozen=True)
class ScheduledTask:
    """A block on declared, possibly nonuniform instants; no fictitious period.

    Within one priority periodic tasks retain fastest-clock order, followed by
    explicit schedules in declaration order. Use priority for causal dependencies.
    Instants are owned and validated once; the window uses exactly [t0, t1).
    """

    times: Sequence[float]
    update: Callable[[float, NDArray[np.floating]], None]
    priority: int = 0

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        if times.ndim != 1 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
            msg = 'scheduled instants must be finite, scalar and strictly increasing'
            raise ValueError(msg)
        object.__setattr__(self, 'times', tuple(float(t) for t in times))

    @property
    def order_key(self) -> tuple[int, float]:
        return self.priority, float('inf')

    def hits(self, t0: float, t1: float) -> NDArray[np.float64]:
        times = np.asarray(self.times, dtype=float)
        return times[(times >= t0) & (times < t1)]


@dataclass
class RateTransition:
    """Carry a value across clocks: latch ``source`` into ``target`` at the READER's period.

    Fast-to-slow, the slow reader sees the fast value of its own sample instant (not one that changed
    between its samples); slow-to-fast, the fast reader sees the slow value held. Either way the
    transfer is deterministic, which is the whole point -- an unlatched read across clocks depends on
    which task happened to run last.
    """

    source: Hold
    period: float
    target: Hold = field(default_factory=Hold)
    offset: float = 0.0

    def task(self) -> SampledTask:
        def latch(_t: float, _u: NDArray[np.floating]) -> None:
            self.target.value = self.source.value

        return SampledTask(self.period, latch, self.offset)


def sample_hits(t0: float, t1: float, period: float, offset: float = 0.0) -> NDArray[np.float64]:
    """Clock instants in exactly ``[t0, t1)``; a clock starts at nonnegative tick zero."""
    _validate_clock(period, offset)
    # Quotient rounding can land on either side of an integer. Generate one extra
    # candidate at each boundary, then decide membership from the actual instants.
    # A relative clock tolerance must never admit a hit before a window's start or
    # drop a genuine final hit: doing so changes the held-input problem itself.
    k_first = max(0, int(np.floor((t0 - offset) / period)) - 1)
    k_last = int(np.ceil((t1 - offset) / period)) + 1
    hits = offset + period * np.arange(k_first, k_last + 1, dtype=float)
    return hits[(hits >= t0) & (hits < t1)]


def base_sample_time(periods: Sequence[float], *, max_denominator: int = 10**9) -> float:
    """The AUTO sample time: the largest ``T`` with every period an integer multiple of it.

    Each period is read as the nearest rational (``max_denominator`` bounds the search) and the gcd of
    the rationals is taken: ``gcd(p1/q1, p2/q2) = gcd(p1 q2, p2 q1) / (q1 q2)``. ``(1e-4, 1.5e-4)``
    gives ``5e-5``; ``(1, 3)`` gives ``1``. Periods with no common rational base within the bound
    RAISE rather than returning a base that only approximately divides them.
    """
    if not periods:
        msg = 'base_sample_time needs at least one period'
        raise ValueError(msg)
    for period in periods:
        _validate_clock(period, 0.0)
    fracs = [Fraction(p).limit_denominator(max_denominator) for p in periods]
    for p, f in zip(periods, fracs, strict=True):
        if abs(float(f) - p) > _HIT_TOL * p:
            msg = f'period {p!r} has no rational form within denominator {max_denominator}'
            raise ValueError(msg)
    g = fracs[0]
    for f in fracs[1:]:
        num = np.gcd(g.numerator * f.denominator, f.numerator * g.denominator)
        g = Fraction(int(num), g.denominator * f.denominator)
    return float(g)


def schedule(
    tasks: Sequence[SampledTask | ScheduledTask], t0: float, t1: float
) -> tuple[list[SampledTask | ScheduledTask], list[float], list[set[int]]]:
    """The ONE task schedule over ``[t0, t1]``: ``(ordered tasks, segment bounds, due set per segment)``.

    Coincident tasks run by ascending priority, then fastest clock, then declaration order; segment
    ``j`` spans ``bounds[j]..bounds[j + 1]`` and fires ``due[j]`` (indices into the ordered tasks) at
    its start. ``bounds`` starts at ``t0`` and ends at ``t1``.
    """
    order = sorted(tasks, key=lambda task: task.order_key)
    scheduled = sorted((float(h), index) for index, task in enumerate(order) for h in task.hits(t0, t1))
    hits: list[float] = []
    due: list[set[int]] = []
    for hit, index in scheduled:
        # Only arithmetic roundoff joins clock instants; the much wider hit-recognition
        # tolerance must not merge physically distinct offsets or fire a task at another clock.
        if hits and same_instant(hit, hits[-1]):
            due[-1].add(index)
        else:
            hits.append(hit)
            due.append({index})
    if not hits or hits[0] > t0:
        hits.insert(0, t0)
        due.insert(0, set())
    return order, [*hits, t1], due


def checkpointed_schedule(
    tasks: Sequence[SampledTask | ScheduledTask],
    t0: float,
    t1: float,
    *,
    restart: TransientCheckpoint | None,
    checkpoint_key: str | None,
    on_checkpoint,
    task_history: CheckpointHistory | None,
) -> tuple[list[SampledTask | ScheduledTask], list[float], list[set[int]]]:
    """:func:`schedule`, resumed from ``restart``'s cursor when given -- the ONE resume rule of both sampled marches.

    A resume re-derives the schedule from the checkpoint's ``segment_start`` (so every later bound is
    the same float the straight march used), restores the task owner, and does NOT re-fire the first
    segment's tasks: they fired before the step the record was taken after. Refuses a checkpoint
    request that could not resume: tasks without a ``task_history`` owner, a record of another model or
    instant, or one without a schedule cursor.
    """
    if (on_checkpoint is not None or restart is not None) and checkpoint_key is None:
        msg = 'on_checkpoint and restart require checkpoint_key'
        raise ValueError(msg)
    if checkpoint_key is not None and tasks and task_history is None:
        msg = 'a checkpointed sampled march with tasks requires task_history: the tasks hold state the plant cannot rebuild'
        raise ValueError(msg)
    if task_history is not None and not isinstance(task_history, CheckpointHistory):
        msg = 'task_history must implement checkpoint_history / restore_checkpoint_history'
        raise ValueError(msg)
    start = t0
    if restart is not None:
        if restart.schedule_state is None or (restart.problem_key, restart.time) != (checkpoint_key, t0):
            msg = 'checkpoint does not match the problem, the restart time or a sampled schedule cursor'
            raise ValueError(msg)
        if (restart.schedule_state.tasks is None) != (task_history is None):
            msg = 'checkpoint task history does not match the declared task_history owner'
            raise ValueError(msg)
        if task_history is not None:
            task_history.restore_checkpoint_history(restart.schedule_state.tasks)
        start = restart.schedule_state.segment_start
    order, bounds, due = schedule(tasks, start, t1)
    if restart is not None:
        due[0] = set()
    return order, bounds, due


def schedule_cursor(segment_start: float, task_history: CheckpointHistory | None) -> ScheduleCheckpointState:
    """The cursor a sampled march records after an accepted step inside the segment from ``segment_start``."""
    return ScheduleCheckpointState(
        float(segment_start), None if task_history is None else task_history.checkpoint_history()
    )


def integrate_sampled(
    problem,
    window: TimeWindow,
    u0: NDArray[np.floating],
    tasks: Sequence[SampledTask | ScheduledTask],
    *,
    consistent_start: bool = False,
    segment_step_limit: Callable[[float, float], float] | None = None,
    task_history: CheckpointHistory | None = None,
    restart: TransientCheckpoint | None = None,
    checkpoint_key: str | None = None,
    on_checkpoint: Callable[[TransientCheckpoint], None] | None = None,
    **march,
) -> TransientSolution:
    """March ``problem`` over ``window`` with ``tasks`` firing at their hits and holding between them.

    ``window.dt`` bounds the continuous step inside a segment; every ``march`` keyword passes to
    :func:`~numerics_lab.time_integrators.march.integrate_transient`. Coincident tasks run by
    ascending priority, then fastest clock, then declaration order. The returned trajectory is the segments joined at their hits
    (each hit appears once), with every convergence label carried.
    ``consistent_start`` re-solves each segment's start after its tasks update the held inputs:
    the differential unknowns stay where the last segment ended, the algebraic ones a held input
    moves (an ideal source's node voltage) are placed on the new hold, and the rates follow. The
    joined trajectory keeps the left limit at each hit; this option is for index-1 first-order
    systems, not acceleration.
    ``segment_step_limit(start, stop)`` optionally tightens the window's maximum step
    for each actual interval. It changes resolution, never task instants or priorities.
    ``checkpoint_key`` / ``on_checkpoint`` / ``restart`` are the ONE transient checkpoint contract: every
    segment's accepted records carry the schedule cursor and ``task_history``'s state (required whenever
    there are tasks), and a resume at ``window.t0 == restart.time`` is bit-identical to the straight march.
    """
    if consistent_start and window.integrator in (IntegratorKind.NEWMARK, IntegratorKind.GENERALIZED_ALPHA):
        msg = 'consistent_start initializes first-order rates, not second-order acceleration'
        raise ValueError(msg)
    order, bounds, due = checkpointed_schedule(
        tasks,
        window.t0,
        window.t1,
        restart=restart,
        checkpoint_key=checkpoint_key,
        on_checkpoint=on_checkpoint,
        task_history=task_history,
    )
    u = np.asarray(u0 if restart is None else restart.states[-1], dtype=float)
    out = TransientSolution(times=[window.t0], states=[u], state_rates=[], residual_times=[], rate_state_overrides={})
    v, a = march.pop('v0', None), march.pop('a0', None)
    integrator = get_integrator(window.integrator)
    budget = march.pop('budget', None)
    started = time.perf_counter()
    for segment, ((t_a, t_b), firing) in enumerate(zip(itertools.pairwise(bounds), due, strict=True)):
        resuming = restart is not None and segment == 0
        if budget is not None:
            budget.check(started, out.times[-1], u, len(out.times) - 1)
        for index in sorted(firing):
            task = order[index]
            task.update(t_a, u)
            if budget is not None:
                budget.check(started, out.times[-1], u, len(out.times) - 1)
        if t_b <= t_a:
            continue
        dt = min(window.dt, t_b - t_a)
        if segment_step_limit is not None:
            limit = segment_step_limit(t_a, t_b)
            if not np.isfinite(limit) or limit <= 0:
                msg = f'segment step limit must be finite and positive, got {limit!r}'
                raise ValueError(msg)
            dt = min(dt, limit)
        if resuming and restart.time >= t_b:
            # The record closed this segment: hand its history on exactly as the straight march does.
            out.checkpoint = restart
            out.final_integrator_state = IntegratorState(
                us=[np.array(state, dtype=float) for state in restart.states],
                t=restart.time,
                v=None if restart.velocity is None else np.array(restart.velocity, dtype=float),
                a=None if restart.acceleration is None else np.array(restart.acceleration, dtype=float),
                h=restart.previous_dt,
            )
            carried = integrator.restart(out.final_integrator_state, u, t_b)
            v, a = carried.v, carried.a
            continue
        if consistent_start and not resuming:
            start = consistent_initial_condition(problem, u, t_a, tol=march.get('tol', 1e-10))
            u, v = start.u, start.u_dot
            if budget is not None:
                budget.check(started, out.times[-1], u, len(out.times) - 1)
        remaining = budget
        if budget is not None and budget.wall_seconds is not None:
            remaining = replace(budget, wall_seconds=max(0.0, budget.wall_seconds - (time.perf_counter() - started)))
        stamp = None
        if checkpoint_key is not None:

            def stamp(record, origin=t_a):
                return replace(record, schedule_state=schedule_cursor(origin, task_history))

        try:
            seg = integrate_transient(
                problem,
                TimeWindow(t0=restart.time if resuming else t_a, t1=t_b, dt=dt, integrator=window.integrator),
                u,
                v0=None if resuming else v,
                a0=None if resuming else a,
                budget=remaining,
                restart=replace(restart, schedule_state=None) if resuming else None,
                checkpoint_key=checkpoint_key,
                on_checkpoint=None if on_checkpoint is None else (lambda record, s=stamp: on_checkpoint(s(record))),
                **march,
            )
        except BudgetExceeded as exc:
            raise BudgetExceeded(exc.reason, exc.t_reached, exc.u_reached, len(out.times) - 1 + exc.steps) from exc
        out.times.extend(seg.times[1:])
        out.states.extend(seg.states[1:])
        assert out.state_rates is not None and seg.state_rates is not None
        base = len(out.state_rates)
        out.state_rates.extend(seg.state_rates)
        assert out.residual_times is not None and seg.residual_times is not None
        out.residual_times.extend(seg.residual_times)
        if out.rate_state_overrides is not None:
            if seg.rate_state_overrides is None:
                out.rate_state_overrides = None
            else:
                out.rate_state_overrides.update({base + i: v for i, v in seg.rate_state_overrides.items()})
        out.events.extend(seg.events)
        out.nonconverged_times.extend(seg.nonconverged_times)
        out.nonconverged_probe_times.extend(seg.nonconverged_probe_times)
        out.floor_converged_times.extend(seg.floor_converged_times)
        out.worst_step_residual = max(out.worst_step_residual, seg.worst_step_residual)
        u = np.asarray(seg.states[-1], dtype=float)
        assert seg.final_integrator_state is not None
        out.final_integrator_state = seg.final_integrator_state.copy()
        if stamp is not None and seg.checkpoint is not None:
            out.checkpoint = stamp(seg.checkpoint)
        carried = integrator.restart(out.final_integrator_state, u, t_b)
        v, a = carried.v, carried.a
    return out
