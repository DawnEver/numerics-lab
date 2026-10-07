"""Multirate scheduling: sampled tasks with zero-order holds around a continuous march -- physics-blind.

A sampled controller (or any discrete block) runs at its own clock; the continuous problem it acts on
sees its outputs HELD between samples. This module owns the three pieces that makes precise:

* :func:`sample_hits` -- the ONE rule for "which instants of ``[t0, t1)`` are hits of a clock", with
  the floating-point near-integer tolerance every hand-rolled ``round((t1 - t0)/dt)`` re-derived;
* :func:`base_sample_time` -- the AUTO sample time: the largest period every declared clock is an
  integer multiple of (a rational gcd), so integer AND non-integer clock ratios (2:3, 1:1.5) share
  one grid;
* :func:`integrate_sampled` -- fire the due tasks at each hit (fastest clock first, so a slower task
  reading through a :class:`RateTransition` sees the faster task's value OF THAT INSTANT), then march
  the continuous problem to the next hit. Every hit is a discontinuity of the held inputs, so each
  segment restarts the integrator history -- the same rule an event follows.

A task never sees the problem, and the problem never sees a task: they meet in :class:`Hold` values
the caller wires into its residual. What a task COMPUTES is the caller's.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

from motronics.contracts.spec.study_spec import TransientStudy
from motronics.gauss.time_integrators._forms import TransientSolution
from motronics.gauss.time_integrators.march import integrate_transient

__all__ = ['Hold', 'RateTransition', 'SampledTask', 'base_sample_time', 'integrate_sampled', 'sample_hits']

#: Relative slack for "this instant is a clock hit": far below any physical period ratio, far above
#: the round-off of ``offset + k * period``.
_HIT_TOL = 1e-9


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

    def __post_init__(self) -> None:
        if not (np.isfinite(self.period) and self.period > 0.0):
            msg = f'a sampled task needs a finite positive period, got {self.period!r}'
            raise ValueError(msg)

    def due(self, t: float) -> bool:
        k = (t - self.offset) / self.period
        return k > -_HIT_TOL and abs(k - round(k)) <= _HIT_TOL * max(1.0, abs(k))


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
    """The clock's hits in ``[t0, t1)``: ``offset + k * period``, a near-integer count read as integer."""
    k0 = (t0 - offset) / period
    k_first = round(k0) if abs(k0 - round(k0)) <= _HIT_TOL * max(1.0, abs(k0)) else int(np.ceil(k0))
    k1 = (t1 - offset) / period
    k_last = round(k1) - 1 if abs(k1 - round(k1)) <= _HIT_TOL * max(1.0, abs(k1)) else int(np.floor(k1))
    return offset + period * np.arange(k_first, k_last + 1, dtype=float)


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


def integrate_sampled(
    problem,
    study: TransientStudy,
    u0: NDArray[np.floating],
    tasks: Sequence[SampledTask],
    **march,
) -> TransientSolution:
    """March ``problem`` over ``study`` with ``tasks`` firing at their hits and holding between them.

    ``study.dt`` bounds the continuous step inside a segment; every ``march`` keyword passes to
    :func:`~motronics.gauss.time_integrators.march.integrate_transient`. Tasks due at the same
    instant run fastest-clock first, same-clock tasks in declaration order. The returned trajectory is the segments joined at their hits
    (each hit appears once), with every convergence label carried.
    """
    order = sorted(tasks, key=lambda task: task.period)
    hits = sorted({float(h) for task in tasks for h in sample_hits(study.t0, study.t1, task.period, task.offset)})
    if not hits or hits[0] > study.t0 + _HIT_TOL * study.dt:
        hits.insert(0, study.t0)
    bounds = [*hits, study.t1]
    u = np.asarray(u0, dtype=float)
    out = TransientSolution(times=[study.t0], states=[u], state_rates=[], rate_state_overrides={})
    for t_a, t_b in itertools.pairwise(bounds):
        for task in order:
            if task.due(t_a):
                task.update(t_a, u)
        if t_b - t_a <= _HIT_TOL * study.dt:
            continue
        seg = integrate_transient(
            problem,
            TransientStudy(t0=t_a, t1=t_b, dt=min(study.dt, t_b - t_a), integrator=study.integrator),
            u,
            **march,
        )
        out.times.extend(seg.times[1:])
        out.states.extend(seg.states[1:])
        assert out.state_rates is not None and seg.state_rates is not None
        base = len(out.state_rates)
        out.state_rates.extend(seg.state_rates)
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
    return out
