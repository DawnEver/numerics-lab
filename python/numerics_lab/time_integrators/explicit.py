"""The explicit sampled march: classic fourth-order Runge-Kutta between task hits -- physics-blind.

:func:`~numerics_lab.time_integrators.multirate.integrate_sampled` marches a RESIDUAL problem with
the implicit integrators. A non-stiff plant written as an explicit ODE ``u' = f(u, t)`` under held
inputs is integrated far more cheaply, and far more accurately per step, by an explicit fourth-order
scheme: on the sampled drives this module serves, one RK4 step per sample interval agrees with a
converged reference to ~1e-12 where one TR-BDF2 step is ~1e-3 off and a 1e-8 local-error-controlled
TR-BDF2 is ~1e-8 off at 60x the cost (measured 2026-10-08, SRM drive, 2 ms window).

The schedule is :func:`~numerics_lab.time_integrators.multirate.schedule` -- the same task
order, hits and segments the implicit march uses. Every hit is a discontinuity of the held inputs,
so a segment's steps never straddle one: the march lands each hit exactly, fires the due tasks on
the state reached there, and steps on. Inside a segment the step is the largest ``<= dt`` that
divides it evenly. There is no step-size controller here: an explicit method cannot see a stiff
mode or an undeclared edge, and the caller that needs either marches with ``integrate_sampled``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from numerics_lab.time_integrators._forms import TransientSolution
from numerics_lab.time_integrators.checkpoint import CheckpointHistory, TransientCheckpoint
from numerics_lab.time_integrators.multirate import (
    SampledTask,
    ScheduledTask,
    checkpointed_schedule,
    schedule_cursor,
)

__all__ = ['integrate_explicit_sampled']

Derivative = Callable[[NDArray[np.float64], float], NDArray[np.float64]]
Projection = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def integrate_explicit_sampled(
    derivative: Derivative,
    t0: float,
    t1: float,
    dt: float,
    u0: NDArray[np.floating],
    tasks: Sequence[SampledTask | ScheduledTask] = (),
    *,
    project: Projection | None = None,
    task_history: CheckpointHistory | None = None,
    restart: TransientCheckpoint | None = None,
    checkpoint_key: str | None = None,
    on_checkpoint: Callable[[TransientCheckpoint], None] | None = None,
) -> TransientSolution:
    """March ``u' = derivative(u, t)`` from ``u0`` over ``[t0, t1]`` with classic RK4.

    ``tasks`` fire at their hits on the state reached there, exactly as in ``integrate_sampled``.
    ``project`` (optional) maps every accepted step onto the set the state is declared to live in
    (a one-sided winding current, say); it is part of the model, applied after the step and never
    inside a stage.
    ``checkpoint_key`` / ``on_checkpoint`` / ``restart`` are the ONE transient checkpoint contract
    (integrator ``rk4``, one state, the schedule cursor and ``task_history``'s state): a resume at
    ``t0 == restart.time`` re-derives its segment's step from the segment start and is bit-identical.
    """
    if not (math.isfinite(dt) and dt > 0.0):
        msg = f'dt must be finite and positive, got {dt!r}'
        raise ValueError(msg)
    if not (math.isfinite(t0) and math.isfinite(t1) and t1 > t0):
        msg = f'need a finite window t0 < t1, got {t0!r}/{t1!r}'
        raise ValueError(msg)
    order, bounds, due = checkpointed_schedule(
        tasks,
        t0,
        t1,
        restart=restart,
        checkpoint_key=checkpoint_key,
        on_checkpoint=on_checkpoint,
        task_history=task_history,
    )
    if restart is not None and (restart.integrator != 'rk4' or len(restart.states[-1]) != np.size(u0)):
        msg = 'checkpoint does not match the rk4 integrator or the state dimension'
        raise ValueError(msg)
    u = np.asarray(u0 if restart is None else restart.states[-1], dtype=float).copy()
    out = TransientSolution(times=[t0], states=[u.copy()])

    def record(t_a: float, h: float) -> TransientCheckpoint:
        previous = out.times[-2] if len(out.times) > 1 else restart.previous_time
        return TransientCheckpoint(
            checkpoint_key,
            'rk4',
            out.times[-1],
            h,
            (tuple(float(x) for x in u),),
            previous_time=previous,
            schedule_state=schedule_cursor(t_a, task_history),
        )

    if restart is not None:
        out.checkpoint = restart
    for segment, ((t_a, t_b), firing) in enumerate(zip(itertools.pairwise(bounds), due, strict=True)):
        for index in sorted(firing):
            order[index].update(t_a, u)
        if t_b <= t_a:
            continue
        span = t_b - t_a
        whole = max(1, round(span / dt))
        if abs(span - whole * dt) <= 4.0 * (np.spacing(t_a) + np.spacing(t_b)):
            # Whole nominal steps keep ``dt`` EXACTLY: ``t_b - t_a`` is off from it by ulps, and on
            # a chopper that decides by comparison an ulp is a different trajectory.
            steps, h = whole, dt
        else:
            steps = math.ceil(span / dt)
            h = span / steps
        first = 0
        if restart is not None and segment == 0:
            # The resumed segment's lattice is the straight march's: same span, same h, same instants.
            first = steps if restart.time == t_b else round((restart.time - t_a) / h)
            if not 0 <= first <= steps or (first < steps and t_a + first * h != restart.time):
                msg = 'checkpoint time is not an instant of its segment lattice'
                raise ValueError(msg)
        for k in range(first, steps):
            t = t_a + k * h
            k1 = derivative(u, t)
            k2 = derivative(u + 0.5 * h * k1, t + 0.5 * h)
            k3 = derivative(u + 0.5 * h * k2, t + 0.5 * h)
            k4 = derivative(u + h * k3, t + h)
            u = u + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            if project is not None:
                u = project(u)
            out.times.append(t_b if k == steps - 1 else t_a + (k + 1) * h)
            out.states.append(u.copy())
            if checkpoint_key is not None:
                out.checkpoint = record(t_a, h)
                if on_checkpoint is not None:
                    on_checkpoint(out.checkpoint)
    return out
