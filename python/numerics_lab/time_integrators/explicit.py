"""The explicit sampled march: classic fourth-order Runge-Kutta between task hits -- physics-blind.

:func:`~motronics.gauss.time_integrators.multirate.integrate_sampled` marches a RESIDUAL problem with
the implicit integrators. A non-stiff plant written as an explicit ODE ``u' = f(u, t)`` under held
inputs is integrated far more cheaply, and far more accurately per step, by an explicit fourth-order
scheme: on the sampled drives this module serves, one RK4 step per sample interval agrees with a
converged reference to ~1e-12 where one TR-BDF2 step is ~1e-3 off and a 1e-8 local-error-controlled
TR-BDF2 is ~1e-8 off at 60x the cost (measured 2026-10-08, SRM drive, 2 ms window).

The schedule is :func:`~motronics.gauss.time_integrators.multirate.schedule` -- the same task
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

from motronics.gauss.time_integrators._forms import TransientSolution
from motronics.gauss.time_integrators.multirate import SampledTask, ScheduledTask, schedule

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
) -> TransientSolution:
    """March ``u' = derivative(u, t)`` from ``u0`` over ``[t0, t1]`` with classic RK4.

    ``tasks`` fire at their hits on the state reached there, exactly as in ``integrate_sampled``.
    ``project`` (optional) maps every accepted step onto the set the state is declared to live in
    (a one-sided winding current, say); it is part of the model, applied after the step and never
    inside a stage.
    """
    if not (math.isfinite(dt) and dt > 0.0):
        msg = f'dt must be finite and positive, got {dt!r}'
        raise ValueError(msg)
    if not (math.isfinite(t0) and math.isfinite(t1) and t1 > t0):
        msg = f'need a finite window t0 < t1, got {t0!r}/{t1!r}'
        raise ValueError(msg)
    order, bounds, due = schedule(tasks, t0, t1)
    u = np.asarray(u0, dtype=float).copy()
    out = TransientSolution(times=[t0], states=[u.copy()])
    for (t_a, t_b), firing in zip(itertools.pairwise(bounds), due, strict=True):
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
        for k in range(steps):
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
    return out
