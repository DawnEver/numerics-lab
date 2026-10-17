"""The EXACT zero-order-hold march of a linear time-invariant plant under a sampled feedback law.

``dx/dt = M x + sum_k b_k u_k`` with every input ``u_k`` HELD over each sample interval ``dt`` has
the closed-form step ``x_{n+1} = A x_n + sum_k g_k u_k`` with ``A = exp(M dt)`` and
``g_k = M^{-1} (A - I) b_k`` -- no truncation error and no step size to choose, so a sampled
controller closed around it is judged on its control law alone. The controller reads the state at
each sample and returns the inputs held over the next interval; it is called at the last sample too,
so its trace covers every returned instant.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from numerics_lab.time_integrators.checkpoint import (
    CheckpointHistory,
    ScheduleCheckpointState,
    TransientCheckpoint,
)

__all__ = ['march_zoh', 'zoh_discretize']


def zoh_discretize(
    matrix: NDArray[np.float64], columns: Sequence[NDArray[np.float64]], dt: float
) -> tuple[NDArray[np.float64], tuple[NDArray[np.float64], ...]]:
    """``(A, (g_1, ..)``) of the held-input step over ``dt``; ``matrix`` must be invertible."""
    ad = expm(matrix * dt)
    # Exact integral of a held input over the step: M^{-1}(A - I) b.
    m_inv_adi = np.linalg.solve(matrix, ad - np.eye(len(ad)))
    return ad, tuple(m_inv_adi @ column for column in columns)


def march_zoh(
    ad: NDArray[np.float64],
    columns: Sequence[NDArray[np.float64]],
    x0: NDArray[np.float64],
    *,
    t0: float,
    dt: float,
    n_steps: int,
    control: Callable[[float, NDArray[np.float64]], Sequence[float]],
    control_history: CheckpointHistory | None = None,
    restart: TransientCheckpoint | None = None,
    checkpoint_key: str | None = None,
    on_checkpoint: Callable[[TransientCheckpoint], None] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """March ``n_steps`` held-input steps from ``x0`` at ``t0``; ``(times (n+1,), states (n+1, nx))``.

    ``control(t, x)`` returns one held input per column. The inputs are added in column order, so a
    two-input plant advances as ``(A x + g_1 u_1) + g_2 u_2``.

    ``checkpoint_key`` / ``on_checkpoint`` / ``restart`` are the ONE transient checkpoint contract
    (integrator ``zoh``): each sample interval is a schedule segment whose control fired at its start,
    so a record carries the lattice origin, the interval's start and ``control_history``'s state. A
    resume passes ``t0 == restart.time`` and the steps that remain; its instants are the straight
    march's ``origin + k dt``, bit for bit.
    """
    if (on_checkpoint is not None or restart is not None) and checkpoint_key is None:
        msg = 'on_checkpoint and restart require checkpoint_key'
        raise ValueError(msg)
    if control_history is not None and not isinstance(control_history, CheckpointHistory):
        msg = 'control_history must implement checkpoint_history / restore_checkpoint_history'
        raise ValueError(msg)
    origin, first = t0, 0
    if restart is not None:
        if (
            (restart.problem_key, restart.integrator, restart.time) != (checkpoint_key, 'zoh', t0)
            or restart.step_origin is None
            or restart.schedule_state is None
            or len(restart.states[-1]) != len(x0)
            or (restart.schedule_state.tasks is None) != (control_history is None)
        ):
            msg = 'checkpoint does not match the zoh problem, restart time, lattice or control history'
            raise ValueError(msg)
        origin = restart.step_origin
        first = round((t0 - origin) / dt)
        if origin + first * dt != t0:
            msg = 'checkpoint time is not an instant of its zoh lattice'
            raise ValueError(msg)
        if control_history is not None:
            control_history.restore_checkpoint_history(restart.schedule_state.tasks)
        x0 = np.asarray(restart.states[-1], dtype=float)
    times = np.empty(n_steps + 1)
    states = np.empty((n_steps + 1, len(x0)))
    x = np.asarray(x0, dtype=float).copy()
    t = t0
    for step in range(n_steps + 1):
        inputs = control(t, x)
        times[step], states[step] = t, x
        if step < n_steps:
            x = ad @ x
            for column, value in zip(columns, inputs, strict=True):
                x = x + column * value
            t_start = t
            t = origin + (first + step + 1) * dt
            if checkpoint_key is not None and on_checkpoint is not None:
                on_checkpoint(
                    TransientCheckpoint(
                        checkpoint_key,
                        'zoh',
                        t,
                        dt,
                        (tuple(float(value) for value in x),),
                        previous_time=t_start,
                        step_origin=origin,
                        schedule_state=ScheduleCheckpointState(
                            t_start, None if control_history is None else control_history.checkpoint_history()
                        ),
                    )
                )
    return times, states
