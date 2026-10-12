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
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """March ``n_steps`` held-input steps from ``x0`` at ``t0``; ``(times (n+1,), states (n+1, nx))``.

    ``control(t, x)`` returns one held input per column. The inputs are added in column order, so a
    two-input plant advances as ``(A x + g_1 u_1) + g_2 u_2``.
    """
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
            t = t0 + (step + 1) * dt
    return times, states
