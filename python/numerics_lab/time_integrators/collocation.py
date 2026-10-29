"""Direct-collocation defects for ``x' = f(t, x[, u])`` on a node grid: trapezoidal and Hermite-Simpson.

Row ``k`` of the defect is the interval ``[t_k, t_{k+1}]``'s quadrature of the dynamics minus the
state change across it; a trajectory satisfies the collocation scheme exactly when every row
vanishes. The local defect of a smooth exact solution is ``O(h^3)`` for the trapezoid and
``O(h^5)`` for Hermite-Simpson. Controls, when given, are one row per node; Hermite-Simpson takes
the interval midpoint's control as the mean of its ends.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

__all__ = ['hermite_simpson_defects', 'trapezoidal_defects']

Array = NDArray[np.float64]


def _grid(states, times, controls):
    states = np.asarray(states, dtype=float)
    times = np.asarray(times, dtype=float)
    if states.ndim != 2 or states.shape[0] < 2 or times.shape != (states.shape[0],):
        msg = f'collocation needs N + 1 >= 2 node states and their times, got {states.shape} and {times.shape}'
        raise ValueError(msg)
    if not np.all(np.diff(times) > 0.0):
        msg = 'collocation node times must increase strictly'
        raise ValueError(msg)
    if controls is not None:
        controls = np.asarray(controls, dtype=float)
        if controls.ndim != 2 or controls.shape[0] != states.shape[0]:
            msg = f'collocation controls need one row per node, got {controls.shape}'
            raise ValueError(msg)
    return states, times, controls


def _rate(f, t, x, u):
    return np.asarray(f(t, x) if u is None else f(t, x, u), dtype=float)


def trapezoidal_defects(f: Callable[..., Array], states, times, controls=None) -> Array:
    """``x_{k+1} - x_k - h/2 (f_k + f_{k+1})`` per interval."""
    states, times, controls = _grid(states, times, controls)
    rates = np.stack([
        _rate(f, times[k], states[k], None if controls is None else controls[k]) for k in range(states.shape[0])
    ])
    h = np.diff(times)[:, None]
    return states[1:] - states[:-1] - 0.5 * h * (rates[:-1] + rates[1:])


def hermite_simpson_defects(f: Callable[..., Array], states, times, controls=None) -> Array:
    """Hermite-Simpson (separated form eliminated): ``x_{k+1} - x_k - h/6 (f_k + 4 f_m + f_{k+1})``.

    The midpoint state is the cubic Hermite interpolant's, ``(x_k + x_{k+1})/2 + h/8 (f_k - f_{k+1})``.
    """
    states, times, controls = _grid(states, times, controls)
    nodes = states.shape[0]
    rates = np.stack([_rate(f, times[k], states[k], None if controls is None else controls[k]) for k in range(nodes)])
    defects = []
    for k in range(nodes - 1):
        h = times[k + 1] - times[k]
        middle = 0.5 * (states[k] + states[k + 1]) + h / 8.0 * (rates[k] - rates[k + 1])
        u = None if controls is None else 0.5 * (controls[k] + controls[k + 1])
        f_middle = _rate(f, times[k] + 0.5 * h, middle, u)
        defects.append(states[k + 1] - states[k] - h / 6.0 * (rates[k] + 4.0 * f_middle + rates[k + 1]))
    return np.stack(defects)
