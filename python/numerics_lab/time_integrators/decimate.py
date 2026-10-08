"""Bounded-error output decimation of a marched trajectory -- physics-blind.

A march stores every accepted step; a reader needs only enough points that linear interpolation
between them reproduces the trajectory. :func:`decimate` keeps the fewest points (greedy, forward)
such that EVERY dropped point lies within ``bound`` of the chord between its kept neighbours, unknown
by unknown -- the error claim is checked on the points themselves, never assumed. The first and last
points and every located event instant are always kept: an event is where the trajectory kinks.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from motronics.gauss.time_integrators._forms import TransientSolution

__all__ = ['decimate', 'interpolation_error']


def _chord_ok(t: NDArray, u: NDArray, a: int, b: int, bound: NDArray) -> bool:
    if b - a < 2:
        return True
    w = ((t[a + 1 : b] - t[a]) / (t[b] - t[a]))[:, None]
    chord = u[a] + w * (u[b] - u[a])
    return bool(np.all(np.abs(u[a + 1 : b] - chord) <= bound))


def interpolation_error(times, states, kept_times, kept_states) -> NDArray[np.floating]:
    """The per-unknown max ``|u - interp(kept)(t)|`` over ``times`` -- the decimation's own audit."""
    u = np.asarray(states, dtype=float).reshape(len(times), -1)
    k = np.asarray(kept_states, dtype=float).reshape(len(kept_times), -1)
    rebuilt = np.column_stack([np.interp(times, kept_times, k[:, j]) for j in range(k.shape[1])])
    return np.max(np.abs(u - rebuilt), axis=0)


def decimate(solution: TransientSolution, bound: float | NDArray[np.floating]) -> TransientSolution:
    """The solution with stored points dropped while every dropped point stays within ``bound``.

    ``bound`` is absolute, a scalar or one value per unknown (the units differ per unknown). The
    rate record is dropped with the points it belongs to; convergence labels are kept whole.
    """
    t = np.asarray(solution.times, dtype=float)
    u = np.asarray(solution.states, dtype=float).reshape(t.size, -1)
    bound_arr = np.broadcast_to(np.asarray(bound, dtype=float), u.shape[1:])
    pinned = {0, t.size - 1} | {int(np.searchsorted(t, te)) for _i, te in solution.events if te <= t[-1]}
    kept = [0]
    a = 0
    b = 1
    while b < t.size:
        if b in pinned or not _chord_ok(t, u, a, b + 1 if b + 1 < t.size else b, bound_arr):
            # the chord to b+1 would violate (or b must be kept): b is the last good anchor
            kept.append(b)
            a = b
        b += 1
    if kept[-1] != t.size - 1:
        kept.append(t.size - 1)
    kept = sorted(set(kept))
    rates = None
    if solution.state_rates is not None and len(solution.state_rates) == t.size - 1:
        rates = [solution.state_rates[i - 1] for i in kept[1:]]
    return replace(
        solution,
        times=[solution.times[i] for i in kept],
        states=[solution.states[i] for i in kept],
        state_rates=rates,
        rate_state_overrides=None,
    )
