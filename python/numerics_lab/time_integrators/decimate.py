"""Bounded-error output decimation of a marched trajectory -- physics-blind.

A march stores every accepted step; a reader needs only enough points that linear interpolation
between them reproduces the trajectory. :func:`decimate` greedily removes points (forward)
such that EVERY dropped point lies within ``bound`` of the chord between its kept neighbours, unknown
by unknown -- the error claim is checked on the points themselves, never assumed. The first and last
points and every located event instant are always kept: an event is where the trajectory kinks.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from numerics_lab.time_integrators._forms import TransientSolution

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
    Rates are retained at kept points; residual-side overrides are reindexed. Landed convergence
    diagnostics remain pinned. The bound concerns the stored piecewise-linear trajectory, not
    the unknown exact continuous solution; no globally minimal point count is claimed.
    """
    t = np.asarray(solution.times, dtype=float)
    if t.ndim != 1 or not t.size or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0.0):
        msg = 'decimation needs nonempty finite strictly increasing times'
        raise ValueError(msg)
    if len(solution.states) != t.size:
        msg = 'decimation needs one state per stored time'
        raise ValueError(msg)
    u = np.asarray(solution.states, dtype=float).reshape(t.size, -1)
    if not u.shape[1] or not np.all(np.isfinite(u)):
        msg = 'decimation needs finite nonempty states'
        raise ValueError(msg)
    bound_arr = np.broadcast_to(np.asarray(bound, dtype=float), u.shape[1:])
    if not np.all(np.isfinite(bound_arr)) or np.any(bound_arr < 0.0):
        msg = 'decimation bound must be finite and nonnegative'
        raise ValueError(msg)
    indices = {float(ti): i for i, ti in enumerate(t)}
    pinned = {0, t.size - 1}
    for te in [*(te for _i, te in solution.events), *solution.nonconverged_times]:
        if te not in indices:
            msg = 'landed event or convergence diagnostic is absent from stored times'
            raise ValueError(msg)
        pinned.add(indices[te])
    # Floor labels also include discarded probes; only landed labels pin output.
    pinned.update(indices[te] for te in solution.floor_converged_times if te in indices)
    overrides = solution.rate_state_overrides
    if overrides is not None:
        if any(i < 0 or i >= t.size - 1 for i in overrides):
            msg = 'residual-side override index is outside the stored rate record'
            raise ValueError(msg)
        pinned.update(i + 1 for i in overrides)
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
    if solution.state_rates is not None:
        if len(solution.state_rates) != t.size - 1:
            msg = 'decimation needs one recorded rate per accepted state'
            raise ValueError(msg)
        rates = [solution.state_rates[i - 1] for i in kept[1:]]
    residual_times = None
    if solution.residual_times is not None:
        if len(solution.residual_times) != t.size - 1:
            msg = 'decimation needs one residual evaluation time per accepted state'
            raise ValueError(msg)
        residual_times = [solution.residual_times[i - 1] for i in kept[1:]]
    return replace(
        solution,
        times=[solution.times[i] for i in kept],
        states=[solution.states[i] for i in kept],
        state_rates=rates,
        residual_times=residual_times,
        rate_state_overrides=(
            None if overrides is None else {j: overrides[i - 1] for j, i in enumerate(kept[1:]) if i - 1 in overrides}
        ),
    )
