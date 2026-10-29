"""Exact absolute parent targets for implicit fixed marches and fresh restart windows."""

import math

from numerics_lab.time_integrators.instants import same_instant

__all__ = ['fixed_lattice_step_size', 'fixed_lattice_targets']


def fixed_lattice_targets(origin, start, end, dt):
    """Evaluate every target as origin+k*dt, never resumed_start+k*dt.

    Endpoint comparisons are exact. The declared final endpoint is retained;
    nearby physical times are neither rounded nor snapped onto the lattice.
    """
    first = math.floor((start - origin) / dt)
    last = math.floor((end - origin) / dt) + 1
    if last - first > 1_000_000:
        msg = 'fixed parent lattice exceeds its bounded accepted target window'
        raise ValueError(msg)
    result = [origin + tick * dt for tick in range(first, last + 1) if start < origin + tick * dt < end]
    result.append(end)
    if any(not math.isfinite(value) for value in result) or any(
        b <= a for a, b in zip([start, *result], result, strict=False)
    ):
        msg = 'fixed parent lattice must produce finite strictly advancing targets'
        raise ValueError(msg)
    return result


def fixed_lattice_step_size(origin, start, target, dt):
    """Use declared dt only for a complete canonical parent cell.

    An accepted source/device deadline inside that cell leaves a real shorter
    remainder; it must not resume with a fictitious full-length step.
    """
    tick = math.floor((target - origin) / dt)
    for candidate in (tick, tick + 1):
        # A start labelled with a declared endpoint (a resumed checkpoint) is the tick it rounds to.
        if origin + candidate * dt == target and same_instant(origin + (candidate - 1) * dt, start):
            return dt
    return target - start
