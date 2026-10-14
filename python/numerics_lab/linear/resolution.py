"""What a solved system can resolve: ONE owner for "is this value zero or noise?".

A value is solved FROM a system, and its rounding is set by that system's scale -- the largest
row magnitude ``|J| . |u|`` (plus its forcing) over the whole coupled block -- not by the row it
happens to sit in. A row whose operands are all rounding (a physically zero magnetic field at rest,
a free rate on a homogeneous differentiated constraint) has no scale of its own: judged against its
own operands, its noise is "100% wrong" forever. Judged against the system, it is resolved as zero.

Three sites carried a per-row, noise-derived bar for this and each failed on it (2026-10-13/14,
transformer_3ph_concentric and the moving-band networks): the transient step's local check, the
joint border's row bars and the projected-rate consistency check. They all ask this module now.

The resolution is ``FLOOR_SAFETY * eps`` of the system scale -- the same rounding floor
:func:`motronics.gauss.time_integrators._forms.residual_floor` states. It replaces a row's bar ONLY
where that row's own operands are themselves under it (:func:`row_bar`): such a row is noise and is
judged as zero. Every row with real operands keeps its own relative bar, so a large row can never
hide a small real one (ruling 2026-10-14).
"""

from __future__ import annotations

import numpy as np

__all__ = ['FLOOR_SAFETY', 'row_bar', 'system_resolution']

#: Safety factor on one row sum's rounding estimate: the accumulated error over ``n`` assembled
#: terms grows like ``sqrt(n)`` at best and ``n`` at worst, and 64 buys that headroom while sitting
#: orders below any residual a stalled-but-wrong Newton shows (measured 2026-09-12: 1e8x).
FLOOR_SAFETY = 64.0

_EPS = float(np.finfo(float).eps)
#: The smallest NORMAL double. Below it float64 keeps no relative precision (gradual underflow), so a
#: system that small is representation, not physics -- whatever unit it is in.
_TINY = float(np.finfo(float).tiny)


def system_resolution(*magnitudes) -> float:
    """The smallest magnitude the system with these nonnegative row magnitudes can resolve.

    ``magnitudes`` are the system's own row magnitudes (``|J| . |u|``, ``|rhs|``, ...) over the
    WHOLE coupled block; their largest finite entry is the system scale. Anything at or below the
    returned value is that system's rounding, whatever row it sits in. The scale is never taken
    below the normal range: a system decayed into subnormals (3.3e-308 measured on a moving band
    after its pulse) has no relative precision left to judge anything by.
    """
    largest = 0.0
    for magnitude in magnitudes:
        values = np.abs(np.asarray(magnitude, dtype=float)).ravel()
        values = values[np.isfinite(values)]
        if values.size:
            largest = max(largest, float(values.max()))
    return FLOOR_SAFETY * max(_EPS * largest, _TINY)


def row_bar(own_bar, operands, resolution: float):
    """Each row's bar: its own, unless its operands are all under ``resolution`` -- then it is noise.

    A noise row is judged as zero, i.e. against the system's resolution. A row with real operands
    keeps ``own_bar`` exactly, however small it is next to the system's largest row.
    """
    operands = np.asarray(operands, dtype=float)
    return np.where(operands <= resolution, resolution, np.asarray(own_bar, dtype=float))
