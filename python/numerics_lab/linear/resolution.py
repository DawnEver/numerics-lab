"""What a solved system can resolve: ONE owner for "is this value zero or noise?".

A value is solved FROM a system, and its rounding is set by that system's scale -- the largest
row magnitude ``|J| . |u|`` (plus its forcing) over the whole coupled block -- not by the row it
happens to sit in. A row whose operands are all rounding (a physically zero magnetic field at rest,
a free rate on a homogeneous differentiated constraint) has no scale of its own: judged against its
own operands, its noise is "100% wrong" forever. Judged against the system, it is resolved as zero.

Three sites carried a per-row, noise-derived bar for this and each failed on it (2026-10-13/14,
case_l and the moving-band networks): the transient step's local check, the
joint border's row bars and the projected-rate consistency check. They all ask this module now.

The resolution is ``FLOOR_SAFETY * eps`` of the system scale -- the same rounding floor
:func:`numerics_lab.time_integrators._forms.residual_floor` states. It replaces a row's bar ONLY
where that row is HOMOGENEOUS (its own forcing exactly zero: no source, no declared data) and its
operands are themselves under it (:func:`row_bar`). Such a row's true value is exactly zero, so
anything there is rounding and is judged as zero. A row with its own nonzero forcing is real at any
magnitude and keeps its own relative bar, so a large row never hides a small real one (rulings
2026-10-14).
"""

from __future__ import annotations

import numpy as np

__all__ = ['ADMISSION_BAR', 'FLOOR_SAFETY', 'row_bar', 'system_resolution']

#: Safety factor on one row sum's rounding estimate: the accumulated error over ``n`` assembled
#: terms grows like ``sqrt(n)`` at best and ``n`` at worst, and 64 buys that headroom while sitting
#: orders below any residual a stalled-but-wrong Newton shows (measured 2026-09-12: 1e8x).
FLOOR_SAFETY = 64.0

#: THE ONE per-row relative bar a coupled state is judged by: the finite-storage admission's exit
#: ``tolerance`` and the stacked march's step acceptance (ruling 2026-10-14). A march that accepted on
#: the NORM of its residual could keep a small-scale row at 1.2x this bar, and the admission at the
#: next law change then refused the very state the march had accepted -- two bars for one fact
#: (case_a a5, row 16580 at t=0.0134838).
ADMISSION_BAR = 1e-10

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


def row_bar(own_bar, operands, resolution: float, forcing):
    """Each row's bar: its own, unless the row is NOISE -- homogeneous with rounding-sized operands.

    ``forcing`` is each row's own source/declared data. A row is noise iff that forcing is exactly
    zero AND its ``operands`` are at or below ``resolution``; it is judged as zero, against the
    system's resolution. Every other row keeps ``own_bar`` exactly, however small.
    """
    if forcing is None:
        return np.asarray(own_bar, dtype=float)
    operands = np.asarray(operands, dtype=float)
    homogeneous = np.asarray(forcing, dtype=float) == 0.0
    return np.where(homogeneous & (operands <= resolution), resolution, np.asarray(own_bar, dtype=float))
