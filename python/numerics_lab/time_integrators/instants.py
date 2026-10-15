"""The one owner of "which clock instant is this": roundoff joins, physics never does.

Every owner that PROPOSES an instant (a lattice target, a source corner, a controller tick, an
interpolation knot) and every owner that RECOGNISES one (``is_corner``, ``on_landing``) compares
through this module. Two owners computing the same physical instant by different arithmetic
(``origin + 36 * dt`` against ``1 / 2400``) land one ulp apart; an owner that tests with a bare
float ``==`` or ``>`` then never sees its event fire and re-proposes it one ulp later, which the
march would take as a ~1e-20 s step.
"""

import numpy as np

__all__ = ['ROUNDOFF_ULPS', 'canonical', 'is_future', 'same_instant']

#: Two spellings of one instant differ by the roundoff of a few products and sums of clock terms;
#: 4 ulps covers the one-ulp split of two spellings (``origin + 36 * dt`` against ``1 / 2400``) and no
#: more: the window is relative, so a wider one grows with the clock (64 ulps is 1.2e-10 s at
#: 1e4 s, a distinct tick) and joins an endpoint 5 ulps off a uniform lattice (pinned 2026-10-15).
ROUNDOFF_ULPS = 4


def same_instant(a: float, b: float) -> bool:
    """Join only arithmetic roundoff, not physically distinct clock instants."""
    return abs(a - b) <= ROUNDOFF_ULPS * max(abs(np.spacing(a)), abs(np.spacing(b)))


def is_future(candidate: float, time: float) -> bool:
    """Strictly later than ``time`` and not ``time`` itself up to roundoff."""
    return candidate > time and not same_instant(candidate, time)


def canonical(time: float, *anchors: float) -> float:
    """The first anchor ``time`` coincides with, else ``time`` itself (a strictly interior edge)."""
    return next((float(anchor) for anchor in anchors if same_instant(time, anchor)), float(time))
