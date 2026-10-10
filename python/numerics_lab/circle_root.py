"""Lexicographic root selection for a degree-two scalar Fourier waveform on a circle."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise

__all__ = ['circle_root']


def _value(coefficients: Sequence[float], x: float) -> float:
    value = 0.0
    for coefficient in reversed(coefficients):
        value = value * x + coefficient
    return value


def _real_roots(coefficients: Sequence[float]) -> list[float]:
    """Isolate real polynomial roots between derivative roots, then bisect monotone intervals.

    Degrees are at most four. Derivatives are solved from linear upwards, so no angular grid
    misses a close root pair. Exact repeated roots are retained; a roundoff-level tangent is
    also considered through the waveform's extrema by the caller.
    """
    p = list(coefficients)
    while len(p) > 1 and p[-1] == 0.0:
        p.pop()
    degree = len(p) - 1
    if degree == 0:
        return []
    derivatives = [p]
    while len(derivatives[-1]) > 2:
        q = derivatives[-1]
        derivatives.append([i * q[i] for i in range(1, len(q))])
    roots: list[float] = []
    for q in reversed(derivatives):
        # Each of two half-angle charts covers only |tan(phi/2)| <= 1. A fixed compact
        # interval avoids enormous Cauchy bounds when a root approaches chart infinity.
        bound = 1.0
        edges = [-bound, *(x for x in roots if -bound < x < bound), bound]
        found = [x for x in edges if _value(q, x) == 0.0]
        for a, b in pairwise(edges):
            fa, fb = _value(q, a), _value(q, b)
            if fa == 0.0 or fb == 0.0 or (fa < 0.0) == (fb < 0.0):
                continue
            for _ in range(80):
                mid = 0.5 * (a + b)
                fm = _value(q, mid)
                if fm == 0.0 or mid in (a, b):
                    a = b = mid
                    break
                if (fa < 0.0) != (fm < 0.0):
                    b = mid
                else:
                    a, fa = mid, fm
            found.append(0.5 * (a + b))
        roots = sorted(set(found))
    return roots


def _angles(coefficients: Sequence[float]) -> list[float]:
    c0, c1, s1, c2, s2 = coefficients
    angles = []
    for offset, sign in ((0.0, 1.0), (math.pi, -1.0)):
        a, b = sign * c1, sign * s1
        p = (c0 + a + c2, 2.0 * b + 4.0 * s2, 2.0 * c0 - 6.0 * c2, 2.0 * b - 4.0 * s2, c0 - a + c2)
        for root in _real_roots(p):
            phi = offset + 2.0 * math.atan(root)
            angles.append(phi - 2.0 * math.pi if phi > math.pi else phi)
    return angles


def circle_root(coefficients: Sequence[float], preferred: Sequence[float]) -> float:
    """Minimise absolute scalar error, then distance to the preferred direction among roots.

    The waveform is ``c0 + c1 cos(phi) + s1 sin(phi) + c2 cos(2phi) + s2 sin(2phi)``.
    An attainable zero is never traded for a secondary improvement. If zero is unreachable,
    the extrema supply the primary minimum. No weighted objective or angular sampling is used.
    """
    if not all(math.isfinite(x) for x in (*coefficients, *preferred)):
        msg = 'circle roots require finite waveform coefficients and a finite preferred direction'
        raise ValueError(msg)
    if all(c == 0.0 for c in coefficients[1:]):
        return math.atan2(preferred[1], preferred[0])
    scale = max(abs(c) for c in coefficients)
    coefficients = tuple(c / scale for c in coefficients)
    direction_scale = max(abs(x) for x in preferred)
    if direction_scale:
        preferred = tuple(x / direction_scale for x in preferred)
    roots = _angles(coefficients)
    if roots:
        return min(roots, key=lambda phi: (-preferred[0] * math.cos(phi) - preferred[1] * math.sin(phi), phi))
    c0, c1, s1, c2, s2 = coefficients
    extrema = _angles((0.0, s1, -c1, 2.0 * s2, -2.0 * c2))
    candidates = [*extrema, 0.0, math.pi]
    return min(
        candidates,
        key=lambda phi: (
            abs(c0 + c1 * math.cos(phi) + s1 * math.sin(phi) + c2 * math.cos(2.0 * phi) + s2 * math.sin(2.0 * phi)),
            -preferred[0] * math.cos(phi) - preferred[1] * math.sin(phi),
            phi,
        ),
    )
