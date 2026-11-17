"""Quadrature node rules: points and weights on an interval, a box or a simple polygon, independent of any
discretisation.

- :func:`gauss_legendre_1d` -- the Gauss-Legendre rule on one interval;
- :func:`gauss_legendre_box` -- its composite tensor product on a box, split into equal cells;
- :func:`simplex_rule` -- a collapsed (Duffy) Gauss-Legendre rule on the ear-clipped triangles of a simple polygon.

Each returns numpy ``(points, weights)``; a caller that trains on them converts to its own tensor type.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

__all__ = ['gauss_legendre_1d', 'gauss_legendre_box', 'simplex_rule']


def gauss_legendre_1d(n: int, lo: float, hi: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``n``-point Gauss-Legendre points and weights mapped onto ``[lo, hi]``.

    Exact for polynomials up to degree ``2n - 1``; the test asserts the BOUNDARY case, since a rule
    checked only on a low-degree integrand cannot tell a correct rule from an over-generous one.
    """
    x, w = _reference_gauss(int(n))
    half = 0.5 * (hi - lo)
    return lo + half * (x + 1.0), half * w


def gauss_legendre_box(
    lower: Sequence[float], upper: Sequence[float], order: int, *, cells: int = 1
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """A composite tensor Gauss-Legendre rule on the box ``[lower, upper]``: points ``(n, dim)``, weights ``(n,)``.

    ``cells`` equal cells per coordinate, ``order`` points per cell per coordinate, exact for polynomials of degree
    ``2 order - 1`` per coordinate on each cell; the weights sum to the box's measure. Give each material region
    its own box so an interface is a cell boundary, never inside one.
    """
    if order < 1 or cells < 1:
        msg = f'a Gauss-Legendre rule takes order >= 1 and cells >= 1, got order={order} cells={cells}'
        raise ValueError(msg)
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    if lo.shape != hi.shape or lo.ndim != 1 or not (hi > lo).all():
        msg = f'a quadrature box has lower < upper per coordinate, got {lo} and {hi}'
        raise ValueError(msg)
    node, weight = _reference_gauss(int(order))
    axes, factors = [], []
    for a, b in zip(lo, hi, strict=True):
        edges = np.linspace(a, b, cells + 1)
        half = 0.5 * (edges[1:] - edges[:-1])
        axes.append(((edges[:-1] + half)[:, None] + half[:, None] * node).ravel())
        factors.append((half[:, None] * weight).ravel())
    points = np.stack([g.ravel() for g in np.meshgrid(*axes, indexing='ij')], axis=1)
    weights = np.prod(np.stack([g.ravel() for g in np.meshgrid(*factors, indexing='ij')], axis=1), axis=1)
    return points, weights


def _cross2(a: NDArray[np.float64], b: NDArray[np.float64], c: NDArray[np.float64]) -> float:
    """Twice the signed area of the triangle ``a b c``: > 0 counter-clockwise."""
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _ear_clip(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """The simple polygon ``v (k, 2)`` as ``(k - 2, 3, 2)`` counter-clockwise triangles."""
    if v.ndim != 2 or v.shape[1] != 2 or len(v) < 3:
        msg = f'a polygon is (k >= 3, 2) vertices, got shape {v.shape}'
        raise ValueError(msg)
    x, y = v[:, 0], v[:, 1]
    area = 0.5 * float(x @ np.roll(y, -1) - y @ np.roll(x, -1))
    if area == 0.0:
        msg = f'a polygon to integrate over has area, got {v.tolist()}'
        raise ValueError(msg)
    pts = v if area > 0 else v[::-1]
    tol = 1e-12 * float(np.max(np.ptp(pts, axis=0))) ** 2  # a flat corner is not an ear
    idx = list(range(len(pts)))
    out: list[NDArray[np.float64]] = []
    while len(idx) > 3:
        for k in range(len(idx)):
            i, j, m = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = pts[i], pts[j], pts[m]
            if _cross2(a, b, c) <= tol:
                continue  # reflex or flat
            others = (pts[n] for n in idx if n not in (i, j, m))
            if any(_cross2(a, b, p) >= -tol and _cross2(b, c, p) >= -tol and _cross2(c, a, p) >= -tol for p in others):
                continue  # another vertex inside or on it
            out.append(np.stack([a, b, c]))
            del idx[k]
            break
        else:
            msg = f'polygon {v.tolist()} is not simple: no ear to clip'
            raise ValueError(msg)
    last = pts[idx]
    if _cross2(*last) <= tol:
        msg = f'polygon {v.tolist()} is not simple: its last triangle is not counter-clockwise'
        raise ValueError(msg)
    out.append(last)
    return np.stack(out)


def simplex_rule(
    vertices: NDArray[np.float64], order: int, *, cells: int = 1
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """A quadrature on the simple polygon ``vertices (k, 2)`` (either orientation), exact for polynomials of TOTAL
    degree ``2 order - 1``.

    The polygon is ear-clipped into triangles, each split into ``cells^2`` similar ones, each carrying the collapsed
    (Duffy) Gauss-Legendre rule of ``order x (order + 1)`` points. Returns points ``(n, 2)`` strictly inside and
    positive weights ``(n,)`` summing to the area; a polygon region is to :func:`gauss_legendre_box` what a box
    region is.
    """
    if order < 1 or cells < 1:
        msg = f'a simplex rule takes order >= 1 and cells >= 1, got order={order} cells={cells}'
        raise ValueError(msg)
    triangles = _ear_clip(np.asarray(vertices, dtype=float))
    gu, wu = _reference_gauss(int(order))
    gv, wv = _reference_gauss(int(order) + 1)
    u, t1 = 0.5 * (gu + 1.0), 0.5 * (gv + 1.0)
    uu, tt = np.meshgrid(u, t1, indexing='ij')
    s, t = (uu * (1.0 - tt)).ravel(), tt.ravel()  # on the unit triangle
    w_ref = (0.25 * wu[:, None] * wv[None, :] * (1.0 - t1)[None, :]).ravel()  # sums to 1/2
    subs = []
    for i in range(cells):
        for j in range(cells - i):
            subs.append(((i, j), (i + 1, j), (i, j + 1)))
            if i + j < cells - 1:
                subs.append(((i + 1, j), (i + 1, j + 1), (i, j + 1)))
    points, weights = [], []
    for a, b, c in triangles:
        e1, e2 = (b - a) / cells, (c - a) / cells
        for corners in subs:
            p0, p1, p2 = (a + i * e1 + j * e2 for i, j in corners)
            points.append(p0 + s[:, None] * (p1 - p0) + t[:, None] * (p2 - p0))
            weights.append(_cross2(p0, p1, p2) * w_ref)
    return np.concatenate(points), np.concatenate(weights)


@lru_cache(maxsize=64)
def _reference_gauss(n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The nodes and weights on ``[-1, 1]``, which depend on ``n`` and on NOTHING ELSE.

    ``leggauss`` solves an eigenproblem, and the span it is mapped onto is a scaling applied after.
    Assembling one motor asked for the same three or four values of ``n`` **8660 times**: 2.455 s of
    a 9.101 s whole-motor assembly, 27% of it, spent recomputing identical arrays. The cache is on
    the reference rule rather than on the mapped one because the interval is different almost every
    call and would make every entry a miss.

    The arrays are marked read-only: a cached array a caller mutates in place is a defect that
    shows up somewhere else entirely.
    """
    x, w = np.polynomial.legendre.leggauss(n)
    x.flags.writeable = False
    w.flags.writeable = False
    return x, w
