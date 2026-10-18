"""Quasi-Newton secant updates: BFGS, the L-BFGS two-loop recursion, SR1 and Broyden.

A secant pair is ``s = x_{k+1} - x_k`` and ``y = g_{k+1} - g_k`` (``g`` a gradient or a residual).
Two forms are published, and both are used:

- MATRIX-FREE, for large systems: :func:`two_loop` applies the BFGS inverse model ``H g`` over a
  list of :class:`SecantPair` on top of a caller-supplied ``H0`` action; :func:`broyden_apply`
  applies Broyden's ("good") inverse model as stored rank-1 :class:`BroydenUpdate` terms. The
  nonlinear BFGS / L-BFGS / Broyden strategies run on these.
- DENSE, for small models (a trust-region or SQP Hessian): :func:`bfgs_update`,
  :func:`bfgs_inverse_update`, :func:`sr1_update`, :func:`broyden_inverse_update`.

Every update keeps its secant equation exactly in exact arithmetic (``H y = s`` or ``B s = y``), and
an update whose denominator vanishes is SKIPPED (returns ``None`` or the model unchanged), never
divided through.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    'SECANT_FLOOR',
    'BroydenUpdate',
    'SecantPair',
    'bfgs_inverse_update',
    'bfgs_update',
    'broyden_apply',
    'broyden_inverse_update',
    'broyden_update',
    'secant_pair',
    'sr1_update',
    'two_loop',
]

#: A curvature ``y . s`` (BFGS) or denominator ``s . H y`` (Broyden) at or below this magnitude
#: carries no information; the pair is skipped.
SECANT_FLOOR = 1e-30

Vector = NDArray[np.float64]


@dataclass(frozen=True)
class SecantPair:
    """One BFGS secant pair with ``rho = 1 / (y . s)``."""

    s: Vector
    y: Vector
    rho: float


def secant_pair(s: Vector, y: Vector, *, floor: float = SECANT_FLOOR) -> SecantPair | None:
    """The pair ``(s, y)`` with its ``rho``, or ``None`` when ``|y . s| <= floor``."""
    ys = float(y @ s)
    if abs(ys) > floor:
        return SecantPair(s.copy(), y, 1.0 / ys)
    return None


def two_loop(q: Vector, pairs: Sequence[SecantPair], apply_h0: Callable[[Vector], Vector]) -> Vector:
    """``H q`` by the L-BFGS two-loop recursion over ``pairs`` (oldest first), ``H0`` applied by ``apply_h0``.

    BFGS and L-BFGS differ only in how many pairs they keep, so this one body serves both.
    """
    q = q.copy()
    alphas = []
    for pair in reversed(pairs):
        a = pair.rho * float(pair.s @ q)
        q = q - a * pair.y
        alphas.append(a)
    z = apply_h0(q)
    for pair, a in zip(pairs, reversed(alphas), strict=True):
        b = pair.rho * float(pair.y @ z)
        z = z + (a - b) * pair.s
    return z


@dataclass(frozen=True)
class BroydenUpdate:
    """One rank-1 term of Broyden's inverse model: ``H <- H + u s^T H`` stored as ``(s, u)``."""

    s: Vector
    u: Vector


def broyden_update(s: Vector, h_y: Vector, *, floor: float = SECANT_FLOOR) -> BroydenUpdate | None:
    """Broyden's "good" inverse update from ``s`` and the current model's ``H y``; ``None`` if ``|s . H y| <= floor``.

    ``H+ = H + (s - H y) s^T H / (s^T H y)``, so ``H+ y = s``.
    """
    denominator = float(s @ h_y)
    if abs(denominator) > floor:
        return BroydenUpdate(s.copy(), (s - h_y) / denominator)
    return None


def broyden_apply(v: Vector, updates: Sequence[BroydenUpdate]) -> Vector:
    """Apply the stored updates (oldest first) to ``v = H0 r``: the model's ``H r``."""
    for update in updates:
        v = v + (update.s @ v) * update.u
    return v


def bfgs_inverse_update(h: NDArray[np.float64], s: Vector, y: Vector) -> NDArray[np.float64]:
    """``(I - rho s y^T) H (I - rho y s^T) + rho s s^T``; ``H`` unchanged when ``y . s <= 0``.

    A non-positive curvature would destroy positive definiteness, so that pair is skipped.
    """
    ys = float(y @ s)
    if not ys > SECANT_FLOOR:
        return h
    rho = 1.0 / ys
    hy = h @ y
    return h - rho * (np.outer(s, hy) + np.outer(hy, s)) + (rho * rho * float(y @ hy) + rho) * np.outer(s, s)


def bfgs_update(b: NDArray[np.float64], s: Vector, y: Vector) -> NDArray[np.float64]:
    """The direct BFGS Hessian update ``B - B s s^T B / (s^T B s) + y y^T / (y^T s)``; skipped when ``y . s <= 0``."""
    ys = float(y @ s)
    bs = b @ s
    sbs = float(s @ bs)
    if not ys > SECANT_FLOOR or not sbs > SECANT_FLOOR:
        return b
    return b - np.outer(bs, bs) / sbs + np.outer(y, y) / ys


def sr1_update(b: NDArray[np.float64], s: Vector, y: Vector, *, skip: float = 1e-8) -> NDArray[np.float64]:
    """The symmetric rank-1 update ``B + (y - B s)(y - B s)^T / ((y - B s)^T s)``.

    Skipped (``B`` returned) when ``|(y - B s) . s| < skip ||s|| ||y - B s||`` (Nocedal and Wright,
    eq. 6.26): the denominator is then too small to trust. SR1 does not keep positive definiteness.
    """
    v = y - b @ s
    denominator = float(v @ s)
    if abs(denominator) < skip * float(np.linalg.norm(s)) * float(np.linalg.norm(v)) or denominator == 0.0:
        return b
    return b + np.outer(v, v) / denominator


def broyden_inverse_update(h: NDArray[np.float64], s: Vector, y: Vector) -> NDArray[np.float64]:
    """Dense Broyden "good" inverse update ``H + (s - H y) s^T H / (s^T H y)``; skipped on a vanishing denominator."""
    hy = h @ y
    denominator = float(s @ hy)
    if abs(denominator) <= SECANT_FLOOR:
        return h
    return h + np.outer(s - hy, s @ h) / denominator
