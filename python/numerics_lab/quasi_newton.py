"""Quasi-Newton secant updates for nonlinear SOLVES: the secant pair, the two-loop recursion and Broyden.

A secant pair is ``s = x_{k+1} - x_k`` and ``y = r_{k+1} - r_k`` (``r`` a residual). The nonlinear
BFGS / L-BFGS / Broyden strategies run on these MATRIX-FREE forms: :func:`two_loop` applies the
inverse model ``H r`` over a list of :class:`SecantPair` on top of a caller-supplied ``H0`` action;
:func:`broyden_apply` applies Broyden's ("good") inverse model as stored rank-1
:class:`BroydenUpdate` terms, and :func:`broyden_inverse_update` is its dense form.

Every update keeps its secant equation exactly in exact arithmetic (``H y = s``), and an update whose
denominator vanishes is SKIPPED (returns ``None`` or the model unchanged), never divided through. The
minimizer's dense Hessian models and its pair memory are optimi-lab's (``optimi_lab.operator.quasi_newton``).
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
    'broyden_apply',
    'broyden_inverse_update',
    'broyden_update',
    'secant_pair',
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


def broyden_inverse_update(h: NDArray[np.float64], s: Vector, y: Vector) -> NDArray[np.float64]:
    """Dense Broyden "good" inverse update ``H + (s - H y) s^T H / (s^T H y)``; skipped on a vanishing denominator."""
    hy = h @ y
    denominator = float(s @ hy)
    if abs(denominator) <= SECANT_FLOOR:
        return h
    return h + np.outer(s - hy, s @ h) / denominator
