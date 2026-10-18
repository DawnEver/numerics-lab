"""Quasi-Newton secant updates: BFGS, the L-BFGS two-loop recursion and pair memory, SR1 and Broyden.

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
    'LbfgsMemory',
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


class LbfgsMemory:
    """The L-BFGS pair memory: the ONE bookkeeping behind every limited-memory BFGS model here.

    Keeps the last ``size`` secant pairs (oldest first), SKIPS a pair whose curvature ``y . s`` is not
    above :data:`SECANT_FLOOR`, and scales by ``gamma = s.y / y.y`` of the newest kept pair (``1``
    when empty). One model, two forms: :meth:`inverse_apply` is the two-loop ``H q`` on ``gamma I``;
    :meth:`hessian` is the compact direct form ``B = theta I - W M W^T`` with ``theta = 1 / gamma``
    (Byrd, Nocedal and Schnabel 1994, eq. 3.5), which a bound-constrained (Cauchy point) step needs.
    """

    def __init__(self, size: int) -> None:
        if size < 1:
            msg = f'an L-BFGS memory holds at least one secant pair, got {size!r}'
            raise ValueError(msg)
        self.size = size
        self.pairs: tuple[SecantPair, ...] = ()
        self.gamma = 1.0
        self._dimension: int | None = None

    def push(self, s: Vector, y: Vector) -> bool:
        """Keep ``(s, y)`` if its curvature is positive, dropping the oldest past ``size``; True if kept."""
        s, y = np.asarray(s, dtype=float), np.asarray(y, dtype=float)
        self._dimension = s.size
        ys = float(y @ s)
        if not ys > SECANT_FLOOR:
            return False
        self.pairs = (*self.pairs[max(0, len(self.pairs) + 1 - self.size) :], SecantPair(s.copy(), y.copy(), 1.0 / ys))
        self.gamma = ys / float(y @ y)
        return True

    def clear(self) -> None:
        """Forget every pair (a restart): the model is the identity again."""
        self.pairs, self.gamma = (), 1.0

    def inverse_apply(self, q: Vector) -> Vector:
        """``H q``: the two-loop recursion over the kept pairs on ``H0 = gamma I``."""
        return two_loop(np.asarray(q, dtype=float), self.pairs, lambda v: self.gamma * v)

    def compact_factors(self) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
        """``(theta, W, M)`` with ``B = theta I - W M W^T``; ``W`` is ``n x 2m`` (``[Y, theta S]``)."""
        theta = 1.0 / self.gamma
        if not self.pairs:
            return theta, np.zeros((self._dimension or 0, 0)), np.zeros((0, 0))
        s = np.column_stack([pair.s for pair in self.pairs])
        y = np.column_stack([pair.y for pair in self.pairs])
        sy = s.T @ y
        lower = np.tril(sy, -1)
        middle = np.block([[-np.diag(np.diag(sy)), lower.T], [lower, theta * (s.T @ s)]])
        return theta, np.hstack([y, theta * s]), np.linalg.inv(middle)

    def hessian(self) -> NDArray[np.float64]:
        """The dense direct model ``B`` (the compact form expanded); for small ``n`` and for tests."""
        if self._dimension is None:
            msg = 'an L-BFGS memory that never saw a pair has no dimension'
            raise ValueError(msg)
        theta, w, m = self.compact_factors()
        return theta * np.eye(self._dimension) - w @ m @ w.T
