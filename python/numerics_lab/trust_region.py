"""Trust-region primitives: the agreement ratio, the radius update, and the dogleg step.

The model is ``m(p) = g . p + 1/2 p . B p`` inside ``||p|| <= radius``. The matrix-free subproblem
solver, Steihaug-CG, lives with the other Krylov methods (:func:`numerics_lab.linear.krylov.steihaug_cg`).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

__all__ = ['agreement', 'dogleg', 'update_radius']

Vector = NDArray[np.float64]


def agreement(actual_reduction: float, predicted_reduction: float) -> float:
    """``rho = actual / predicted``; a non-positive prediction is no model at all and reads ``-inf``."""
    if not predicted_reduction > 0.0:
        return -math.inf
    return actual_reduction / predicted_reduction


def update_radius(
    radius: float,
    rho: float,
    step_norm: float,
    *,
    max_radius: float,
    poor: float = 0.25,
    good: float = 0.75,
    shrink: float = 0.25,
    grow: float = 2.0,
) -> float:
    """The next radius (Nocedal and Wright, Algorithm 4.1).

    ``rho < poor`` shrinks to ``shrink * step_norm``; ``rho > good`` with the step ON the boundary
    grows to ``min(grow * radius, max_radius)``; otherwise the radius is kept. Whether to ACCEPT the
    step is the caller's separate decision (usually ``rho > eta`` for a small ``eta``).
    """
    if not 0.0 < poor < good < 1.0:
        msg = f'the agreement thresholds must satisfy 0 < poor < good < 1, got {poor!r}, {good!r}'
        raise ValueError(msg)
    if rho < poor:
        return shrink * step_norm
    if rho > good and step_norm >= (1.0 - 1e-12) * radius:
        return min(grow * radius, max_radius)
    return radius


def dogleg(
    gradient: Vector,
    hessian_apply: Callable[[Vector], Vector],
    newton_step: Vector,
    radius: float,
) -> Vector:
    """The dogleg step for a POSITIVE DEFINITE model: Newton inside, else the path through the Cauchy point.

    ``newton_step`` is ``-B^{-1} g``, solved by the caller (the factorisation is its choice). The
    path runs from 0 to the unconstrained Cauchy point ``-(g.g / g.Bg) g`` and on to the Newton
    step; the step is where it leaves the ball.
    """
    if float(np.linalg.norm(newton_step)) <= radius:
        return newton_step
    g_norm = float(np.linalg.norm(gradient))
    if g_norm == 0.0:
        return np.zeros_like(gradient)
    curvature = float(gradient @ hessian_apply(gradient))
    if not curvature > 0.0:
        msg = 'dogleg needs a positive definite model; g . B g <= 0'
        raise ValueError(msg)
    cauchy = -(g_norm * g_norm / curvature) * gradient
    cauchy_norm = float(np.linalg.norm(cauchy))
    if cauchy_norm >= radius:
        return -(radius / g_norm) * gradient
    leg = newton_step - cauchy
    a = float(leg @ leg)
    b = 2.0 * float(cauchy @ leg)
    c = cauchy_norm * cauchy_norm - radius * radius
    tau = (-b + math.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
    return cauchy + tau * leg
