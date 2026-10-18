"""Matrix-free Krylov methods, self-implemented: CG, MINRES, and Steihaug's truncated CG.

Each takes the operator as a callable ``v -> A v`` (or anything with ``@``), so an assembled matrix,
a ``LinearOperator`` and a Jacobian-free product all work. These are the PUBLIC primitives an
optimiser composes (a Newton-CG inner solve, a trust-region subproblem); the registry's ``cg`` and
``minres`` backends (:data:`numerics_lab.linear.LINEAR_SOLVER_REGISTRY`) are the solve seam's own
and are characterised by the linear goldens.

- :func:`conjugate_gradient` -- symmetric positive definite ``A``.
- :func:`minres` -- symmetric, possibly indefinite ``A`` (Paige and Saunders).
- :func:`steihaug_cg` -- the trust-region subproblem ``min g.p + 1/2 p.B p`` s.t. ``||p|| <= radius``,
  stopping on the boundary or at negative curvature.

Convergence is ``||b - A x|| <= max(rtol ||b||, atol)``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

__all__ = ['KrylovSolution', 'SteihaugStep', 'conjugate_gradient', 'minres', 'steihaug_cg']

Vector = NDArray[np.float64]


@dataclass(frozen=True)
class KrylovSolution:
    """``x``, the iterations taken, and whether the residual met its bar (``residual_norm`` is the true one)."""

    x: Vector
    iterations: int
    converged: bool
    residual_norm: float


def _operator(a) -> Callable[[Vector], Vector]:
    if callable(a):
        return lambda v: np.asarray(a(v), dtype=float)
    return lambda v: np.asarray(a @ v, dtype=float)


def _bar(b_norm: float, rtol: float, atol: float) -> float:
    if rtol < 0.0 or atol < 0.0:
        msg = f'Krylov tolerances must be nonnegative, got rtol={rtol!r} atol={atol!r}'
        raise ValueError(msg)
    return max(rtol * b_norm, atol)


def _finish(apply, b, x, iterations, bar) -> KrylovSolution:
    residual = float(np.linalg.norm(b - apply(x)))
    return KrylovSolution(x, iterations, residual <= bar, residual)


def conjugate_gradient(
    a, b: Vector, *, x0: Vector | None = None, rtol: float = 1e-10, atol: float = 0.0, max_iterations: int | None = None
) -> KrylovSolution:
    """Hestenes-Stiefel CG for a symmetric positive definite ``a``."""
    apply = _operator(a)
    b = np.asarray(b, dtype=float)
    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=float).copy()
    limit = 10 * b.size if max_iterations is None else max_iterations
    bar = _bar(float(np.linalg.norm(b)), rtol, atol)
    r = b - apply(x)
    d = r.copy()
    rr = float(r @ r)
    iterations = 0
    while math.sqrt(rr) > bar and iterations < limit:
        ad = apply(d)
        curvature = float(d @ ad)
        if not curvature > 0.0:
            break  # not positive definite along d: CG has no step to take
        alpha = rr / curvature
        x = x + alpha * d
        r = r - alpha * ad
        rr_next = float(r @ r)
        d = r + (rr_next / rr) * d
        rr = rr_next
        iterations += 1
    return _finish(apply, b, x, iterations, bar)


def minres(
    a, b: Vector, *, x0: Vector | None = None, rtol: float = 1e-10, atol: float = 0.0, max_iterations: int | None = None
) -> KrylovSolution:
    """MINRES (Paige and Saunders) for a symmetric, possibly indefinite ``a``: minimises ``||b - A x||`` on the Krylov space."""
    apply = _operator(a)
    b = np.asarray(b, dtype=float)
    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=float).copy()
    limit = 10 * b.size if max_iterations is None else max_iterations
    bar = _bar(float(np.linalg.norm(b)), rtol, atol)
    r = b - apply(x)
    beta = float(np.linalg.norm(r))
    iterations = 0
    if beta <= bar:
        return _finish(apply, b, x, iterations, bar)
    v_previous = np.zeros_like(b)
    v = r / beta
    w_older = np.zeros_like(b)
    w_old = np.zeros_like(b)
    gamma_old, gamma = 1.0, 1.0
    sigma_old, sigma = 0.0, 0.0
    eta = beta
    while iterations < limit:
        p = apply(v) - beta * v_previous
        alpha = float(v @ p)
        p = p - alpha * v
        beta_next = float(np.linalg.norm(p))
        delta = gamma * alpha - gamma_old * sigma * beta
        rho1 = math.hypot(delta, beta_next)
        rho2 = sigma * alpha + gamma_old * gamma * beta
        rho3 = sigma_old * beta
        if rho1 == 0.0:
            break  # an exactly singular step; the residual cannot be reduced further
        gamma_next, sigma_next = delta / rho1, beta_next / rho1
        w = (v - rho3 * w_older - rho2 * w_old) / rho1
        x = x + gamma_next * eta * w
        eta = -sigma_next * eta
        iterations += 1
        if abs(eta) <= bar or beta_next == 0.0:
            break
        v_previous, v = v, p / beta_next
        w_older, w_old = w_old, w
        gamma_old, gamma = gamma, gamma_next
        sigma_old, sigma = sigma, sigma_next
        beta = beta_next
    return _finish(apply, b, x, iterations, bar)


@dataclass(frozen=True)
class SteihaugStep:
    """The subproblem's step, why the iteration stopped, and how many CG iterations it took."""

    step: Vector
    stop: Literal['converged', 'boundary', 'negative_curvature', 'max_iterations']
    iterations: int


def _to_boundary(z: Vector, d: Vector, radius: float) -> Vector:
    """``z + tau d`` with ``tau >= 0`` and ``||z + tau d|| = radius``."""
    dd = float(d @ d)
    zd = float(z @ d)
    zz = float(z @ z)
    tau = (-zd + math.sqrt(zd * zd + dd * (radius * radius - zz))) / dd
    return z + tau * d


def steihaug_cg(
    hessian, gradient: Vector, radius: float, *, rtol: float = 1e-8, max_iterations: int | None = None
) -> SteihaugStep:
    """Steihaug-Toint truncated CG for ``min g.p + 1/2 p.B p`` subject to ``||p|| <= radius``.

    Stops when the model gradient ``g + B p`` falls below ``rtol ||g||``, when an iterate would leave
    the ball (the step is cut at the boundary), or when ``d.B d <= 0`` (the step follows ``d`` to the
    boundary: negative curvature is descent all the way out).
    """
    if not radius > 0.0:
        msg = f'the trust radius must be positive, got {radius!r}'
        raise ValueError(msg)
    apply = _operator(hessian)
    g = np.asarray(gradient, dtype=float)
    limit = 10 * g.size if max_iterations is None else max_iterations
    z = np.zeros_like(g)
    r = g.copy()
    bar = rtol * float(np.linalg.norm(g))
    if float(np.linalg.norm(r)) <= bar:
        return SteihaugStep(z, 'converged', 0)
    d = -r
    rr = float(r @ r)
    for iteration in range(1, limit + 1):
        bd = apply(d)
        curvature = float(d @ bd)
        if not curvature > 0.0:
            return SteihaugStep(_to_boundary(z, d, radius), 'negative_curvature', iteration)
        alpha = rr / curvature
        z_next = z + alpha * d
        if float(np.linalg.norm(z_next)) >= radius:
            return SteihaugStep(_to_boundary(z, d, radius), 'boundary', iteration)
        r = r + alpha * bd
        rr_next = float(r @ r)
        z = z_next
        if math.sqrt(rr_next) <= bar:
            return SteihaugStep(z, 'converged', iteration)
        d = -r + (rr_next / rr) * d
        rr = rr_next
    return SteihaugStep(z, 'max_iterations', limit)
