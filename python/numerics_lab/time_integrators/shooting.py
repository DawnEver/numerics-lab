"""Shooting: the multiple-shooting defects, and matrix-free periodic shooting on a stateless cycle map."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from operator import index

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, gmres

from numerics_lab.line_search import armijo

__all__ = ['ShootingResult', 'periodic_shooting', 'shooting_defects']

Vector = NDArray[np.float64]


def shooting_defects(
    propagate: Callable[[Vector, float, float], Vector], states: NDArray[np.float64], times: Vector
) -> NDArray[np.float64]:
    """The continuity defects of multiple shooting: row ``k`` is ``propagate(x_k, t_k, t_{k+1}) - x_{k+1}``.

    ``states`` has one row per node (``N + 1`` of them) and ``times`` the node instants; the
    ``N`` defects vanish exactly when the nodes lie on one trajectory of ``propagate``. Boundary or
    periodicity conditions are the caller's further rows (``x_N - x_0`` closes a period).
    """
    states = np.asarray(states, dtype=float)
    times = np.asarray(times, dtype=float)
    if states.ndim != 2 or times.shape != (states.shape[0],) or states.shape[0] < 2:
        msg = f'shooting needs N + 1 >= 2 node states and their times, got {states.shape} and {times.shape}'
        raise ValueError(msg)
    if not np.all(np.diff(times) > 0.0):
        msg = 'shooting node times must increase strictly'
        raise ValueError(msg)
    return np.stack([
        np.asarray(propagate(states[k].copy(), float(times[k]), float(times[k + 1])), dtype=float) - states[k + 1]
        for k in range(states.shape[0] - 1)
    ])


@dataclass(frozen=True)
class ShootingResult:
    state: NDArray[np.float64]
    residual_norm: float
    cycle_evaluations: int


#: The Armijo trials a periodic-shooting Newton step may spend before it takes the full step.
_SHOOTING_BACKTRACKS = 8
#: The Krylov dimension (cycle evaluations) one Newton step may spend on its linear solve.
_KRYLOV_DIMENSION = 20


def periodic_shooting(
    cycle_map: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    initial: NDArray[np.float64],
    *,
    atol: float = 1e-9,
    rtol: float = 1e-7,
    max_iterations: int = 30,
    sign: int = 1,
) -> ShootingResult:
    """Solve ``Phi(u) - sign*u = 0`` without a dense monodromy matrix (Jacobian-free Newton-GMRES).

    The cycle Jacobian acts by a forward difference of the map, solved by GMRES to a forcing
    tolerance ``min(0.1, ||F||)``; the Newton step is searched by Armijo backtracking on ``||F||`` and
    taken whole when no trial decreases it. The map must start a fresh cycle on every call; mutable
    material or integration history is part of the state and must not leak between trial
    evaluations. Failure raises rather than publishing a nonperiodic seed as steady state.
    """
    seed = np.asarray(initial, dtype=float)
    if isinstance(sign, bool) or not isinstance(sign, int) or sign not in (-1, 1):
        msg = 'shooting sign must be the explicit integer +1 or -1'
        raise ValueError(msg)
    if seed.ndim != 1 or not seed.size or not np.all(np.isfinite(seed)):
        msg = 'shooting initial state must be a nonempty finite vector'
        raise ValueError(msg)
    if not np.isfinite(atol) or atol <= 0 or not np.isfinite(rtol) or rtol < 0:
        msg = 'shooting tolerances require finite atol > 0 and rtol >= 0'
        raise ValueError(msg)
    if isinstance(max_iterations, bool) or index(max_iterations) < 1:
        msg = 'shooting max_iterations must be a positive integer'
        raise ValueError(msg)
    evaluations = 0

    def residual(state):
        nonlocal evaluations
        end = np.asarray(cycle_map(np.asarray(state, dtype=float).copy()), dtype=float)
        evaluations += 1
        if end.shape != seed.shape or not np.all(np.isfinite(end)):
            msg = 'cycle map must return a finite vector with the initial shape'
            raise ValueError(msg)
        return end - sign * state

    state = seed.copy()
    defect = residual(state)
    for _ in range(max_iterations):
        if float(np.linalg.norm(defect, ord=np.inf)) <= atol:
            break
        step = _newton_step(residual, state, defect)
        norm = float(np.linalg.norm(defect))
        search = armijo(
            lambda lam, s=state, d=step: float(np.linalg.norm(residual(s + lam * d))),
            norm,
            -norm,
            tries=_SHOOTING_BACKTRACKS,
        )
        state = state + (search.step if search.accepted else 1.0) * step
        defect = residual(state)
    norm = float(np.linalg.norm(defect, ord=np.inf))
    if norm > atol + rtol * float(np.linalg.norm(state, ord=np.inf)):
        msg = 'periodic shooting failed the final cycle-closure tolerance'
        raise RuntimeError(msg)
    return ShootingResult(state.copy(), norm, evaluations)


def _newton_step(residual, state: Vector, defect: Vector) -> Vector:
    """``J^{-1}(-F)`` with ``J v`` a forward difference of ``residual`` at ``state``, solved by GMRES."""
    scale = math.sqrt(np.finfo(float).eps) * max(1.0, float(np.linalg.norm(state)))

    def jv(v):
        v = np.ravel(v)
        v_norm = float(np.linalg.norm(v))
        if v_norm == 0.0:
            return np.zeros_like(v)
        h = scale / v_norm
        return (residual(state + h * v) - defect) / h

    operator = LinearOperator((state.size, state.size), matvec=jv, dtype=float)
    forcing = min(0.1, float(np.linalg.norm(defect)))
    step, _ = gmres(operator, -defect, rtol=forcing, atol=0.0, restart=min(state.size, _KRYLOV_DIMENSION), maxiter=1)
    step = np.asarray(step, dtype=float)
    if not np.all(np.isfinite(step)) or not np.any(step):
        msg = 'periodic shooting has a singular cycle Jacobian'
        raise RuntimeError(msg)
    return step
