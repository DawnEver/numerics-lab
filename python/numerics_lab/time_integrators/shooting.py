"""Matrix-free periodic shooting on a caller-supplied, stateless cycle map."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from operator import index

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import NoConvergence, newton_krylov

__all__ = ['ShootingResult', 'periodic_shooting']


@dataclass(frozen=True)
class ShootingResult:
    state: NDArray[np.float64]
    residual_norm: float
    cycle_evaluations: int


def periodic_shooting(
    cycle_map: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    initial: NDArray[np.float64],
    *,
    atol: float = 1e-9,
    rtol: float = 1e-7,
    max_iterations: int = 30,
    sign: int = 1,
) -> ShootingResult:
    """Solve ``Phi(u)-sign*u=0`` without a dense monodromy matrix.

    The map must start a fresh cycle on every call; mutable material or integration
    history is part of the state and must not leak between trial evaluations.
    Failure raises rather than publishing a nonperiodic seed as steady state.
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

    try:
        state = np.asarray(newton_krylov(residual, seed.copy(), f_tol=atol, maxiter=max_iterations), dtype=float)
    except (NoConvergence, np.linalg.LinAlgError) as exc:
        msg = 'periodic shooting failed to converge'
        raise RuntimeError(msg) from exc
    except ValueError as exc:
        if 'Jacobian inversion yielded zero vector' not in str(exc):
            raise
        msg = 'periodic shooting has a singular cycle Jacobian'
        raise RuntimeError(msg) from exc
    defect = residual(state)
    norm = float(np.linalg.norm(defect, ord=np.inf))
    if norm > atol + rtol * float(np.linalg.norm(state, ord=np.inf)):
        msg = 'periodic shooting failed the final cycle-closure tolerance'
        raise RuntimeError(msg)
    return ShootingResult(state.copy(), norm, evaluations)
