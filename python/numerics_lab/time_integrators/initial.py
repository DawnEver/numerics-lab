"""Consistent initial conditions for an index-1 DAE ``R(u, u_dot, t) = 0`` -- physics-blind.

A march started from an inconsistent state takes its first step to repair it, and that step is a
lie about the physics: an algebraic unknown jumps, the first rate is wrong, and a trapezoid seeded
with it rings. The consistent start solves, at ``t0``, for the ALGEBRAIC unknowns and the
DIFFERENTIAL rates with the differential values held -- IDA's ``YA_YDP_INIT``. Which unknown is which
is read off the problem itself: a column of ``dR/du_dot`` that is identically zero is algebraic.

THE RULE for a conflict, stated once: the differential values are the state the caller declared and
are never moved; the algebraic ones are derived. A caller who PINS an algebraic unknown (``fixed``)
to a value the constraints do not allow gets a :class:`InconsistentInitialCondition` naming it --
never a silent override of either side. A system whose IC Jacobian is singular (a differential
unknown fixed by a constraint, index > 1: a capacitor across an ideal source) RAISES with its
residual: there is no index-1 consistent start to return.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from numerics_lab.formulation import DerivativeWeights

__all__ = ['ConsistentStart', 'InconsistentInitialCondition', 'consistent_initial_condition']


class InconsistentInitialCondition(ValueError):
    """The declared start cannot be made consistent under the rule; carries the residual reached."""

    def __init__(self, message: str, residual: float) -> None:
        super().__init__(f'{message} (residual {residual:.3e})')
        self.residual = residual


@dataclass(frozen=True)
class ConsistentStart:
    """``u`` and ``u_dot`` satisfying ``R = 0`` at ``t0``; ``adjusted`` names the algebraic unknowns moved."""

    u: NDArray[np.floating]
    u_dot: NDArray[np.floating]
    algebraic: tuple[int, ...]
    adjusted: tuple[int, ...]
    residual: float


def _dense(matrix) -> NDArray[np.floating]:
    m = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix, dtype=float)
    return np.atleast_2d(m)


def consistent_initial_condition(
    problem,
    u0: NDArray[np.floating],
    t0: float = 0.0,
    *,
    fixed: Sequence[int] = (),
    tol: float = 1e-10,
    max_newton: int = 30,
) -> ConsistentStart:
    """Solve for the algebraic unknowns and differential rates at ``t0`` (differential values held).

    ``tol`` is relative to the residual's scale ``max(|R(u0, 0)|, 1)``. ``fixed`` lists algebraic
    unknowns the caller insists on; their declared value must already be consistent.
    """
    u = np.asarray(u0, dtype=float).copy()
    n = u.size
    zero = np.zeros(n)
    jm = _dense(problem.jacobian(u, zero, zero, t0, DerivativeWeights(alpha0=0.0, alpha1=1.0)))
    differential = np.flatnonzero(np.any(jm != 0.0, axis=0))
    algebraic = np.setdiff1d(np.arange(n), differential)
    u_dot = np.zeros(n)
    scale = max(float(np.max(np.abs(problem.residual(u, u_dot, zero, t0)))), 1.0)
    residual = np.inf
    for _ in range(max_newton):
        r = np.asarray(problem.residual(u, u_dot, zero, t0), dtype=float)
        residual = float(np.max(np.abs(r))) if r.size else 0.0
        if residual <= tol * scale:
            break
        jk = _dense(problem.jacobian(u, u_dot, zero, t0, DerivativeWeights(alpha0=1.0)))
        jm = _dense(problem.jacobian(u, u_dot, zero, t0, DerivativeWeights(alpha0=0.0, alpha1=1.0)))
        block = np.hstack([jk[:, algebraic], jm[:, differential]])
        if np.linalg.matrix_rank(block) < n:
            msg = (
                'the initial-condition system is singular: a differential unknown is fixed by an '
                'algebraic constraint (index > 1), so no index-1 consistent start exists'
            )
            raise InconsistentInitialCondition(msg, residual)
        step = np.linalg.solve(block, -r)
        u[algebraic] += step[: algebraic.size]
        u_dot[differential] += step[algebraic.size :]
    else:
        msg = f'no consistent start within {max_newton} Newton iterations'
        raise InconsistentInitialCondition(msg, residual)
    declared = np.asarray(u0, dtype=float)
    moved = tuple(int(i) for i in algebraic if abs(u[i] - declared[i]) > tol * max(1.0, abs(declared[i])))
    pinned = sorted(set(moved) & {int(i) for i in fixed})
    if pinned:
        msg = (
            f'algebraic unknown(s) {pinned} were pinned at {declared[pinned].tolist()} but the '
            f'constraints require {u[pinned].tolist()}'
        )
        raise InconsistentInitialCondition(msg, float(np.max(np.abs(u[pinned] - declared[pinned]))))
    return ConsistentStart(
        u=u, u_dot=u_dot, algebraic=tuple(int(i) for i in algebraic), adjusted=moved, residual=residual
    )
