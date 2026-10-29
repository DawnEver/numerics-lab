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
    if n == 0:
        return ConsistentStart(u=u, u_dot=zero, algebraic=(), adjusted=(), residual=0.0)
    jm = _dense(problem.jacobian(u, zero, zero, t0, DerivativeWeights(alpha0=0.0, alpha1=1.0)))
    differential = np.flatnonzero(np.any(jm != 0.0, axis=0))
    algebraic = np.setdiff1d(np.arange(n), differential)
    pinned = np.intersect1d(algebraic, np.asarray(fixed, dtype=int))
    # A PINNED value is the caller's: it is held OUT of the solve, never moved and then compared.
    free = np.setdiff1d(algebraic, pinned)
    u_dot = np.zeros(n)
    scale = max(float(np.max(np.abs(problem.residual(u, u_dot, zero, t0)))), 1.0)

    def residual_of() -> tuple[NDArray[np.floating], float]:
        r = np.asarray(problem.residual(u, u_dot, zero, t0), dtype=float)
        return r, float(np.max(np.abs(r))) if r.size else 0.0

    r, residual = residual_of()
    for _ in range(max_newton):
        # A START INSIDE THE TOLERANCE IS STILL SOLVED ONCE: `tol` says when to stop iterating, not
        # that the declared algebraic values and zero rates are the answer. Returned untouched, a
        # settled sampled loop's held port voltages kept a 1e-10-sized stale value and zero rates at
        # every hit (case_q tracking 5.4e-15 -> 4.4e-13). So the residual is read AFTER each step.
        jk = _dense(problem.jacobian(u, u_dot, zero, t0, DerivativeWeights(alpha0=1.0)))
        jm = _dense(problem.jacobian(u, u_dot, zero, t0, DerivativeWeights(alpha0=0.0, alpha1=1.0)))
        block = np.hstack([jk[:, free], jm[:, differential]])
        if np.linalg.matrix_rank(block) < block.shape[1] or (pinned.size == 0 and block.shape[1] < n):
            if residual <= tol * scale:
                break  # already consistent; an index > 1 block has no index-1 correction to take
            msg = (
                'the initial-condition system is singular: a differential unknown is fixed by an '
                'algebraic constraint (index > 1), so no index-1 consistent start exists'
            )
            raise InconsistentInitialCondition(msg, residual)
        # Square without pins; with pins the rows outnumber the unknowns, and a consistent pin leaves
        # them a consistent (zero-residual) least-squares system.
        step = np.linalg.lstsq(block, -r, rcond=None)[0]
        u[free] += step[: free.size]
        u_dot[differential] += step[free.size :]
        r, residual = residual_of()
        if residual <= tol * scale:
            break
    else:
        if pinned.size:
            msg = (
                f'algebraic unknown(s) {pinned.tolist()} were pinned at {u[pinned].tolist()} but the '
                f'constraints cannot be met there'
            )
            raise InconsistentInitialCondition(msg, residual)
        msg = f'no consistent start within {max_newton} Newton iterations'
        raise InconsistentInitialCondition(msg, residual)
    declared = np.asarray(u0, dtype=float)
    moved = tuple(int(i) for i in free if abs(u[i] - declared[i]) > tol * max(1.0, abs(declared[i])))
    return ConsistentStart(
        u=u, u_dot=u_dot, algebraic=tuple(int(i) for i in algebraic), adjusted=moved, residual=residual
    )
