"""Storage-preserving sparse Newton projection over actual DAE operators."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from numerics_lab.linear.initial_sparse import projected_rate_consistent, sparse_initial_pair

__all__ = ['constant_storage_initial_pair', 'constant_storage_right_state']


def constant_storage_right_state(
    initial, *, residual_at, jacobian_at, mass_at, storage_rows, max_newton=30, tolerance=1e-10
):
    """Unique right algebraic state, without inventing a continuous outgoing rate.

    Preserve the actual stored coordinates at a law change. This needs no
    explicit-time derivative: moving geometry is evaluated at the same physical
    instant, never at a perturbed clock. Operators must leave accepted history
    unchanged throughout the projection.
    """
    initial = np.asarray(initial, dtype=float)
    if (
        initial.ndim != 1
        or not np.isfinite(initial).all()
        or max_newton < 1
        or not np.isfinite(tolerance)
        or tolerance <= 0
    ):
        msg = 'right projection requires finite state and positive numerical controls'
        raise ValueError(msg)
    mass = sp.csr_matrix(mass_at(initial))
    algebraic = storage_rows.algebraic_rows
    stored = mass[storage_rows.stored_rows]
    state = initial.copy()
    for _ in range(max_newton):
        change = sp.csr_matrix(mass_at(state)) - mass
        change.eliminate_zeros()
        if change.nnz:
            msg = 'state-dependent storage requires its actual finite stored-coordinate map'
            raise ValueError(msg)
        residual = np.asarray(residual_at(state))
        jacobian = sp.csr_matrix(jacobian_at(state))
        matrix = sp.vstack((algebraic @ jacobian, stored), format='csr')
        rhs = np.concatenate((-algebraic @ residual, -stored @ (state - initial)))
        row_norm = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
        row_norm[row_norm == 0] = 1.0
        matrix, rhs = sp.diags(1 / row_norm) @ matrix, rhs / row_norm
        try:
            correction = splu(matrix.tocsc()).solve(rhs)
        except RuntimeError as error:
            msg = 'accepted right state needs additional physical algebraic/storage authority'
            raise ValueError(msg) from error
        state += correction
        change = sp.csr_matrix(mass_at(state)) - mass
        change.eliminate_zeros()
        if change.nnz:
            msg = 'state-dependent storage requires its actual finite stored-coordinate map'
            raise ValueError(msg)
        actual = np.asarray(algebraic @ residual_at(state))
        scale = np.linalg.norm((algebraic @ jacobian).data) * np.linalg.norm(state) + np.linalg.norm(
            algebraic @ residual
        )
        storage_defect = mass @ (state - initial)
        storage_scale = np.linalg.norm(mass.data) * (np.linalg.norm(state) + np.linalg.norm(initial))
        if np.linalg.norm(actual) <= tolerance * scale and np.linalg.norm(storage_defect) <= tolerance * storage_scale:
            if not np.isfinite(state).all():
                msg = 'accepted right projection produced a nonfinite state'
                raise ValueError(msg)
            return state
    msg = 'actual stored-coordinate-preserving right projection did not converge'
    raise ValueError(msg)


def constant_storage_initial_pair(
    initial, *, residual_at, jacobian_at, mass_at, partial_time_at, storage_rows, max_newton=30, tolerance=1e-10
):
    """Project state/rate without advancing history or guessing storage coordinates.

    Operator callbacks must be pure evaluations of one frozen physical law.
    A changing mass is not a finite stored-state map and is refused. Actual
    algebraic equations and their outgoing derivatives are checked at the final
    nonlinear state, not merely at the preceding Newton linearization.
    """
    initial = np.asarray(initial, dtype=float)
    if max_newton < 1 or not np.isfinite(tolerance) or tolerance <= 0:
        msg = 'sparse initial projection requires positive iteration and consistency tolerances'
        raise ValueError(msg)
    mass = sp.csr_matrix(mass_at(initial))
    state = initial.copy()
    zero = np.zeros_like(initial)
    for _ in range(max_newton):
        difference = sp.csr_matrix(mass_at(state)) - mass
        difference.eliminate_zeros()
        if difference.nnz:
            msg = 'state-dependent storage requires its actual finite stored-coordinate map'
            raise ValueError(msg)
        jacobian = sp.csr_matrix(jacobian_at(state))
        residual = np.asarray(residual_at(state, zero))
        partial = np.asarray(partial_time_at(state))
        pair = sparse_initial_pair(
            jacobian,
            residual,
            mass,
            state,
            algebraic_rows=storage_rows.algebraic_rows,
            stored_rows=storage_rows.stored_rows,
            partial_time=partial,
            tolerance=tolerance,
        )
        state = pair.state
        difference = sp.csr_matrix(mass_at(state)) - mass
        difference.eliminate_zeros()
        if difference.nnz:
            msg = 'state-dependent storage requires its actual finite stored-coordinate map'
            raise ValueError(msg)
        actual = np.asarray(residual_at(state, pair.rate))
        updated_jacobian = sp.csr_matrix(jacobian_at(state))
        updated_partial = np.asarray(partial_time_at(state))
        scale = (
            np.linalg.norm(residual)
            + np.linalg.norm(jacobian.data) * np.linalg.norm(state)
            + np.linalg.norm(mass @ pair.rate)
        )
        if np.linalg.norm(actual) <= tolerance * scale and projected_rate_consistent(
            updated_jacobian,
            pair.rate,
            updated_partial,
            algebraic_rows=storage_rows.algebraic_rows,
            tolerance=tolerance,
        ):
            return pair
    msg = 'actual storage-preserving state/rate projection did not converge'
    raise ValueError(msg)
