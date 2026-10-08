"""Storage-preserving sparse Newton projection over actual DAE operators."""

import numpy as np
import scipy.sparse as sp

from motronics.gauss.linear.initial_sparse import sparse_initial_pair


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
        rate_defect = storage_rows.algebraic_rows @ (updated_jacobian @ pair.rate + updated_partial)
        scale = (
            np.linalg.norm(residual)
            + np.linalg.norm(jacobian.data) * np.linalg.norm(state)
            + np.linalg.norm(mass @ pair.rate)
        )
        rate_scale = np.linalg.norm(updated_jacobian.data) * np.linalg.norm(pair.rate) + np.linalg.norm(updated_partial)
        if np.linalg.norm(actual) <= tolerance * scale and np.linalg.norm(rate_defect) <= tolerance * rate_scale:
            return pair
    msg = 'actual storage-preserving state/rate projection did not converge'
    raise ValueError(msg)
