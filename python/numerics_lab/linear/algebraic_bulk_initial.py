"""Sparse algebraic bulk elimination with rank revelation confined to its declared border."""

from numbers import Integral

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from numerics_lab.linear.border_rank import border_affine_space
from numerics_lab.linear.initial_sparse import SparseInitialPair
from numerics_lab.linear.storage_border import storage_border_rows

__all__ = ['algebraic_bulk_initial_pair']


def algebraic_bulk_initial_pair(
    jacobian, residual, mass, initial, *, bulk_size, partial_time, stored_increment, tolerance, fixed_state=False
):
    """Eliminate proven storage-free bulk; never select a free physical state.

    The numerical representative of a free algebraic rate is not an initial
    condition: every affected physical component is explicitly marked unknown.
    Dense rank work is confined to the caller-declared small border.
    """
    jacobian, mass = sp.csr_matrix(jacobian), sp.csr_matrix(mass)
    initial, residual, partial_time = map(np.asarray, (initial, residual, partial_time))
    n = initial.size
    k = bulk_size
    if (
        isinstance(k, (bool, np.bool_))
        or not isinstance(k, Integral)
        or not 0 < k < n
        or initial.shape != (n,)
        or residual.shape != (n,)
        or partial_time.shape != (n,)
        or jacobian.shape != (n, n)
        or mass.shape != (n, n)
        or not all(np.isfinite(value).all() for value in (initial, residual, partial_time, jacobian.data, mass.data))
        or not np.isfinite(tolerance)
        or tolerance <= 0
        or n - k > 256
        or mass[:k].nnz
        or mass[:, :k].nnz
    ):
        msg = 'algebraic bulk requires explicit storage-free rows and columns and a bounded border'
        raise ValueError(msg)
    factor = splu(jacobian[:k, :k].tocsc())
    sensitivity = factor.solve(jacobian[:k, k:].toarray())
    residual_bulk = factor.solve(residual[:k])
    partial_bulk = factor.solve(partial_time[:k])
    coupling = jacobian[k:, :k]
    reduced_jacobian = jacobian[k:, k:].toarray() - coupling @ sensitivity
    reduced_residual = residual[k:] - coupling @ residual_bulk
    reduced_partial = partial_time[k:] - coupling @ partial_bulk
    reduced_mass = mass[k:, k:]
    m = n - k
    storage = storage_border_rows(reduced_mass, bulk_indices=np.array([], dtype=int), border_rows=np.arange(m))
    algebraic = storage.algebraic_rows.toarray()
    selected = storage.stored_rows
    matrix = np.block([
        [
            np.zeros((len(selected), m)) if fixed_state else reduced_jacobian,
            reduced_mass[selected].toarray() if fixed_state else reduced_mass.toarray(),
        ],
        [
            np.eye(m) if fixed_state else reduced_mass[selected].toarray(),
            np.zeros((m if fixed_state else len(selected), m)),
        ],
        [np.zeros((len(algebraic), m)), algebraic @ reduced_jacobian],
    ])
    change = np.asarray(stored_increment)
    if change.shape != (len(selected),) or not np.isfinite(change).all():
        msg = 'algebraic bulk must retain every actual independent border storage increment'
        raise ValueError(msg)
    if not isinstance(fixed_state, (bool, np.bool_)) or (fixed_state and np.any(change != 0)):
        msg = 'rate-only projection requires a fixed admitted state without a pending storage increment'
        raise ValueError(msg)
    rhs = np.concatenate((
        -residual[k:][selected] if fixed_state else -reduced_residual,
        np.zeros(m) if fixed_state else change,
        -algebraic @ reduced_partial,
    ))
    evaluation_roundoff = np.zeros(len(rhs))
    physical_evaluation = np.abs(jacobian) @ np.abs(initial) + np.abs(residual)
    # A sparse row dot product followed by addition of its residual has this
    # actual operation-count bound. Initial-state cancellation is arithmetic
    # uncertainty, not permission to relax a finite storage increment.
    eps = np.finfo(float).eps
    physical_operations = np.diff(jacobian.indptr) + 1
    physical_gamma = physical_operations * eps / (1 - physical_operations * eps)
    physical_roundoff = physical_gamma * physical_evaluation
    if fixed_state:
        # The admitted state can retain one ULP of dynamic residual after
        # subtracting real physical terms. Its evaluation uncertainty is not
        # the magnitude of that cancellation residual itself. Derived rate
        # constraints and exact state/storage anchors receive NO such bound.
        evaluation_roundoff[: len(selected)] = physical_roundoff[k:][selected]
        order = np.concatenate((
            np.arange(len(selected), len(selected) + m),
            np.arange(len(selected) + m, len(rhs)),
            np.arange(len(selected)),
        ))
        matrix, rhs, evaluation_roundoff = matrix[order], rhs[order], evaluation_roundoff[order]
    else:
        # Propagate each original residual equation's evaluation bound through
        # its actual Schur row operator. This bound belongs neither to stored
        # increments nor to differentiated rate equations.
        row_projection = factor.solve(coupling.T.toarray(), trans='T').T
        evaluation_scale = physical_evaluation[k:] + np.abs(row_projection) @ physical_evaluation[:k]
        projection_operations = np.count_nonzero(row_projection, axis=1) + 1
        projection_gamma = projection_operations * eps / (1 - projection_operations * eps)
        evaluation_roundoff[:m] = (
            physical_roundoff[k:] + np.abs(row_projection) @ physical_roundoff[:k] + projection_gamma * evaluation_scale
        )
        # Preserve small algebraic state corrections (including actual linkage)
        # before eliminating dynamic rows with potentially enormous rates.
        # Otherwise a stiff derivative can lose a finite-Q correction in the
        # arithmetic of an unrelated current/rate equation.
        algebraic_indices = np.flatnonzero(np.asarray(reduced_mass.getnnz(axis=1)).ravel() == 0)
        dynamic_indices = np.setdiff1d(np.arange(m), algebraic_indices)
        order = np.concatenate((
            np.arange(m, m + len(selected)),
            algebraic_indices,
            np.arange(m + len(selected), len(rhs)),
            dynamic_indices,
        ))
        matrix, rhs, evaluation_roundoff = matrix[order], rhs[order], evaluation_roundoff[order]
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1
    matrix, rhs = matrix / norms[:, None], rhs / norms
    evaluation_roundoff /= norms
    # Rates and stored coordinates need not share units or magnitude. Rank
    # revelation uses dimensionless columns; otherwise a real small inductance
    # makes an algebraic rate column appear numerically absent.
    column_norms = np.linalg.norm(matrix, axis=0)
    column_norms[column_norms == 0] = 1
    equilibrated = matrix / column_norms
    answer_scaled, free_scaled = border_affine_space(
        equilibrated,
        rhs,
        mandatory_rows=np.arange(m if fixed_state else len(selected)),
    )
    rank_bar = np.finfo(float).eps * max(matrix.shape) * 16
    if np.any(free_scaled[:m] != 0):
        msg = 'actual border initial state requires additional physical authority'
        raise ValueError(msg)
    answer = answer_scaled / column_norms
    free = free_scaled / column_norms[:, None]
    defect = matrix @ answer - rhs
    scale = np.abs(matrix) @ np.abs(answer) + np.abs(rhs)
    if np.any(np.abs(defect) > tolerance * scale + evaluation_roundoff):
        msg = 'actual border stored state contradicts its state/rate equations'
        raise ValueError(msg)
    state = (
        initial.copy()
        if fixed_state
        else initial + np.concatenate((-residual_bulk - sensitivity @ answer[:m], answer[:m]))
    )
    rate = np.concatenate((-partial_bulk - sensitivity @ answer[m:], answer[m:]))
    rate_free = free[m:]
    border_unique = np.all(free_scaled[m:] == 0, axis=1)
    bulk_free = sensitivity @ rate_free
    bulk_scale = np.abs(sensitivity) @ np.abs(rate_free)
    bulk_unique = np.all(np.abs(bulk_free) <= rank_bar * bulk_scale, axis=1)
    return SparseInitialPair(state, rate, np.concatenate((bulk_unique, border_unique)))
