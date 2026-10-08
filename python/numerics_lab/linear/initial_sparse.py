"""Sparse initial pair over caller-proven independent storage/algebraic rows."""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.sparse.linalg import splu


@dataclass(frozen=True)
class SparseInitialPair:
    state: np.ndarray
    rate: np.ndarray
    rate_is_unique: np.ndarray


def sparse_initial_pair(
    jacobian, residual, mass, initial, *, algebraic_rows, stored_rows, partial_time, tolerance=1e-10
):
    """Solve actual index-two equations without dense field-sized nullspaces.

    The caller owns the rank decomposition: ``stored_rows`` selects independent
    rows of mass and ``algebraic_rows`` spans its left nullspace. Free algebraic
    rates are marked, not presented as physical zero derivatives. No free state
    or stored rate is selected by a least-squares/minimum-norm convention.
    """
    initial = np.asarray(initial, dtype=float)
    residual, partial_time = np.asarray(residual, dtype=float), np.asarray(partial_time, dtype=float)
    jacobian, mass, algebraic = map(sp.csr_matrix, (jacobian, mass, algebraic_rows))
    n = initial.size
    stored_rows = np.asarray(stored_rows)
    if (
        initial.shape != (n,)
        or residual.shape != (n,)
        or partial_time.shape != (n,)
        or jacobian.shape != (n, n)
        or mass.shape != (n, n)
        or algebraic.shape[1] != n
        or stored_rows.ndim != 1
        or stored_rows.dtype.kind not in 'iu'
        or len(set(stored_rows.tolist())) != len(stored_rows)
        or np.any(stored_rows < 0)
        or np.any(stored_rows >= n)
        or len(stored_rows) + algebraic.shape[0] != n
        or not all(
            np.isfinite(value).all()
            for value in (initial, residual, partial_time, jacobian.data, mass.data, algebraic.data)
        )
        or not np.isfinite(tolerance)
        or tolerance <= 0
    ):
        msg = 'sparse initial pair requires finite actual matrices and an explicit complete storage rank decomposition'
        raise ValueError(msg)
    defect = algebraic @ mass
    if np.linalg.norm(defect.data) > tolerance * np.linalg.norm(algebraic.data) * np.linalg.norm(mass.data):
        msg = 'declared algebraic rows are not in the actual mass left nullspace'
        raise ValueError(msg)
    stored = mass[stored_rows]
    matrix = sp.bmat(
        [
            [jacobian, mass],
            [stored, sp.csr_matrix(stored.shape)],
            [sp.csr_matrix(algebraic.shape), algebraic @ jacobian],
        ],
        format='csr',
    )
    rhs = np.concatenate((-residual, np.zeros(len(stored_rows)), -algebraic @ partial_time))
    matrix.eliminate_zeros()
    scale = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    scale[scale == 0] = 1.0
    matrix, rhs = sp.diags(1 / scale) @ matrix, rhs / scale
    # Only structurally zero rate columns may be omitted. Their value never
    # affects these equations; expose that fact to outgoing-mode consumers.
    nonzero = np.asarray(matrix.getnnz(axis=0)).ravel() != 0
    if not nonzero[:n].all():
        msg = 'initial state requires additional physical authority'
        raise ValueError(msg)
    selected = np.flatnonzero(nonzero)
    reduced = matrix[:, selected].tocsr()
    matched_rows = maximum_bipartite_matching(reduced, perm_type='row')
    if np.any(matched_rows < 0):
        msg = 'initial state/rate equations require additional physical authority'
        raise ValueError(msg)
    try:
        factor = splu(reduced[matched_rows].tocsc())
    except RuntimeError as error:
        msg = 'initial state/rate equations lack a nonsingular sparse physical constraint subsystem'
        raise ValueError(msg) from error
    pair = np.zeros(2 * n)
    pair[selected] = factor.solve(rhs[matched_rows])
    bar = tolerance * (np.linalg.norm(matrix.data) * np.linalg.norm(pair) + np.linalg.norm(rhs))
    if np.linalg.norm(matrix @ pair - rhs) > bar:
        msg = 'stored initial state contradicts actual state/rate constraints'
        raise ValueError(msg)
    # Validate every original storage row, not just the rank representative.
    storage_defect = mass @ pair[:n]
    if np.linalg.norm(storage_defect) > tolerance * np.linalg.norm(mass.data) * np.linalg.norm(pair[:n]):
        msg = 'declared storage rank decomposition does not preserve actual stored state'
        raise ValueError(msg)
    return SparseInitialPair(initial + pair[:n], pair[n:], nonzero[n:].copy())
