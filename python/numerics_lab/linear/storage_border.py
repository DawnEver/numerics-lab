"""Sparse bulk plus declared small-border mass rank decomposition."""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.linalg import qr, solve, svd
from scipy.sparse.linalg import splu


@dataclass(frozen=True)
class StorageRows:
    stored_rows: np.ndarray
    algebraic_rows: sp.csr_matrix


def storage_border_rows(mass, *, bulk_indices, border_rows):
    """Rank a declared invertible bulk block and its remaining border rows.

    The caller identifies the actual bulk coordinates/rows and the small border;
    all other rows must be exactly algebraic. Bulk rows may depend only on bulk
    rates. Dense work is border-by-state, never state-by-state. Singular bulk
    factors are refused, not regularized into invented storage.
    """
    mass = sp.csr_matrix(mass)
    mass.eliminate_zeros()
    bulk, border = np.asarray(bulk_indices), np.asarray(border_rows)
    n = mass.shape[0]
    if (
        mass.shape != (n, n)
        or not np.isfinite(mass.data).all()
        or any(value.ndim != 1 or value.dtype.kind not in 'iu' for value in (bulk, border))
        or len(set(np.concatenate((bulk, border)).tolist())) != len(bulk) + len(border)
        or any(np.any(value < 0) or np.any(value >= n) for value in (bulk, border))
    ):
        msg = 'storage border requires finite square mass and disjoint explicit bulk/border rows'
        raise ValueError(msg)
    remaining = np.setdiff1d(np.arange(n), bulk)
    algebraic_indices = np.setdiff1d(remaining, border)
    if mass[algebraic_indices].nnz or mass[bulk][:, remaining].nnz:
        msg = 'declared storage bulk/border omits actual mass rows or bulk rate dependencies'
        raise ValueError(msg)
    # Dense allocation is only the explicitly declared border width.
    tail = mass[border][:, remaining].toarray()
    norms = np.linalg.norm(tail, axis=1)
    norms[norms == 0] = 1.0
    normalized = tail / norms[:, None]
    if len(border):
        singular = svd(normalized, compute_uv=False)
        rank_bar = np.finfo(float).eps * max(normalized.shape) * (singular[0] if singular.size else 0.0)
        rank = int(np.count_nonzero(singular > rank_bar))
        _, _, pivots = qr(normalized.T, mode='economic', pivoting=True)
        independent = pivots[:rank]
        independent_border = border[independent]
        dependent = np.setdiff1d(np.arange(len(border)), independent)
        null_border = np.zeros((len(dependent), len(border)))
        null_border[np.arange(len(dependent)), dependent] = 1.0
        if rank and len(dependent):
            # Explicit actual row dependencies, not a dense SVD rotation of
            # disconnected blocks. SVD roundoff otherwise invents nonzero rate
            # columns through null-vector noise in an unrelated storage row.
            basis = normalized[independent]
            _, _, columns = qr(basis, mode='economic', pivoting=True)
            minor = columns[:rank]
            dependency = solve(basis[:, minor].T, normalized[dependent][:, minor].T).T
            null_border[:, independent] = -dependency
        null_border /= norms[None, :]
    else:
        independent_border = np.array([], dtype=int)
        null_border = np.zeros((0, 0))
    if len(bulk):
        try:
            factor = splu(mass[bulk][:, bulk].tocsc())
        except RuntimeError as error:
            msg = 'declared storage bulk is not numerically invertible'
            raise ValueError(msg) from error
        # E D^-1: transpose solve keeps the only dense dimension the border.
        correction = factor.solve(mass[border][:, bulk].T.toarray(), trans='T').T
    else:
        correction = np.zeros((len(border), 0))
    selectors = sp.eye(n, format='csr')
    algebraic = sp.vstack(
        [
            selectors[algebraic_indices],
            sp.csr_matrix(null_border) @ selectors[border] - sp.csr_matrix(null_border @ correction) @ selectors[bulk],
        ],
        format='csr',
    )
    return StorageRows(np.concatenate((bulk, independent_border)), algebraic)
