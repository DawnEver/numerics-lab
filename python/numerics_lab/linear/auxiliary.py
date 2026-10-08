"""Additive auxiliary-space inverse: diagonal vector solve plus a factored scalar correction."""

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator, splu

__all__ = ['additive_auxiliary_preconditioner']


def additive_auxiliary_preconditioner(matrix, transfer, scalar_matrix, *, scale):
    """Apply diag(A)^-1 + scale G S^-1 G.T; transpose deliberately is not conjugated."""
    matrix, transfer, scalar = csr_matrix(matrix), csr_matrix(transfer), csc_matrix(scalar_matrix)
    if (
        matrix.shape[0] != matrix.shape[1]
        or transfer.shape[0] != matrix.shape[0]
        or scalar.shape != (transfer.shape[1], transfer.shape[1])
    ):
        msg = 'auxiliary preconditioner requires compatible square operator and transfer shapes'
        raise ValueError(msg)
    diagonal = matrix.diagonal()
    if (
        not np.all(np.isfinite(matrix.data))
        or not np.all(np.isfinite(diagonal))
        or np.any(diagonal == 0)
        or not np.all(np.isfinite(transfer.data))
        or not np.all(np.isfinite(scalar.data))
        or not np.isfinite(scale)
    ):
        msg = 'auxiliary preconditioner requires finite factors and nonzero vector diagonal'
        raise ValueError(msg)
    dtype = np.result_type(matrix.dtype, transfer.dtype, scalar.dtype, scale)
    factor = splu(scalar.astype(dtype)) if scalar.shape[0] else None

    def apply(value):
        value = np.asarray(value, dtype=dtype)
        result = value / diagonal
        if factor is not None:
            result += scale * (transfer @ factor.solve(transfer.T @ value))
        return result

    return LinearOperator(matrix.shape, matvec=apply, dtype=dtype)
