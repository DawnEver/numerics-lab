"""Performed sparse row arithmetic, with explicit unavailable-row provenance."""

import numpy as np
from scipy.sparse import csr_matrix

from motronics.contracts.runtime.ports import ResidualAssemblyBound

__all__ = ['affine_residual_bound', 'matrix_vector_roundoff', 'transport_residual_bound']


def _row_arithmetic(matrix, vector, offset):
    matrix = csr_matrix(matrix)
    vector, offset = np.asarray(vector), np.asarray(offset)
    if vector.shape != (matrix.shape[1],) or offset.shape != (matrix.shape[0],):
        msg = 'sparse arithmetic operands require aligned actual row and column shapes'
        raise ValueError(msg)
    entries = matrix.tocoo()
    products = entries.data * vector[entries.col]
    known = np.ones(matrix.shape[0], dtype=bool)
    unsupported = (
        ~np.isfinite(products)
        | ((entries.data != 0) & (vector[entries.col] != 0) & (products == 0))
        | ((products != 0) & (abs(products) < np.finfo(float).tiny))
    )
    known[entries.row[unsupported]] = False
    magnitude = abs(offset) + np.asarray(abs(matrix) @ abs(vector)).ravel()
    counts = np.bincount(entries.row, minlength=matrix.shape[0])
    operations = np.maximum(2 * counts - 1, 0) + 1
    gamma = operations * (np.finfo(float).eps / 2)
    if not np.isfinite(magnitude).all() or np.any(gamma >= 0.5):
        msg = 'sparse row arithmetic requires finite representable operands and operation counts'
        raise ValueError(msg)
    gamma /= 1 - gamma
    values = gamma * magnitude / (1 - gamma)
    known[(magnitude != 0) & (values < np.finfo(float).tiny)] = False
    return matrix, values, known, gamma


def matrix_vector_roundoff(matrix, vector, offset):
    """Gamma bound for the actual multiply, row sum and offset addition."""
    _matrix, values, known, _gamma = _row_arithmetic(matrix, vector, offset)
    values[~known] = 0.0
    return ResidualAssemblyBound(values, known)


def affine_residual_bound(matrix, vector, residual, bound):
    """Actual matrix-vector addition to an already evaluated residual."""
    _matrix, values, known, gamma = _row_arithmetic(matrix, vector, residual)
    known &= bound.known
    values += bound.values / (1 - gamma)
    values[~known] = 0.0
    return ResidualAssemblyBound(values, known)


def transport_residual_bound(matrix, residual, bound):
    """Transport only rows with entirely certified nonzero input support."""
    matrix, values, known, gamma = _row_arithmetic(matrix, residual, np.zeros(matrix.shape[0]))
    support = matrix.copy()
    support.eliminate_zeros()
    known &= np.asarray(abs(support) @ (~bound.known).astype(float)).ravel() == 0
    propagated = np.asarray(abs(matrix) @ bound.values).ravel()
    entries = matrix.tocoo()
    products = abs(entries.data) * bound.values[entries.col]
    underflow = (entries.data != 0) & (bound.values[entries.col] != 0) & (products == 0)
    known[entries.row[underflow]] = False
    values = (values + propagated) / (1 - gamma)
    known[(propagated != 0) & (propagated < np.finfo(float).tiny)] = False
    values[~known] = 0.0
    return ResidualAssemblyBound(values, known)
