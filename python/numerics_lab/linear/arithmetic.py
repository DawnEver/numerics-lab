"""Performed sparse row arithmetic, with explicit unavailable-row provenance."""

import math

import numpy as np
from scipy.sparse import csr_matrix

from numerics_lab.formulation import ResidualAssemblyBound

__all__ = [
    'affine_residual_bound',
    'certified_sparse_product',
    'matrix_vector_roundoff',
    'rounded_defect',
    'sparse_product_bound',
    'transport_residual_bound',
]

#: The column count the exact (pure-Python) arithmetic here is priced for -- the same bound the
#: exact rate and border reconstructions already carry. Above it a caller keeps the working-precision
#: residual, whose last-ulp blindness costs nothing a field-size refinement can measure anyway.
EXACT_DEFECT_MAX_COLUMNS = 512


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


def sparse_product_bound(left, right):
    """``(left @ right, gamma_k * (|left| @ |right|))``: the product and its own summation bound."""
    left, right = csr_matrix(left), csr_matrix(right)
    product = csr_matrix(left @ right)
    magnitude = csr_matrix(abs(left) @ abs(right))
    counts = csr_matrix((left != 0).astype(float) @ (right != 0).astype(float))
    product.sort_indices()
    for matrix in (magnitude, counts):
        matrix.sort_indices()
    gamma = counts.multiply(np.finfo(float).eps).tocsr()
    gamma.data = gamma.data / (1 - gamma.data)
    return product, csr_matrix(magnitude.multiply(gamma))


def certified_sparse_product(left, right):
    """``left @ right`` without coefficients its own summation cannot certify.

    An entry no larger than the gamma bound of its own dot product,
    ``gamma_k * (|left| @ |right|)``, has neither sign nor magnitude
    authority: an exact cancellation (a left-null storage row against a
    column in the range of the mass) lands there. Kept, it reads as a
    structural dependence and a sparse LU pivots on it.
    """
    product, bound = sparse_product_bound(left, right)
    # JUDGED ON THE PRODUCT'S OWN ENTRIES: a sparse `<=` is True at every implicit zero, and
    # materialised n^2 of them on a field-sized product (76 GiB, case_a, 2026-10-12).
    product.eliminate_zeros()
    rows = np.repeat(np.arange(product.shape[0]), np.diff(product.indptr))
    limit = np.asarray(bound[rows, product.indices]).ravel()
    product.data[np.abs(product.data) <= limit] = 0.0
    product.eliminate_zeros()
    return csr_matrix(product)


def rounded_defect(matrix, value, rhs):
    """``rhs - matrix @ value``, each row the CORRECTLY ROUNDED exact defect (up to underflow).

    A working-precision residual rounds the last ulp of an answer away: ``3 - (x + 2)`` is 0.0 for
    ``x = 1 - 2**-53``, so an iterative refinement fed it stops one ulp short of a representable
    exact solution. Each product is split exactly by ``fma`` and every term is summed by ``fsum``.
    """
    matrix = csr_matrix(matrix)
    value, rhs = np.asarray(value, dtype=float), np.asarray(rhs, dtype=float)
    if value.shape != (matrix.shape[1],) or rhs.shape != (matrix.shape[0],):
        msg = 'an exact defect requires aligned actual row and column shapes'
        raise ValueError(msg)
    defect = np.empty(matrix.shape[0])
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row : row + 2]
        terms = [float(rhs[row])]
        for column, entry in zip(matrix.indices[start:end], matrix.data[start:end], strict=True):
            product = float(entry) * float(value[column])
            terms += (-product, -math.fma(float(entry), float(value[column]), -product))
        # An overflowed product has no exact split; the row's defect is then not finite either way.
        defect[row] = math.fsum(terms) if all(map(math.isfinite, terms)) else math.nan
    return defect


def matrix_vector_roundoff(matrix, vector, offset):
    """Gamma bound for the actual multiply, row sum and offset addition."""
    _matrix, values, known, _gamma = _row_arithmetic(matrix, vector, offset)
    values[~known] = 0.0
    return ResidualAssemblyBound(values, known)


def affine_residual_bound(matrix, vector, residual, bound):
    """Actual matrix-vector addition to an already evaluated residual; its law term rides along unchanged."""
    _matrix, values, known, gamma = _row_arithmetic(matrix, vector, residual)
    known &= bound.known
    values += bound.values / (1 - gamma)
    values[~known] = 0.0
    return ResidualAssemblyBound(values, known, bound.law)


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
    return ResidualAssemblyBound(values, known, np.asarray(abs(matrix) @ bound.law).ravel())
