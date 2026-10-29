"""Smoothed-aggregation algebraic multigrid for real SPD nodal operators.

This is not JMAG's 3D A-phi electromagnetic-decoupling algorithm. Its supported
surface is an actual Galerkin V-cycle, selectable by the existing FEA/MEC seam.
No mesh coordinates or optional package are needed; coarse spaces come from
strong matrix couplings and a constant tentative near-null vector.
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator

__all__ = ['amg_preconditioner']


def _aggregates(matrix):
    """Greedy strongest-neighbour pairs; unmatched rows form singleton aggregates."""
    size = matrix.shape[0]
    labels = np.full(size, -1, dtype=int)
    count = 0
    for row in range(size):
        if labels[row] >= 0:
            continue
        labels[row] = count
        start, end = matrix.indptr[row : row + 2]
        columns = matrix.indices[start:end]
        values = matrix.data[start:end]
        available = (columns != row) & (labels[columns] < 0) & (values != 0)
        if np.any(available):
            neighbour = columns[available][np.argmax(np.abs(values[available]))]
            labels[neighbour] = count
        count += 1
    return sp.csr_matrix((np.ones(size), (np.arange(size), labels)), shape=(size, count))


def amg_preconditioner(matrix) -> LinearOperator:
    """Symmetric pre/post Jacobi V-cycle with smoothed Galerkin coarse spaces.

    Assumes SPD, as CG does; refuses non-real, nonsymmetric and nonpositive
    diagonals. Coarsest Cholesky additionally refuses indefinite/singular coarse
    operators. The weighted smoother uses a Gershgorin spectral upper bound,
    not a guessed omega that might destabilize a strongly coupled operator.
    """
    matrix = sp.csr_matrix(matrix, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if (
        matrix.shape[0] != matrix.shape[1]
        or not matrix.shape[0]
        or np.iscomplexobj(matrix.data)
        or not np.all(np.isfinite(matrix.data))
    ):
        msg = 'AMG requires a finite real nonempty square SPD operator'
        raise ValueError(msg)
    difference = (matrix - matrix.T).tocoo()
    if difference.nnz:
        scale = np.maximum(
            np.abs(np.asarray(matrix[difference.row, difference.col]).ravel()),
            np.abs(np.asarray(matrix[difference.col, difference.row]).ravel()),
        )
        if np.any(np.abs(difference.data) > 64 * np.finfo(float).eps * scale):
            msg = 'AMG requires a symmetric SPD operator'
            raise ValueError(msg)
        matrix = (0.5 * matrix + 0.5 * matrix.T).tocsr()
    levels = []
    operators = []
    while True:
        operators.append(matrix)
        diagonal = matrix.diagonal()
        if np.any(diagonal <= 0):
            msg = 'AMG requires positive diagonals on every level'
            raise ValueError(msg)
        if matrix.shape[0] <= 32:
            break
        tentative = _aggregates(matrix)
        if tentative.shape[1] == matrix.shape[0]:
            break  # disconnected diagonal system: exact diagonal solve below
        upper = float(np.max(np.asarray(abs(matrix).sum(axis=1)).ravel() / diagonal))
        # Strict damping keeps I-B A nonsingular even at the spectral bound,
        # so smoothing cannot erase a tentative coarse column.
        inverse = (2.0 / 3.0) / (upper * diagonal)
        prolongation = (tentative - sp.diags(inverse) @ matrix @ tentative).tocsr()
        levels.append((matrix, inverse, prolongation))
        matrix = (prolongation.T @ matrix @ prolongation).tocsr()
    if matrix.shape[0] > 32:

        def coarse_solve(rhs):
            return rhs / diagonal

    else:
        try:
            factor = cho_factor(matrix.toarray())
        except np.linalg.LinAlgError as exc:
            msg = 'AMG coarse operator is not positive definite'
            raise ValueError(msg) from exc

        def coarse_solve(rhs):
            return cho_solve(factor, rhs)

    def cycle(level, rhs):
        if level == len(levels):
            return coarse_solve(rhs)
        operator, inverse, prolongation = levels[level]
        estimate = inverse * rhs
        residual = rhs - operator @ estimate
        estimate += prolongation @ cycle(level + 1, prolongation.T @ residual)
        return estimate + inverse * (rhs - operator @ estimate)

    result = LinearOperator(
        operators[0].shape,
        matvec=lambda rhs: cycle(0, np.asarray(rhs).ravel()),
        rmatvec=lambda rhs: cycle(0, np.asarray(rhs).ravel()),
        dtype=float,
    )
    result.level_sizes = tuple(operator.shape[0] for operator in operators)
    return result
