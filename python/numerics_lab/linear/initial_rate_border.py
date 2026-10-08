"""Sparse stored-acceleration elimination with rank revelation only at its small border."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from motronics.gauss.linear.border_rank import border_affine_space


def complete_rate_border(
    matrix,
    rhs,
    *,
    mass,
    jacobian,
    acceleration_rhs,
    rate_is_unique,
    stored_rows,
    stored_columns,
    evaluation_scale,
    tolerance,
    acceleration_equation_start,
):
    """Keep admitted rates exact; expose remaining rate freedoms, not chosen zeros.

    The caller proves independent mass rows AND columns. Field accelerations
    are solved sparsely; dense storage is state-by-small-border, never a dense
    field rank decomposition. Every original equation is checked afterwards.
    """
    mass, jacobian, matrix = map(sp.csr_matrix, (mass, jacobian, matrix))
    n = mass.shape[0]
    rows, columns = np.asarray(stored_rows), np.asarray(stored_columns)
    unknown = np.flatnonzero(~np.asarray(rate_is_unique))
    free_acceleration = np.setdiff1d(np.arange(n), columns)
    width = len(unknown) + len(free_acceleration)
    if (
        mass.shape != (n, n)
        or jacobian.shape != mass.shape
        or matrix.shape[1] != 2 * n
        or rows.ndim != 1
        or columns.ndim != 1
        or rows.dtype.kind not in 'iu'
        or columns.dtype.kind not in 'iu'
        or len(rows) != len(columns)
        or len(set(rows.tolist())) != len(rows)
        or len(set(columns.tolist())) != len(columns)
        or np.any(rows < 0)
        or np.any(rows >= n)
        or np.any(columns < 0)
        or np.any(columns >= n)
        or not 0 < width <= 512
    ):
        msg = 'stored acceleration elimination requires proven independent coordinates and a bounded actual border'
        raise ValueError(msg)
    factor = splu(mass[rows][:, columns].tocsc())
    base = np.zeros(2 * n)
    base[n + columns] = factor.solve(np.asarray(acceleration_rhs)[rows])
    sensitivity = np.zeros((n, width))
    forcing = sp.hstack((-jacobian[rows][:, unknown], -mass[rows][:, free_acceleration]), format='csr')
    sensitivity[columns] = factor.solve(forcing.toarray())
    sensitivity[free_acceleration, len(unknown) + np.arange(len(free_acceleration))] = 1
    rate_mapping = sp.hstack((sp.eye(n, format='csr')[:, unknown], sp.csr_matrix((n, len(free_acceleration)))))
    mapping = sp.vstack((rate_mapping, sp.csr_matrix(sensitivity)), format='csr')
    reduced = (matrix @ mapping).toarray()
    forcing = rhs - matrix @ base
    # These equations DEFINE the sparse elimination, rather than independent
    # numerical rank authority manufactured from its floating-point remainder.
    eliminated = acceleration_equation_start + rows
    reduced[eliminated] = 0
    forcing[eliminated] = 0
    norms = np.linalg.norm(reduced, axis=1)
    norms[norms == 0] = 1
    reduced, forcing = reduced / norms[:, None], forcing / norms
    column_norms = np.linalg.norm(reduced, axis=0)
    column_norms[column_norms == 0] = 1
    answer_scaled, free_scaled = border_affine_space(reduced / column_norms, forcing)
    answer = answer_scaled / column_norms
    correction = np.asarray(mapping @ answer).ravel() + base
    defect = matrix @ correction - rhs
    scale = np.asarray(abs(matrix) @ abs(correction)).ravel() + abs(rhs) + evaluation_scale
    if np.any(abs(defect) > tolerance * scale):
        msg = 'declared curvature contradicts actual stored-acceleration and differentiated equations'
        raise ValueError(msg)
    unique = np.asarray(rate_is_unique).copy()
    unique[unknown] = np.all(free_scaled[: len(unknown)] == 0, axis=1)
    return correction, unique
