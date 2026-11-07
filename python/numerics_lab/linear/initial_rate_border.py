"""Sparse stored-acceleration elimination with rank revelation only at its small border."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.sparse.linalg import splu

from numerics_lab.formulation import ResidualAssemblyBound
from numerics_lab.linear.border_rank import border_affine_space, rank_bar

__all__ = ['complete_rate_border']


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
    equation_bound,
    tolerance,
    acceleration_equation_start,
    algebraic_acceleration_rows,
):
    """Keep admitted rates exact; expose remaining rate freedoms, not chosen zeros.

    The caller proves independent mass rows AND columns. Field accelerations
    are solved sparsely; dense storage is state-by-small-border, never a dense
    field rank decomposition. A free acceleration of a coordinate with no
    storage (a massless field) past the bounded border is not a border freedom: the caller's
    ``algebraic_acceleration_rows`` -- twice-differentiated algebraic rows of
    ``matrix``, acceleration-only -- are matched to free accelerations and
    eliminated sparsely beside the stored rows. Only accelerations no such row
    reaches stay in the border. Every original equation is checked afterwards.
    """
    mass, jacobian, matrix = map(sp.csr_matrix, (mass, jacobian, matrix))
    n = mass.shape[0]
    rhs, acceleration_rhs, evaluation_scale = map(np.asarray, (rhs, acceleration_rhs, evaluation_scale))
    rate_is_unique = np.asarray(rate_is_unique)
    if (
        rhs.shape != (matrix.shape[0],)
        or not isinstance(equation_bound, ResidualAssemblyBound)
        or equation_bound.values.shape != rhs.shape
        or acceleration_rhs.shape != (n,)
        or evaluation_scale.shape != rhs.shape
        or rate_is_unique.shape != (n,)
        or rate_is_unique.dtype.kind != 'b'
        or any(value.dtype.kind not in 'fiu' for value in (rhs, acceleration_rhs, evaluation_scale))
        or any(not np.isfinite(value).all() for value in (rhs, acceleration_rhs, evaluation_scale))
        or any(not np.isfinite(value.data).all() for value in (mass, jacobian, matrix))
        or np.any(evaluation_scale < 0)
        or isinstance(tolerance, (bool, np.bool_))
        or not np.isscalar(tolerance)
        or not np.isreal(tolerance)
        or not np.isfinite(tolerance)
        or tolerance <= 0
        or isinstance(acceleration_equation_start, (bool, np.bool_))
        or not isinstance(acceleration_equation_start, (int, np.integer))
        or acceleration_equation_start < 0
        or acceleration_equation_start + n > matrix.shape[0]
    ):
        msg = 'stored acceleration border requires finite aligned actual equations and a boolean rate mask'
        raise ValueError(msg)
    rows, columns = np.asarray(stored_rows), np.asarray(stored_columns)
    algebraic_rows = np.asarray(algebraic_acceleration_rows)
    if (
        algebraic_rows.ndim != 1
        or (algebraic_rows.size and algebraic_rows.dtype.kind not in 'iu')
        or len(set(algebraic_rows.tolist())) != len(algebraic_rows)
        or np.any(algebraic_rows < 0)
        or np.any(algebraic_rows >= matrix.shape[0])
        or np.isin(algebraic_rows, acceleration_equation_start + np.arange(n)).any()
        or matrix[algebraic_rows.astype(int)][:, :n].count_nonzero()
    ):
        msg = 'algebraic acceleration rows must be distinct acceleration-only rows of the actual equations'
        raise ValueError(msg)
    algebraic_rows = algebraic_rows.astype(int)
    unknown = np.flatnonzero(~np.asarray(rate_is_unique))
    free_acceleration = np.setdiff1d(np.arange(n), columns)
    # Structural matching of the acceleration-only rows onto the free accelerations: a matched
    # pair joins the sparse elimination; its nonsingularity is proven by the factorization below.
    algebraic_block = matrix[algebraic_rows][:, n + free_acceleration].tocsr()
    algebraic_block.eliminate_zeros()
    # Only a border past its bound needs them: a bounded border keeps its rank-revealing dense
    # treatment, where a structurally matched but numerically dependent row stays a border row.
    needed = algebraic_rows.size and len(unknown) + len(free_acceleration) > 512
    matched_row = maximum_bipartite_matching(algebraic_block, perm_type='row') if needed else None
    reached = np.zeros(len(free_acceleration), dtype=bool) if matched_row is None else matched_row >= 0
    eliminated_rows = algebraic_rows[matched_row[reached]] if reached.any() else np.array([], dtype=int)
    columns_eliminated = np.concatenate((columns, free_acceleration[reached])).astype(int)
    free_acceleration = free_acceleration[~reached]
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
    # The eliminated equations in acceleration form: stored rows ``J v + M a = r`` and the
    # matched acceleration-only rows ``A a = r`` (their own rows of ``matrix``, scaled alike).
    acceleration_rows = sp.vstack((mass[rows], matrix[eliminated_rows][:, n:]), format='csr')
    rate_rows = sp.vstack((jacobian[rows], sp.csr_matrix((len(eliminated_rows), n))), format='csr')
    eliminated_rhs = np.concatenate((np.asarray(acceleration_rhs)[rows], rhs[eliminated_rows]))
    try:
        factor = splu(acceleration_rows[:, columns_eliminated].tocsc())
    except RuntimeError as exc:
        msg = 'declared stored acceleration coordinates are not independent'
        raise ValueError(msg) from exc
    base = np.zeros(2 * n)
    base[n + columns_eliminated] = factor.solve(eliminated_rhs)
    sensitivity = np.zeros((n, width))
    forcing = sp.hstack((-rate_rows[:, unknown], -acceleration_rows[:, free_acceleration]), format='csr')
    sensitivity[columns_eliminated] = factor.solve(forcing.toarray())
    sensitivity[free_acceleration, len(unknown) + np.arange(len(free_acceleration))] = 1
    rate_mapping = sp.hstack((sp.eye(n, format='csr')[:, unknown], sp.csr_matrix((n, len(free_acceleration)))))
    mapping = sp.vstack((rate_mapping, sp.csr_matrix(sensitivity)), format='csr')
    reduced = (matrix @ mapping).toarray()
    forcing = rhs - matrix @ base
    # These equations DEFINE the sparse elimination, rather than independent
    # numerical rank authority manufactured from its floating-point remainder.
    eliminated = np.concatenate((acceleration_equation_start + rows, eliminated_rows))
    reduced[eliminated] = 0
    forcing[eliminated] = 0
    norms = np.linalg.norm(reduced, axis=1)
    norms[norms == 0] = 1
    reduced, forcing = reduced / norms[:, None], forcing / norms
    column_norms = np.linalg.norm(reduced, axis=0)
    column_norms[column_norms == 0] = 1
    # These rows inherit the already-admitted first pair's evaluated arithmetic,
    # not a new physical zero floor. The final original-equation guard below
    # still decides acceptance with exactly its existing scale and tolerance.
    performed_allowance = equation_bound.allowance()
    allowance = (tolerance * (evaluation_scale + abs(rhs) + abs(matrix) @ abs(base)) + performed_allowance) / norms
    answer_scaled, free_scaled = border_affine_space(reduced / column_norms, forcing, inherited_allowance=allowance)
    answer = answer_scaled / column_norms
    correction = np.asarray(mapping @ answer).ravel() + base
    # Re-evaluate the actual stored equations with the final border values.
    # Summing base and sensitivity solves can lose a small prescribed
    # acceleration between huge opposing storage/source contributions.
    stored_rhs = (
        eliminated_rhs
        - rate_rows @ correction[:n]
        - acceleration_rows[:, free_acceleration] @ correction[n + free_acceleration]
    )
    stored_block = acceleration_rows[:, columns_eliminated]
    stored_acceleration = factor.solve(stored_rhs)
    for _ in range(3):
        stored_defect = stored_rhs - stored_block @ stored_acceleration
        if not np.any(stored_defect):
            break
        stored_acceleration += factor.solve(stored_defect)
    correction[n + columns_eliminated] = stored_acceleration
    defect = matrix @ correction - rhs
    scale = np.asarray(abs(matrix) @ abs(correction)).ravel() + abs(rhs) + evaluation_scale
    if np.any(abs(defect) > tolerance * scale + performed_allowance):
        msg = 'declared curvature contradicts actual stored-acceleration and differentiated equations'
        raise ValueError(msg)
    unique = np.asarray(rate_is_unique).copy()
    # The free directions are a NUMERICAL border solve of dimensionless columns: an exact zero comes back
    # as an ulp of residue on some LAPACKs, so it is judged against the rank bar, never an exact zero.
    unique[unknown] = np.linalg.norm(free_scaled[: len(unknown)], axis=1) <= rank_bar(reduced.shape)
    return correction, unique
