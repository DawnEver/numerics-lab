"""Joint state/rate authority after sparse stored-coordinate elimination."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from motronics.gauss.linear.border_rank import border_affine_space
from motronics.gauss.linear.initial_rate import sparse_initial_rate
from motronics.gauss.linear.storage_border import free_pinned_columns


def _frozen_rate_forcing(jacobian, residual, correction, mass, state, tolerance):
    # Bound only the rounding of this actual top-row RHS construction. A
    # differentiated constraint gets no inherited large dynamic-row allowance.
    entries = jacobian.tocoo()
    multipliers = correction[entries.col]
    products = entries.data * multipliers
    if (
        not np.isfinite(products).all()
        or np.any((entries.data != 0) & (multipliers != 0) & (products == 0))
        or np.any((products != 0) & (abs(products) < np.finfo(float).tiny))
    ):
        msg = 'joint rate forcing requires representable normal finite products'
        raise ValueError(msg)
    terms = np.asarray(jacobian.getnnz(axis=1)).ravel()
    operations = np.where(terms != 0, 2.0 * terms + 1, 0.0)
    unit = np.finfo(float).eps / 2
    if not np.isfinite(operations).all() or np.any(operations * unit >= 1):
        msg = 'joint rate forcing operation count exceeds its finite arithmetic bound'
        raise ValueError(msg)
    gamma = operations * unit / (1 - operations * unit)
    if not np.isfinite(gamma).all() or np.any(gamma >= 1):
        msg = 'joint rate forcing operation count exceeds its finite arithmetic bound'
        raise ValueError(msg)
    unsigned = abs(residual) + np.asarray(abs(jacobian) @ abs(correction)).ravel()
    roundoff = gamma * unsigned / (1 - gamma)
    if np.any((gamma != 0) & (unsigned != 0) & (roundoff < np.finfo(float).tiny)):
        msg = 'joint rate forcing cannot certify an underflowed arithmetic bound'
        raise ValueError(msg)
    nonzero_mass = mass.copy()
    nonzero_mass.eliminate_zeros()
    zero_mass = np.asarray(nonzero_mass.getnnz(axis=1)).ravel() == 0
    roundoff[zero_mass] += tolerance * np.asarray(abs(jacobian[zero_mass]) @ abs(state)).ravel()
    if not np.isfinite(roundoff).all():
        msg = 'joint rate forcing requires finite row-local arithmetic bounds'
        raise ValueError(msg)
    forcing = -residual - jacobian @ correction
    if not np.isfinite(forcing).all():
        msg = 'joint rate forcing requires a finite actual right-hand side'
        raise ValueError(msg)
    return forcing, roundoff


def stored_initial_correction(
    jacobian,
    residual,
    mass,
    algebraic,
    stored_rows,
    stored_columns,
    stored_change,
    partial,
    *,
    tolerance,
    state,
    rate_tangent,
):
    """Eliminate stored state then stored rate; reveal only the actual free border."""
    n = mass.shape[0]
    columns = np.asarray(stored_columns)
    if (
        columns.shape != stored_rows.shape
        or columns.dtype.kind not in 'iu'
        or len(set(columns.tolist())) != len(columns)
        or np.any(columns < 0)
        or np.any(columns >= n)
    ):
        msg = 'joint storage elimination requires independently declared mass columns'
        raise ValueError(msg)
    columns = free_pinned_columns(columns, mass[stored_rows], algebraic @ rate_tangent)
    free = np.setdiff1d(np.arange(n), columns)
    width = 2 * len(free)
    if not 0 < width <= 512:
        msg = 'joint storage elimination requires a bounded actual free state/rate border'
        raise NotImplementedError(msg)
    block = mass[stored_rows][:, columns].tocsc()
    try:
        factor = splu(block)
    except RuntimeError as exc:
        msg = 'joint storage elimination requires independent actual stored coordinates'
        raise ValueError(msg) from exc

    def solve(rhs):
        value = factor.solve(rhs)
        for _ in range(3):
            defect = rhs - block @ value
            if not np.any(defect):
                break
            value += factor.solve(defect)
        return value

    state_base = np.zeros(n)
    state_base[columns] = solve(stored_change)
    state_map = np.zeros((n, width))
    state_map[free, np.arange(len(free))] = 1
    state_map[columns, : len(free)] = solve(-mass[stored_rows][:, free].toarray())
    rate_base = np.zeros(n)
    rate_base[columns] = solve(-residual[stored_rows] - jacobian[stored_rows] @ state_base)
    rate_map = np.zeros((n, width))
    rate_map[free, len(free) + np.arange(len(free))] = 1
    rate_map[columns] = solve(-jacobian[stored_rows] @ state_map - mass[stored_rows] @ rate_map)
    mapping = sp.csr_matrix(np.vstack((state_map, rate_map)))
    base = np.concatenate((state_base, rate_base))
    stored = mass[stored_rows]
    matrix = sp.bmat(
        [
            [jacobian, mass],
            [stored, sp.csr_matrix(stored.shape)],
            [sp.csr_matrix(algebraic.shape), algebraic @ rate_tangent],
        ],
        format='csr',
    )
    rhs = np.concatenate((-residual, stored_change, -algebraic @ partial))
    reduced = (matrix @ mapping).toarray()
    forcing = rhs - matrix @ base
    eliminated = np.concatenate((stored_rows, n + np.arange(len(stored_rows))))
    reduced[eliminated], forcing[eliminated] = 0, 0
    row_norm = np.linalg.norm(reduced, axis=1)
    row_norm[row_norm == 0] = 1
    reduced, forcing = reduced / row_norm[:, None], forcing / row_norm
    column_norm = np.linalg.norm(reduced, axis=0)
    column_norm[column_norm == 0] = 1
    answer, directions = border_affine_space(reduced / column_norm, forcing)
    if np.any(directions[: len(free)] != 0):
        msg = 'joint stored-coordinate equations require additional physical state authority'
        raise ValueError(msg)
    correction = np.asarray(mapping @ (answer / column_norm)).ravel() + base
    # Reconstruct final stored coordinates from their actual RHS, not sums of
    # sensitive responses to very differently scaled border contributions.
    correction[columns] = solve(stored_change - mass[stored_rows][:, free] @ correction[free])
    correction[n + columns] = solve(
        -residual[stored_rows]
        - jacobian[stored_rows] @ correction[:n]
        - mass[stored_rows][:, free] @ correction[n + free]
    )
    # Stored state is exact authority; stored rate is not reconstructed by a
    # subtraction of large dynamic operands. Solve every original frozen-state
    # rate equation, retaining the independently differentiated weak constraints.
    forcing, roundoff = _frozen_rate_forcing(jacobian, residual, correction[:n], mass, state, tolerance)
    # Give the independent differentiated equations first choice of numerical
    # pivots. All dynamic rows still remain original physical witnesses.
    try:
        correction[n:], rate_unique = sparse_initial_rate(
            sp.vstack((algebraic @ rate_tangent, mass), format='csr'),
            np.concatenate((-algebraic @ partial, forcing)),
            tolerance=tolerance,
            residual_allowance=np.concatenate((np.zeros(algebraic.shape[0]), roundoff)),
        )
    except ValueError as error:
        msg = f'joint stored state contradicts actual state/rate equations: {error}'
        raise ValueError(msg) from error
    defect = matrix @ correction - rhs
    equation_scale = np.asarray(abs(matrix) @ abs(correction)).ravel() + abs(rhs)
    # The residual was evaluated at the actual state: a row whose coordinates
    # are all fixed by storage keeps that evaluation's size as its only scale,
    # the same scale the nonlinear owner admits its algebraic state rows by.
    equation_scale[:n] += np.asarray(abs(jacobian) @ abs(state)).ravel()
    if np.any(abs(defect) > tolerance * equation_scale):
        msg = 'joint stored state contradicts actual state/rate equations'
        raise ValueError(msg)
    unique = np.ones(n, dtype=bool)
    rate_directions = rate_map @ (directions / column_norm[:, None])
    unique[np.any(rate_directions != 0, axis=1)] = False
    return correction, unique & rate_unique
