"""Joint state/rate authority after sparse stored-coordinate elimination."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from motronics.gauss.linear.border_rank import border_affine_space
from motronics.gauss.linear.initial_rate import normalize_sparse_rows, sparse_initial_rate
from motronics.gauss.linear.refinement import refine_factored_solution
from motronics.gauss.linear.storage_border import free_pinned_columns


def _sparse_joint_border(matrix, rhs, mass, algebraic, stored_rows, indices):
    """Eliminate an explicitly selected sparse state/rate block, not a dense field."""
    n = mass.shape[0]
    indices = np.asarray(indices)
    if (
        indices.ndim != 1
        or indices.dtype.kind not in 'iu'
        or not len(indices)
        or len(set(indices.tolist())) != len(indices)
        or np.any(indices < 0)
        or np.any(indices >= n)
    ):
        msg = 'sparse state elimination requires unique actual coordinate indices'
        raise ValueError(msg)
    stored_positions = np.flatnonzero(np.isin(stored_rows, indices))
    normalized = algebraic.tocsr(copy=True)
    normalized.eliminate_zeros()
    selectors = {}
    for row in range(normalized.shape[0]):
        start, end = normalized.indptr[row : row + 2]
        if end - start == 1 and normalized.data[start] == 1:
            selectors[int(normalized.indices[start])] = row
    zero_mass = mass.copy()
    zero_mass.eliminate_zeros()
    differential = [
        selectors[int(column)] for column in indices if zero_mass[column].nnz == 0 and int(column) in selectors
    ]
    selected_rows = np.concatenate((
        indices,
        n + stored_positions,
        n + len(stored_rows) + np.asarray(differential, dtype=int),
    ))
    selected_columns = np.concatenate((indices, n + indices))
    if len(selected_rows) != len(selected_columns):
        msg = 'sparse state elimination requires original stored and algebraic selector witnesses'
        raise ValueError(msg)
    remaining_rows = np.setdiff1d(np.arange(matrix.shape[0]), selected_rows)
    remaining_columns = np.setdiff1d(np.arange(2 * n), selected_columns)
    if len(remaining_columns) > 512:
        msg = 'joint storage elimination requires a bounded actual residual border'
        raise NotImplementedError(msg)
    block = matrix[selected_rows][:, selected_columns].tocsr()
    raw_solve = _triangular_joint_solver(matrix, mass, stored_rows, stored_positions, indices, differential)

    def solve(forcing):
        return refine_factored_solution(block, forcing, raw_solve(forcing), raw_solve)

    if not len(remaining_columns):
        correction = np.zeros(2 * n)
        correction[selected_columns] = solve(rhs[selected_rows])
        return correction, np.empty((n, 0))

    coupling = matrix[selected_rows][:, remaining_columns]
    response = np.column_stack([
        solve(coupling[:, column].toarray().ravel()) for column in range(len(remaining_columns))
    ])
    base = solve(rhs[selected_rows])
    left = matrix[remaining_rows][:, selected_columns]
    schur = matrix[remaining_rows][:, remaining_columns].toarray() - left @ response
    forcing = rhs[remaining_rows] - left @ base
    answer, directions = border_affine_space(schur, forcing)
    correction = np.zeros(2 * n)
    correction[remaining_columns] = answer
    correction[selected_columns] = solve(rhs[selected_rows] - coupling @ answer)
    full_directions = np.zeros((2 * n, directions.shape[1]))
    full_directions[remaining_columns] = directions
    full_directions[selected_columns] = -response @ directions
    if np.any(full_directions[:n] != 0):
        msg = 'joint stored-coordinate equations require additional physical state authority'
        raise ValueError(msg)
    return correction, full_directions[n:]


def _triangular_joint_solver(matrix, mass, stored_rows, stored_positions, indices, differential):
    """Factor actual sparse storage/state/rate minors without mixed-unit pivots."""
    mass = mass.copy()
    mass.eliminate_zeros()
    n, size = mass.shape[0], len(indices)
    bulk = stored_rows[stored_positions]
    air = indices[~np.isin(indices, bulk)]
    if mass[bulk][:, air].nnz or mass[air].nnz or len(differential) != len(air):
        msg = 'sparse triangular elimination requires exact stored/zero-mass coordinate support'
        raise ValueError(msg)
    positions = {int(column): index for index, column in enumerate(indices)}
    bulk_positions = np.array([positions[int(column)] for column in bulk], dtype=int)
    air_positions = np.array([positions[int(column)] for column in air], dtype=int)
    stored_minor = mass[bulk][:, bulk].tocsr()
    state_minor = matrix[air][:, air].tocsr()
    differentiated = matrix[n + len(stored_rows) + np.asarray(differential)]
    rate_minor = differentiated[:, n + air].tocsr()

    def factored(operator):
        if operator.shape == (0, 0):
            return lambda forcing: np.array(forcing, copy=True)
        normalized, _ = normalize_sparse_rows(operator, np.zeros(operator.shape[0]))
        try:
            factor = splu(normalized.tocsc())
        except RuntimeError as error:
            msg = 'declared sparse state/rate minor lacks independent actual equation authority'
            raise ValueError(msg) from error

        def solve(forcing):
            def correction(defect):
                return factor.solve(normalize_sparse_rows(operator, defect)[1])

            return refine_factored_solution(operator, forcing, correction(forcing), correction)

        return solve

    solve_stored = factored(stored_minor)
    solve_state = factored(state_minor)
    solve_rate = solve_state if (rate_minor - state_minor).nnz == 0 else factored(rate_minor)

    def solve(forcing):
        value = np.zeros(2 * size)
        value[bulk_positions] = solve_stored(forcing[size : size + len(bulk)])
        value[air_positions] = solve_state(forcing[air_positions] - matrix[air][:, bulk] @ value[bulk_positions])
        value[size + bulk_positions] = solve_stored(forcing[bulk_positions] - matrix[bulk][:, indices] @ value[:size])
        value[size + air_positions] = solve_rate(
            forcing[size + len(bulk) :] - differentiated[:, n + bulk] @ value[size + bulk_positions]
        )
        return value

    return solve


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
    sparse_state_indices=None,
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
    if sparse_state_indices is not None and width > 512:
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
        correction, rate_directions = _sparse_joint_border(
            matrix, rhs, mass, algebraic, stored_rows, sparse_state_indices
        )
        return _validate_joint_correction(
            correction,
            rate_directions,
            matrix,
            rhs,
            jacobian,
            residual,
            mass,
            algebraic,
            partial,
            tolerance,
            state,
            rate_tangent,
        )
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
    rate_directions = rate_map @ (directions / column_norm[:, None])
    return _validate_joint_correction(
        correction,
        rate_directions,
        matrix,
        rhs,
        jacobian,
        residual,
        mass,
        algebraic,
        partial,
        tolerance,
        state,
        rate_tangent,
    )


def _validate_joint_correction(
    correction,
    rate_directions,
    matrix,
    rhs,
    jacobian,
    residual,
    mass,
    algebraic,
    partial,
    tolerance,
    state,
    rate_tangent,
):
    n = mass.shape[0]
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
    unique[np.any(rate_directions != 0, axis=1)] = False
    return correction, unique & rate_unique
