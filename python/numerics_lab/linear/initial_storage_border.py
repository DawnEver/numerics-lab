"""Joint state/rate authority after sparse stored-coordinate elimination."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from motronics.gauss.linear.arithmetic import certified_sparse_product
from motronics.gauss.linear.border_rank import border_affine_space, homogeneous_zero_coordinates
from motronics.gauss.linear.index_two import index_two_frozen_rate
from motronics.gauss.linear.initial_rate import normalize_sparse_rows
from motronics.gauss.linear.refinement import refine_factored_solution
from motronics.gauss.linear.storage_border import free_pinned_columns

__all__ = ['joint_coordinate_sparse_correction', 'stored_initial_correction']


def joint_coordinate_sparse_correction(
    matrix,
    rhs,
    *,
    mass,
    algebraic_rows,
    stored_rows,
    indices,
    residual_allowance=None,
    state_directions=None,
    row_scale=None,
    tolerance=1e-10,
):
    """Solve supplied joint columns through the actual sparse bulk/small border.

    ``row_scale`` is each original row's state operand scale; with it a dependent border row set
    is weighted by the rows' own bars (see ``_weighted_by_row_bars``).
    """
    n = mass.shape[0]
    if state_directions is not None:
        entries = state_directions.data if sp.issparse(state_directions) else np.asarray(state_directions, dtype=object)
        if any(isinstance(value, (bool, np.bool_)) for value in np.asarray(entries).flat):
            msg = 'sparse state range requires finite real bounded actual direction columns'
            raise ValueError(msg)
        directions = sp.csr_matrix(state_directions)
        if (
            directions.shape[0] != n
            or directions.shape[1] > 512
            or directions.dtype.kind not in 'fiu'
            or not np.isfinite(directions.data).all()
        ):
            msg = 'sparse state range requires finite real bounded actual direction columns'
            raise ValueError(msg)
        # Parametrize the owned range directly. A rational annihilator need
        # not have an exactly representable binaryfloat normalization.
        matrix = sp.bmat(
            [
                [matrix, sp.csr_matrix((matrix.shape[0], directions.shape[1]))],
                [sp.hstack((sp.eye(n, format='csr'), sp.csr_matrix((n, n))), format='csr'), -directions],
            ],
            format='csr',
        )
        rhs = np.concatenate((rhs, np.zeros(n)))
        residual_allowance = None if residual_allowance is None else np.concatenate((residual_allowance, np.zeros(n)))
        row_scale = None if row_scale is None else np.concatenate((row_scale, np.zeros(n)))
    change, rate_directions, _admitted = _sparse_joint_border(
        matrix,
        rhs,
        mass,
        algebraic_rows,
        stored_rows,
        indices,
        residual_allowance=residual_allowance,
        state_directions=None if state_directions is None else directions,
        row_scale=row_scale,
        tolerance=tolerance,
    )
    unique = np.concatenate((np.ones(mass.shape[0], dtype=bool), np.all(rate_directions == 0, axis=1)))
    return change[: 2 * n], unique


def _sparse_joint_border(
    matrix,
    rhs,
    mass,
    algebraic,
    stored_rows,
    indices,
    *,
    residual_allowance=None,
    state_directions=None,
    row_scale=None,
    tolerance=1e-10,
):
    """Eliminate an explicitly selected sparse state/rate block, not a dense field."""
    n = mass.shape[0]
    allowance = np.zeros(matrix.shape[0]) if residual_allowance is None else np.asarray(residual_allowance)
    if (
        allowance.shape != (matrix.shape[0],)
        or allowance.dtype.kind not in 'fiu'
        or not np.isfinite(allowance).all()
        or np.any(allowance < 0)
        or (
            residual_allowance is not None
            and any(isinstance(value, (bool, np.bool_)) for value in np.asarray(residual_allowance, dtype=object).flat)
        )
    ):
        msg = 'sparse joint allowance requires finite nonnegative original row bounds'
        raise ValueError(msg)
    unanchored, _ = homogeneous_zero_coordinates(matrix, rhs)
    original_support = np.asarray(matrix.getnnz(axis=1)).ravel() != 0
    retained_support = np.asarray(matrix[:, unanchored].getnnz(axis=1)).ravel() != 0
    if np.any(original_support & ~retained_support & (abs(rhs) > allowance)):
        msg = 'joint stored state contradicts actual state/rate equations at exact homogeneous anchors'
        raise ValueError(msg)
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
    remaining_columns = np.setdiff1d(np.arange(matrix.shape[1]), selected_columns)
    if len(remaining_columns) > 512:
        msg = 'joint storage elimination requires a bounded actual residual border'
        raise NotImplementedError(msg)
    block = matrix[selected_rows][:, selected_columns].tocsr()
    raw_solve = _triangular_joint_solver(matrix, mass, stored_rows, stored_positions, indices, differential)

    def solve(forcing):
        return refine_factored_solution(block, forcing, raw_solve(forcing), raw_solve)

    if not len(remaining_columns):
        correction = np.zeros(matrix.shape[1])
        correction[selected_columns] = solve(rhs[selected_rows])
        correction[~unanchored] = 0
        return correction, np.empty((n, 0)), allowance

    coupling = matrix[selected_rows][:, remaining_columns]

    def column_forcing(column):
        entries = coupling[:, column].tocoo()
        forcing = np.zeros(block.shape[0])
        np.add.at(forcing, entries.row, entries.data)
        return forcing

    def column_response(column):
        forcing = column_forcing(column)
        return solve(forcing) if np.any(forcing) else forcing

    def response_error(column):
        residual = column_forcing(column) - block @ response[:, column]
        return abs(raw_solve(residual)) if np.any(residual) else residual

    response = np.column_stack([column_response(column) for column in range(len(remaining_columns))])
    left = matrix[remaining_rows][:, selected_columns]
    direct = matrix[remaining_rows][:, remaining_columns]
    schur = direct - sp.csr_matrix(left @ response)
    schur.eliminate_zeros()
    # The Schur border is COMPUTED, and a solid body's net-current row is exactly
    # dependent on its hidden row: in the border it is a cancellation of terms many
    # orders larger than what is left. Its entries' own error seeds the eliminator, or
    # it pivots on that remainder (Cummins step 1: body rows at 1e-15 of their 5.2
    # operands, subsystem cond 1e18). Formation rounding is gamma_k over the summed
    # |terms|; the responses' forward error is the image of their next refinement
    # correction, an ESTIMATE as in `_driven_allowance_image`.
    count = np.asarray(sp.csr_matrix(left).getnnz(axis=1)).ravel() + 1.0
    unit = np.finfo(float).eps / 2
    gamma = count * unit / (1 - count * unit)
    errors = np.column_stack([response_error(column) for column in range(len(remaining_columns))])
    abs_left = abs(sp.csr_matrix(left))
    schur_errors = gamma[:, None] * np.asarray(abs(direct) + abs_left @ abs(response)) + abs_left @ errors
    # An eliminated row's ADMITTED slack (its forcing within its own allowance -- the remainder a
    # converged field seed hands over) does not vanish with the row: elimination carries it into
    # the Schur forcing. So the forcing is formed as two images, driven and slack, and a retained
    # row inherits the slack image in full plus the rounding of forming and adding the two
    # (gamma_k over the |terms| summed, k the row's term count) -- otherwise a homogeneous anchor
    # downstream reads a converged seed's remainder as a contradicted declaration.
    selected_rhs = rhs[selected_rows]
    slack = np.where(abs(selected_rhs) <= allowance[selected_rows], selected_rhs, 0.0)
    driven_base = solve(selected_rhs - slack)
    slack_base = solve(slack) if np.any(slack) else np.zeros_like(driven_base)
    driven = rhs[remaining_rows] - left @ driven_base
    carried = -(left @ slack_base)
    # With row bars, an eliminated row's slack (inside its own certified bound) STAYS on it: it
    # is not re-solved and not exported through elimination onto a retained row whose own bar
    # is far smaller (Cummins: the field rows' remainders summed onto a body row at 1.3x).
    kept = slack if row_scale is not None else np.zeros_like(slack)
    forcing = driven if row_scale is not None else driven + carried
    terms = np.asarray(sp.csr_matrix(left).getnnz(axis=1)).ravel() + 2
    gamma = terms * np.finfo(float).eps / (1.0 - terms * np.finfo(float).eps)
    inherited = (
        allowance[remaining_rows]
        + abs(carried)
        + gamma * (abs(rhs[remaining_rows]) + abs(left) @ (abs(driven_base) + abs(slack_base)) + abs(carried))
    )
    driven_rows = abs(selected_rhs) > allowance[selected_rows]
    inherited += _driven_allowance_image(
        block, left, np.where(driven_rows, allowance[selected_rows], 0.0), abs(forcing) > inherited
    )
    supported = np.asarray(schur.getnnz(axis=1)).ravel() != 0
    if np.any(abs(forcing[~supported]) > inherited[~supported]):
        msg = 'joint stored state contradicts actual eliminated original rows'
        raise ValueError(msg)
    answer, directions = border_affine_space(
        schur[supported].toarray(),
        forcing[supported],
        inherited_allowance=inherited[supported],
        entry_errors=schur_errors[supported],
    )
    if row_scale is not None:
        # A DEPENDENT border row set is consistent only to its data's rounding, and the
        # elimination leaves that inconsistency on whichever row it did not pivot. Cummins
        # admission 80: the integrator's stored target meets a solid body's algebraic
        # net-current constraint to ~2e-15 of that constraint's operands, and the whole of it
        # landed on the body row, whose own operands are 3e5x smaller -- 6e-10 of its scale,
        # every iteration, beyond any refinement. Each row is weighted by its own bar.
        trial = np.zeros(matrix.shape[1])
        trial[remaining_columns] = answer
        trial[selected_columns] = solve(rhs[selected_rows] - kept - coupling @ answer)
        rows = remaining_rows[supported]
        bars = tolerance * (np.asarray(abs(matrix[rows]) @ abs(trial)).ravel() + abs(rhs[rows]) + row_scale[rows])
        answer = _weighted_by_row_bars(schur[supported].toarray(), forcing[supported], answer, bars)
    correction = np.zeros(matrix.shape[1])
    correction[remaining_columns] = answer
    correction[selected_columns] = solve(rhs[selected_rows] - kept - coupling @ answer)
    full_directions = np.zeros((matrix.shape[1], directions.shape[1]))
    full_directions[remaining_columns] = directions
    full_directions[selected_columns] = -response @ directions
    if state_directions is not None:
        correction[:n] = state_directions @ correction[2 * n :]
        full_directions[:n] = state_directions @ full_directions[2 * n :]
    # Reconstruct coordinates proved zero by original Q/constraint rows, not
    # cancellation of driven sparse responses. The same exact lift applies to
    # every homogeneous direction; all original equations remain validators.
    correction[~unanchored] = 0
    full_directions[~unanchored] = 0
    if np.any(full_directions[:n] != 0):
        msg = 'joint stored-coordinate equations require additional physical state authority'
        raise ValueError(msg)
    # What each original row may keep: its own certified bound, and on a retained row
    # also what elimination carried into it (slack image plus driven rounding image).
    admitted = allowance.copy()
    admitted[remaining_rows] = inherited
    return correction, full_directions[n : 2 * n], admitted


def _weighted_by_row_bars(schur, forcing, answer, bars):
    """Move ``answer`` to the least squares of the border rows, each in units of its own bar.

    Independent rows are already met; only a dependent set's residual moves, onto the rows whose
    bars can hold it. A row whose bar is zero (or below the normal range) is EXACT: it is kept as an
    equality constraint, never divided by. Free columns (no entry) are not touched.
    """
    live = np.flatnonzero(np.any(schur != 0, axis=0))
    if not live.size:
        return answer
    matrix = schur[:, live]
    residual = forcing - schur @ answer
    exact = ~(bars >= np.finfo(float).tiny)
    step = np.zeros(len(live))
    basis = np.eye(len(live))
    if np.any(exact):
        # Exact rows: the minimum-norm step that meets them, then move only in their null space.
        u, sigma, vt = np.linalg.svd(matrix[exact], full_matrices=True)
        rank = int(np.sum(sigma > sigma[0] * np.finfo(float).eps * max(matrix[exact].shape))) if sigma.size else 0
        step = vt[:rank].T @ ((u[:, :rank].T @ residual[exact]) / sigma[:rank])
        basis = vt[rank:].T
    soft = ~exact
    if np.any(soft) and basis.shape[1]:
        weighted = (matrix[soft] @ basis) / bars[soft, None]
        norm = np.linalg.norm(weighted, axis=0)
        usable = norm > 0
        if np.any(usable):
            target = (residual[soft] - matrix[soft] @ step) / bars[soft]
            reduced, *_ = np.linalg.lstsq(weighted[:, usable] / norm[usable], target, rcond=None)
            step = step + basis[:, usable] @ (reduced / norm[usable])
    moved = answer.copy()
    moved[live] += step
    return moved


def _driven_allowance_image(block, left, bound, needed):
    """``|left B^-1| bound`` on the rows that need it: a DRIVEN row's certified rounding, eliminated.

    A driven eliminated row (its forcing outside its own allowance) still carries its residual's
    rounding ``|delta_j| <= bound_j``, and elimination maps it into a retained row as
    ``left B^-1 delta`` -- signed, so no forward solve bounds it. The row's adjoint ``y`` (``B^T y
    = left^T``) does: ``|y| . bound``. Formed only for rows whose forcing exceeds the allowance
    they already inherit, and granted only when the refined adjoint is NORMWISE backward stable
    (``max|B^T y - left^T| <= gamma_k max(|B^T||y| + |left|)``); otherwise the row inherits
    nothing more and its refusal stands. ``y`` is the computed elimination's adjoint -- the same
    computed elimination whose forward image the inherited term already reads -- and its residual
    forward error is estimated by the next refinement correction, whose image is added; that one
    term is an estimate, not a certificate, and is stated as such. Componentwise stability is not
    asked: measured on Cummins, rows of the mixed-unit block with a two-entry support keep their
    whole (1e-14..1e-8) defect while the normwise ratio is 0.006..0.013.
    """
    image = np.zeros(left.shape[0])
    rows = np.flatnonzero(needed & (np.asarray(sp.csr_matrix(left).getnnz(axis=1)).ravel() != 0))
    if not np.any(bound) or not rows.size:
        return image
    transposed = sp.csr_matrix(block.T)
    try:
        normalized, _ = normalize_sparse_rows(transposed, np.zeros(transposed.shape[0]))
        factor = splu(normalized.tocsc())
    except (RuntimeError, ValueError):
        return image

    def correction(defect):
        return factor.solve(normalize_sparse_rows(transposed, defect)[1])

    count = float(np.max(transposed.getnnz(axis=1))) + 1.0
    gamma = count * np.finfo(float).eps / (1 - count * np.finfo(float).eps)
    for row in rows:
        target = np.asarray(left[row].toarray()).ravel()
        # Refined here rather than through `refine_factored_solution`, whose magnitude gate
        # refuses this mixed-unit adjoint (cond_1 ~ 1e25 on Cummins) although its refinement
        # trajectory collapses; the normwise check below is the acceptance.
        adjoint = correction(target)
        for _ in range(4):
            adjoint = adjoint + correction(target - transposed @ adjoint)
        defect = target - transposed @ adjoint
        scale = np.max(abs(transposed) @ abs(adjoint) + abs(target))
        if np.all(np.isfinite(adjoint)) and np.max(abs(defect)) <= gamma * scale:
            # The remaining forward error of ``y`` is the next correction; its image is added.
            image[row] = (abs(adjoint) + abs(correction(defect))) @ bound
    return image


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
    residual_rate = matrix[:n, n:].tocsr()
    residual_rate.eliminate_zeros()
    if residual_rate[bulk][:, air].nnz or residual_rate[air][:, indices].nnz:
        msg = 'sparse joint rate minor requires actual stored/zero-rate coordinate support'
        raise ValueError(msg)
    residual_rate_minor = residual_rate[bulk][:, bulk]
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
    solve_residual_rate = (
        solve_stored if (residual_rate_minor - stored_minor).nnz == 0 else factored(residual_rate_minor)
    )
    solve_state = factored(state_minor)
    solve_rate = solve_state if (rate_minor - state_minor).nnz == 0 else factored(rate_minor)

    def solve(forcing):
        value = np.zeros(2 * size)
        value[bulk_positions] = solve_stored(forcing[size : size + len(bulk)])
        value[air_positions] = solve_state(forcing[air_positions] - matrix[air][:, bulk] @ value[bulk_positions])
        value[size + bulk_positions] = solve_residual_rate(
            forcing[bulk_positions] - matrix[bulk][:, indices] @ value[:size]
        )
        value[size + air_positions] = solve_rate(
            forcing[size + len(bulk) :]
            - differentiated[:, indices] @ value[:size]
            - differentiated[:, n + bulk] @ value[size + bulk_positions]
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
    # These original rows contain no rate: their already-validated STATE
    # residual keeps exactly the same unsigned operands after freezing state.
    # Dynamic and independently differentiated rows inherit no such allowance.
    state_scale = unsigned + np.asarray(abs(jacobian) @ abs(state)).ravel()
    roundoff[zero_mass] += tolerance * state_scale[zero_mass]
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
    residual_allowance=None,
):
    """Eliminate stored state then stored rate; reveal only the actual free border.

    ``residual_allowance`` is the certified rounding bound of ``residual`` per row; it
    travels through the sparse elimination and every original row may keep its image.
    """
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
    columns = free_pinned_columns(columns, mass[stored_rows], certified_sparse_product(algebraic, rate_tangent))
    free = np.setdiff1d(np.arange(n), columns)
    width = 2 * len(free)
    if sparse_state_indices is not None:
        stored = mass[stored_rows]
        matrix = sp.bmat(
            [
                [jacobian, mass],
                [stored, sp.csr_matrix(stored.shape)],
                [sp.csr_matrix(algebraic.shape), certified_sparse_product(algebraic, rate_tangent)],
            ],
            format='csr',
        )
        rhs = np.concatenate((-residual, stored_change, -algebraic @ partial))
        allowance = (
            None if residual_allowance is None else np.concatenate((residual_allowance, np.zeros(matrix.shape[0] - n)))
        )
        row_scale = np.concatenate((
            np.asarray(abs(jacobian) @ abs(state)).ravel(),
            np.asarray(abs(stored) @ abs(state)).ravel(),
            np.zeros(algebraic.shape[0]),
        ))
        correction, rate_directions, admitted = _sparse_joint_border(
            matrix,
            rhs,
            mass,
            algebraic,
            stored_rows,
            sparse_state_indices,
            residual_allowance=allowance,
            row_scale=row_scale,
            tolerance=tolerance,
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
            admitted=admitted,
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
            [sp.csr_matrix(algebraic.shape), certified_sparse_product(algebraic, rate_tangent)],
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
    *,
    admitted=None,
):
    n = mass.shape[0]
    # Each original row keeps the certified rounding image the elimination carried to it.
    admitted = np.zeros(matrix.shape[0]) if admitted is None else admitted
    state_defect = jacobian @ correction[:n] + mass @ correction[n:] + residual
    state_scale = (
        np.asarray(abs(jacobian) @ abs(correction[:n]) + abs(mass) @ abs(correction[n:])).ravel()
        + abs(residual)
        + np.asarray(abs(jacobian) @ abs(state)).ravel()
    )
    if (
        not np.isfinite(state_defect).all()
        or not np.isfinite(state_scale).all()
        or np.any(abs(state_defect) > tolerance * state_scale + admitted[:n])
    ):
        msg = 'joint stored state contradicts actual state/rate equations'
        raise ValueError(msg)
    # Stored state is exact authority; stored rate is not reconstructed by a
    # subtraction of large dynamic operands. Solve every original frozen-state
    # rate equation, retaining the independently differentiated weak constraints.
    forcing, roundoff = _frozen_rate_forcing(jacobian, residual, correction[:n], mass, state, tolerance)
    roundoff = roundoff + admitted[:n]
    # Give the independent differentiated equations first choice of numerical
    # pivots. All dynamic rows still remain original physical witnesses.
    try:
        # An index-two state (a solid body's voltage) is re-solved here with
        # the rates: its value is fixed only by the differentiated hidden rows.
        columns, state_step, correction[n:], rate_unique = index_two_frozen_rate(
            algebraic,
            certified_sparse_product(algebraic, rate_tangent),
            mass,
            jacobian,
            -algebraic @ partial,
            forcing,
            tolerance=tolerance,
            allowance=roundoff,
        )
        correction[columns] += state_step
    except ValueError as error:
        msg = f'joint stored state contradicts actual state/rate equations: {error}'
        raise ValueError(msg) from error
    defect = matrix @ correction - rhs
    equation_scale = np.asarray(abs(matrix) @ abs(correction)).ravel() + abs(rhs)
    # The residual was evaluated at the actual state: a row whose coordinates
    # are all fixed by storage keeps that evaluation's size as its only scale,
    # the same scale the nonlinear owner admits its algebraic state rows by.
    equation_scale[:n] += np.asarray(abs(jacobian) @ abs(state)).ravel()
    stored = matrix[n : 2 * n - algebraic.shape[0], :n]
    equation_scale[n : 2 * n - algebraic.shape[0]] += np.asarray(abs(stored) @ abs(state)).ravel()
    if np.any(abs(defect) > tolerance * equation_scale + admitted):
        msg = 'joint stored state contradicts actual state/rate equations'
        raise ValueError(msg)
    unique = np.ones(n, dtype=bool)
    unique[np.any(rate_directions != 0, axis=1)] = False
    return correction, unique & rate_unique
