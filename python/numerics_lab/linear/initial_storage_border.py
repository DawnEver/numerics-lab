"""Joint state/rate authority after sparse stored-coordinate elimination."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from numerics_lab.linear.arithmetic import certified_sparse_product
from numerics_lab.linear.border_rank import border_affine_space, homogeneous_zero_coordinates
from numerics_lab.linear.index_two import _column_components, _sparse_block_columns, index_two_frozen_rate
from numerics_lab.linear.initial_rate import normalize_sparse_rows
from numerics_lab.linear.refinement import refine_factored_solution
from numerics_lab.linear.resolution import row_bar, system_resolution
from numerics_lab.linear.storage_border import free_pinned_columns

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
    state_direction_columns=None,
    row_scale=None,
    tolerance=1e-10,
):
    """Solve supplied joint columns through the actual sparse bulk/small border.

    ``row_scale`` is each original row's state operand scale; with it a dependent border row set
    is weighted by the rows' own bars (see ``_weighted_by_row_bars``).
    """
    n = mass.shape[0]
    if state_direction_columns is not None and state_directions is None:
        msg = 'mixed state direction columns require the actual state range'
        raise ValueError(msg)
    if state_directions is not None:
        entries = state_directions.data if sp.issparse(state_directions) else np.asarray(state_directions, dtype=object)
        if any(isinstance(value, (bool, np.bool_)) for value in np.asarray(entries).flat):
            msg = 'sparse state range requires finite real bounded actual direction columns'
            raise ValueError(msg)
        directions = sp.csr_matrix(state_directions)
        if directions.shape[0] != n or directions.dtype.kind not in 'fiu' or not np.isfinite(directions.data).all():
            msg = 'sparse state range requires finite real bounded actual direction columns'
            raise ValueError(msg)
        extra = sp.csr_matrix((matrix.shape[0], directions.shape[1]))
        if state_direction_columns is not None:
            entries = (
                state_direction_columns.data
                if sp.issparse(state_direction_columns)
                else np.asarray(state_direction_columns, dtype=object)
            )
            extra = sp.csr_matrix(state_direction_columns)
            if (
                extra.shape != (matrix.shape[0], directions.shape[1])
                or extra.dtype.kind not in 'fiu'
                or not np.isfinite(extra.data).all()
                or any(isinstance(value, (bool, np.bool_)) for value in np.asarray(entries).flat)
            ):
                msg = 'mixed state direction columns require finite real original equation dimensions'
                raise ValueError(msg)
        # Parametrize the owned range directly. A rational annihilator need
        # not have an exactly representable binaryfloat normalization.
        matrix = sp.bmat(
            [
                [matrix, extra],
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
    unique = np.concatenate((np.ones(mass.shape[0], dtype=bool), _rate_uniqueness(rate_directions)))
    return change[: 2 * n], unique


def _rate_uniqueness(directions):
    if sp.issparse(directions):
        return np.asarray((directions != 0).getnnz(axis=1)).ravel() == 0
    return np.all(directions == 0, axis=1)


def _component_affine(matrix, forcing, errors, allowance):
    """Rank only actual Schur/error-support components, never discovery components."""
    pattern = (matrix != 0).maximum(errors != 0)
    components = list(_column_components(pattern))
    answer = np.zeros(matrix.shape[1])

    def blocks():
        for members, rows in components:
            if len(members) > 512 or len(rows) * len(members) > 512**2:
                msg = 'joint correction requires bounded actual Schur component rank work'
                raise NotImplementedError(msg)
            if len(rows):
                value, directions = border_affine_space(
                    matrix[rows][:, members].toarray(),
                    forcing[rows],
                    entry_errors=errors[rows][:, members].toarray(),
                    inherited_allowance=allowance[rows],
                )
                answer[members] = value
            else:
                directions = np.eye(len(members))
            yield members, directions

    directions = _sparse_block_columns(matrix.shape[1], blocks())
    return answer, directions, components


def _component_weighted(matrix, forcing, answer, bars, components):
    moved = answer.copy()
    for members, rows in components:
        if len(rows):
            moved[members] = _weighted_by_row_bars(
                matrix[rows][:, members].toarray(),
                forcing[rows],
                answer[members],
                bars[rows],
            )
    return moved


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
    # A supplied state range is computed in floating point. Its lift equations
    # enforce that range, but are not original exact-rank authority: rounding a
    # rational null direction can spuriously pin every circulating current.
    # Original Q/physical rows still prove their exact zeros independently.
    original_rows = matrix.shape[0] - n if state_directions is not None else matrix.shape[0]
    unanchored, _ = homogeneous_zero_coordinates(matrix[:original_rows], rhs[:original_rows])
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
    # Original homogeneous rows have already proved these coordinates exactly
    # zero. They are fixed coordinates, not degrees of freedom of the Schur
    # solve or its weighted refinement. Allowing them to move and resetting
    # them only after reconstruction breaks the neighboring original rows.
    remaining_columns = remaining_columns[unanchored[remaining_columns]]
    component_mode = len(remaining_columns) > 512
    block = matrix[selected_rows][:, selected_columns].tocsr()
    raw_solve = _triangular_joint_solver(matrix, mass, stored_rows, stored_positions, indices, differential)

    def solve(forcing):
        return refine_factored_solution(block, forcing, raw_solve(forcing), raw_solve)

    def validate_original(correction):
        scale = np.zeros_like(rhs) if row_scale is None else row_scale
        rows = np.arange(original_rows)
        bars = allowance[rows] + _row_bars(matrix, rows, correction, rhs, scale, tolerance)
        if np.any(abs(np.asarray(matrix[rows] @ correction).ravel() - rhs[rows]) > bars):
            msg = 'joint stored state contradicts actual original rows after exact homogeneous anchors'
            raise ValueError(msg)

    if not len(remaining_columns):
        correction = np.zeros(matrix.shape[1])
        correction[selected_columns] = solve(rhs[selected_rows])
        if state_directions is not None:
            absent = np.asarray(state_directions.getnnz(axis=1)).ravel() == 0
            correction[:n][absent] = 0.0
        correction[~unanchored] = 0
        validate_original(correction)
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
        value = response[:, column].toarray().ravel() if component_mode else response[:, column]
        residual = column_forcing(column) - block @ value
        return abs(raw_solve(residual)) if np.any(residual) else residual

    if component_mode:
        response = sp.hstack(
            [sp.csr_matrix(column_response(column)[:, None]) for column in range(len(remaining_columns))],
            format='csr',
        )
    else:
        response = np.column_stack([column_response(column) for column in range(len(remaining_columns))])
    left = matrix[remaining_rows][:, selected_columns]
    direct = matrix[remaining_rows][:, remaining_columns]
    schur = direct - sp.csr_matrix(left @ response)
    schur.eliminate_zeros()
    # The Schur border is COMPUTED, and a solid body's net-current row is exactly
    # dependent on its hidden row: in the border it is a cancellation of terms many
    # orders larger than what is left. Its entries' own error seeds the eliminator, or
    # it pivots on that remainder (case_a step 1: body rows at 1e-15 of their 5.2
    # operands, subsystem cond 1e18). Formation rounding is gamma_k over the summed
    # |terms|; the responses' forward error is the image of their next refinement
    # correction, an ESTIMATE as in `_driven_allowance_image`.
    count = np.asarray(sp.csr_matrix(left).getnnz(axis=1)).ravel() + 1.0
    unit = np.finfo(float).eps / 2
    gamma = count * unit / (1 - count * unit)
    if component_mode:
        errors = sp.hstack(
            [sp.csr_matrix(response_error(column)[:, None]) for column in range(len(remaining_columns))],
            format='csr',
        )
    else:
        errors = np.column_stack([response_error(column) for column in range(len(remaining_columns))])
    abs_left = abs(sp.csr_matrix(left))
    schur_errors = (
        sp.diags(gamma) @ (abs(direct) + abs_left @ abs(response)) + abs_left @ errors
        if component_mode
        else gamma[:, None] * np.asarray(abs(direct) + abs_left @ abs(response)) + abs_left @ errors
    )
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
    # is far smaller (case_a: the field rows' remainders summed onto a body row at 1.3x).
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
    if row_scale is not None:
        # A retained row whose own pivot CANCELS in the elimination is exactly a sum of eliminated rows
        # (a solid bar's net current is the sum of its nodes' field rows). Each eliminated row's residual
        # is EVALUATED to its rounding floor gamma_k |J||u|, and a differential row's allowance is zero
        # (its residual is the rate), so the sum reached the retained row with no bound at all. It
        # inherits their evaluation floors through the same adjoint; a residual above a field row's
        # floor, or an offset of the row's own, still exceeds it.
        cancelled = _cancelled_pivot_rows(direct, schur, schur_errors)
        inherited += _driven_allowance_image(
            block, left, _evaluation_floor(matrix, selected_rows, row_scale), cancelled & (abs(forcing) > inherited)
        )
    supported = np.asarray(schur.getnnz(axis=1)).ravel() != 0
    if np.any(abs(forcing[~supported]) > inherited[~supported]):
        msg = 'joint stored state contradicts actual eliminated original rows'
        raise ValueError(msg)
    components = None
    if component_mode:
        answer, directions, components = _component_affine(
            schur[supported],
            forcing[supported],
            schur_errors[supported],
            inherited[supported],
        )
    elif supported.any():
        answer, directions = border_affine_space(
            schur[supported].toarray(),
            forcing[supported],
            inherited_allowance=inherited[supported],
            entry_errors=schur_errors[supported],
        )
    else:
        # No equations constrain this remaining chart. Its zero representative
        # and full free directions are exact; no artificial border is needed.
        answer = np.zeros(len(remaining_columns))
        directions = np.eye(len(remaining_columns))
    if row_scale is not None:
        # A DEPENDENT border row set is consistent only to its data's rounding, and the
        # elimination leaves that inconsistency on whichever row it did not pivot. case_a
        # admission 80: the integrator's stored target meets a solid body's algebraic
        # net-current constraint to ~2e-15 of that constraint's operands, and the whole of it
        # landed on the body row, whose own operands are 3e5x smaller -- 6e-10 of its scale,
        # every iteration, beyond any refinement. Each row is weighted by its own bar.
        trial = np.zeros(matrix.shape[1])
        trial[remaining_columns] = answer
        trial[selected_columns] = solve(rhs[selected_rows] - kept - coupling @ answer)
        rows = remaining_rows[supported]
        bars = _row_bars(matrix, rows, trial, rhs, row_scale, tolerance)
        dense = schur[supported] if component_mode else schur[supported].toarray()
        answer = (
            _component_weighted(dense, forcing[supported], answer, bars, components)
            if component_mode
            else _weighted_by_row_bars(dense, forcing[supported], answer, bars)
        )
        # A retained row whose Schur row is too weak to share its residual with the other
        # retained rows (case_a admission 338: a body row at 2e-5 of its neighbours, 2.6x its
        # bar) takes it from the ELIMINATED rows, through the elimination's adjoint, each in
        # units of its own bar -- the same weighted least squares, one level down.
        # Judged on the ACTUAL original rows, not the Schur model: the elimination's own
        # error is what put the residual there.
        # Every row is judged at the trial it is judged ON: the bars above were formed at the first
        # trial, where a row with no operands has a zero bar, and the weighted answer then leaves
        # rounding there -- 16203 rows "over" at case_a 90 deg t=0.0011921, so this share never ran.
        # The rows already shared stay in the set, driven to zero together: shared one at a time,
        # the residual of a solid body's dependent net-current row only moved to its sibling bodies.
        selected_bars = _row_bars(matrix, selected_rows, trial, rhs, row_scale, tolerance)
        shared = np.zeros(len(rows), dtype=bool)
        for _ in range(4):
            trial[remaining_columns] = answer
            trial[selected_columns] = solve(rhs[selected_rows] - kept - coupling @ answer)
            defect = np.asarray(matrix[rows] @ trial).ravel() - rhs[rows]
            over = abs(defect) > _row_bars(matrix, rows, trial, rhs, row_scale, tolerance)
            # A dependent residual is a few weak rows; more rows than border columns is no such set.
            if not over.any() or np.count_nonzero(over | shared) > len(remaining_columns):
                break
            shared |= over
            over = np.flatnonzero(shared)
            if component_mode and len(over) * block.shape[0] > 512**2:
                msg = 'joint correction requires bounded actual adjoint-sharing work'
                raise NotImplementedError(msg)
            shift = _eliminated_share(block, left[np.flatnonzero(supported)[over]], selected_bars, defect[over])
            if shift is None:
                break
            kept = kept + shift
            forcing = forcing + left @ solve(shift)
            answer = (
                _component_weighted(dense, forcing[supported], answer, bars, components)
                if component_mode
                else _weighted_by_row_bars(dense, forcing[supported], answer, bars)
            )
    correction = np.zeros(matrix.shape[1])
    correction[remaining_columns] = answer
    correction[selected_columns] = solve(rhs[selected_rows] - kept - coupling @ answer)
    if component_mode:
        selectors = sp.eye(matrix.shape[1], format='csr')
        full_directions = selectors[:, remaining_columns] @ directions - selectors[:, selected_columns] @ (
            response @ directions
        )
    else:
        full_directions = np.zeros((matrix.shape[1], directions.shape[1]))
        full_directions[remaining_columns] = directions
        full_directions[selected_columns] = -response @ directions
    if state_directions is not None:
        # D describes homogeneous state freedom around an admitted numerical
        # particular. Replacing the ORIGINAL physical solve by D*alpha erases
        # particular repairs which its original rows need (notably a tiny
        # winding row beside a driven one). Keep the solved coordinates and
        # their original F/Q/G validation; no computed lift becomes a new bar.
        absent = np.asarray(state_directions.getnnz(axis=1)).ravel() == 0
        correction[:n][absent] = 0.0
        # Homogeneous directions remain a separate explicitly supplied chart;
        # do not infer state uniqueness from the particular's smallness.
        if component_mode:
            full_directions = sp.vstack(
                (state_directions @ full_directions[2 * n :], full_directions[n:]), format='csr'
            )
        else:
            full_directions[:n] = state_directions @ full_directions[2 * n :]
    # Reconstruct coordinates proved zero by original Q/constraint rows, not
    # cancellation of driven sparse responses. The same exact lift applies to
    # every homogeneous direction; all original equations remain validators.
    correction[~unanchored] = 0
    if component_mode:
        full_directions = (sp.diags(unanchored.astype(float)) @ full_directions).tocsr()
        full_directions.eliminate_zeros()
        unresolved = full_directions[:n].nnz != 0
    else:
        full_directions[~unanchored] = 0
        unresolved = np.any(full_directions[:n] != 0)
    if unresolved:
        msg = 'joint stored-coordinate equations require additional physical state authority'
        raise ValueError(msg)
    if state_directions is not None or not supported.any():
        validate_original(correction)
    if component_mode and state_directions is not None:
        chart = np.arange(original_rows, matrix.shape[0])
        scale = np.zeros_like(rhs) if row_scale is None else row_scale
        if np.any(
            abs(np.asarray(matrix[chart] @ correction).ravel() - rhs[chart])
            > allowance[chart] + _row_bars(matrix, chart, correction, rhs, scale, tolerance)
        ):
            msg = 'joint correction contradicts original augmented state chart rows'
            raise ValueError(msg)
    # What each original row may keep: its own certified bound, and on a retained row
    # also what elimination carried into it (slack image plus driven rounding image).
    admitted = allowance.copy()
    admitted[remaining_rows] = inherited
    return correction, full_directions[n : 2 * n], admitted


def _row_bars(matrix, rows, trial, rhs, row_scale, tolerance):
    """Each row's own relative bar; a noise row is judged as zero; a row with no operands stays exact.

    A row whose operands are all rounding (forcing 0, ``|row| . |trial|`` at 1e-46 beside rows at
    the system's scale) has no scale of its own, and its relative bar is noise judging noise --
    MEASURED 2026-10-14 on the moving-band networks: residual 1.1e-44 against bars of 1e-56.
    :func:`~numerics_lab.linear.resolution.system_resolution` owns that floor.
    """
    operands = np.asarray(abs(matrix) @ abs(trial)).ravel() + abs(rhs) + row_scale
    own = operands[rows]
    return np.where(own > 0, row_bar(tolerance * own, own, system_resolution(operands), rhs[rows]), 0.0)


def _weighted_by_row_bars(schur, forcing, answer, bars):
    """Move ``answer`` to the least squares of the border rows, each in units of its own bar.

    Independent rows are already met; only a dependent set's residual moves, onto the rows whose
    bars can hold it. A row whose bar is zero (or below the normal range) is EXACT: it is kept as an
    equality constraint, never divided by. Free columns (no entry) are not touched.
    """
    live = np.flatnonzero(np.any(schur != 0, axis=0))
    residual = forcing - schur @ answer
    exact = ~(bars >= np.finfo(float).tiny)
    # An answer every row already accepts is not a least-squares seed: moving it only writes the
    # solve's rounding into coordinates that are exactly zero (the at-rest switched network: 1e-58).
    if not live.size or (np.all(residual[exact] == 0) and np.all(abs(residual[~exact]) <= bars[~exact])):
        return answer
    matrix = schur[:, live]
    step = np.zeros(len(live))
    # The exact rows' row space is held by its ECONOMY basis ``vt[:rank]`` (rank <= rows); the
    # null space is the implicit projector ``I - V V^T``, never an explicit live x live basis:
    # live spans the whole field (tens of thousands of columns), and the explicit square factor
    # was a 60 GB allocation on the transformer march (2026-10-14).
    row_space = np.zeros((0, len(live)))
    if np.any(exact):
        # Exact rows: the minimum-norm step that meets them, then move only in their null space.
        u, sigma, vt = np.linalg.svd(matrix[exact], full_matrices=False)
        rank = int(np.sum(sigma > sigma[0] * np.finfo(float).eps * max(matrix[exact].shape))) if sigma.size else 0
        step = vt[:rank].T @ ((u[:, :rank].T @ residual[exact]) / sigma[:rank])
        row_space = vt[:rank]
    soft = ~exact
    if np.any(soft) and row_space.shape[0] < len(live):
        projected = matrix[soft] - (matrix[soft] @ row_space.T) @ row_space
        weighted = projected / bars[soft, None]
        norm = np.linalg.norm(weighted, axis=0)
        usable = norm > 0
        if np.any(usable):
            target = (residual[soft] - matrix[soft] @ step) / bars[soft]
            reduced, *_ = np.linalg.lstsq(weighted[:, usable] / norm[usable], target, rcond=None)
            move = np.zeros(len(live))
            move[usable] = reduced / norm[usable]
            step = step + move - row_space.T @ (row_space @ move)
    moved = answer.copy()
    moved[live] += step
    return moved


def _eliminated_share(block, left, bars, residual):
    """Forcing shifts on the eliminated rows that carry ``residual`` off the given retained rows.

    The retained rows' Schur forcing moves by ``-y_i . shift`` with ``B^T y_i = left_i^T``; the
    shift is the minimum of ``sum (shift_j / bar_j)^2`` meeting ``Y shift = residual``. A row with
    a zero bar is exact and takes nothing. ``None`` when the adjoint cannot be formed.
    """
    transposed = sp.csr_matrix(block.T)
    try:
        normalized, _ = normalize_sparse_rows(transposed, np.zeros(transposed.shape[0]))
        factor = splu(normalized.tocsc())
    except (RuntimeError, ValueError):
        return None

    def correction(defect):
        return factor.solve(normalize_sparse_rows(transposed, defect)[1])

    weights = np.where(bars >= np.finfo(float).tiny, bars, 0.0) ** 2
    adjoints = []
    for row in range(left.shape[0]):
        target = np.asarray(left[row].toarray()).ravel()
        adjoint = correction(target)
        for _ in range(4):
            adjoint = adjoint + correction(target - transposed @ adjoint)
        if not np.all(np.isfinite(adjoint)):
            return None
        adjoints.append(adjoint)
    adjoints = np.array(adjoints)
    gram = (adjoints * weights) @ adjoints.T
    multipliers, *_ = np.linalg.lstsq(gram, residual, rcond=None)
    shift = weights * (adjoints.T @ multipliers)
    return shift if np.all(np.isfinite(shift)) else None


def _cancelled_pivot_rows(direct, schur, schur_errors):
    """Rows of the Schur border whose direct entry the elimination cancels below its own formation error.

    The reliability factor is :func:`~numerics_lab.linear.border_rank.homogeneous_zero_coordinates`'s,
    so a row is "cancelled" exactly when that eliminator would refuse to pivot on the entry.
    """
    entries = sp.csr_matrix(direct).tocoo()
    bound = 16 * max(schur.shape)
    if sp.issparse(schur_errors):
        values = np.asarray(schur[entries.row, entries.col]).ravel()
        errors = np.asarray(schur_errors[entries.row, entries.col]).ravel()
    else:
        dense = schur.toarray() if sp.issparse(schur) else np.asarray(schur)
        values = dense[entries.row, entries.col]
        errors = np.asarray(schur_errors, dtype=float)[entries.row, entries.col]
    lost = abs(values) <= bound * errors
    rows = np.zeros(schur.shape[0], dtype=bool)
    rows[entries.row[lost]] = True
    return rows


def _evaluation_floor(matrix, rows, row_scale):
    """``gamma_k |J||u|`` per original row: the rounding of evaluating a k-term residual row."""
    count = np.asarray(sp.csr_matrix(matrix)[rows].getnnz(axis=1)).ravel() + 1.0
    gamma = count * np.finfo(float).eps / (1 - count * np.finfo(float).eps)
    return gamma * np.asarray(row_scale, dtype=float)[rows]


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
    asked: measured on case_a, rows of the mixed-unit block with a two-entry support keep their
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
        # refuses this mixed-unit adjoint (cond_1 ~ 1e25 on case_a) although its refinement
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
    stored_allowance=None,
    differentiated_allowance=None,
):
    """Eliminate stored state then stored rate; reveal only the actual free border.

    ``residual_allowance`` is the certified rounding bound of ``residual`` per row; it
    travels through the sparse elimination and every original row may keep its image.
    """
    n = mass.shape[0]
    derivative_bound = (
        np.zeros(algebraic.shape[0]) if differentiated_allowance is None else np.asarray(differentiated_allowance)
    )
    if (
        derivative_bound.shape != (algebraic.shape[0],)
        or derivative_bound.dtype.kind not in 'fiu'
        or not np.isfinite(derivative_bound).all()
        or np.any(derivative_bound < 0)
    ):
        msg = 'differentiated allowance requires finite nonnegative original hidden-row bounds'
        raise ValueError(msg)
    storage_bound = np.zeros(len(stored_rows)) if stored_allowance is None else np.asarray(stored_allowance)
    if (
        storage_bound.shape != stored_rows.shape
        or storage_bound.dtype.kind not in 'fiu'
        or not np.isfinite(storage_bound).all()
        or np.any(storage_bound < 0)
        or (
            stored_allowance is not None
            and any(isinstance(value, (bool, np.bool_)) for value in np.asarray(stored_allowance, dtype=object).flat)
        )
    ):
        msg = 'stored increment allowance requires finite nonnegative original row bounds'
        raise ValueError(msg)
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
    if sparse_state_indices is None and np.any(storage_bound):
        msg = 'certified stored increments require the declared sparse storage elimination'
        raise NotImplementedError(msg)
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
        allowance = np.concatenate((
            np.zeros(n) if residual_allowance is None else residual_allowance,
            storage_bound,
            derivative_bound,
        ))
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
    inherited = np.concatenate((np.zeros(n + len(stored_rows)), derivative_bound))
    answer, directions = border_affine_space(reduced / column_norm, forcing, inherited_allowance=inherited / row_norm)
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
        admitted=inherited,
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
        or np.any(
            abs(state_defect)
            > row_bar(tolerance * state_scale, state_scale, system_resolution(state_scale), residual) + admitted[:n]
        )
    ):
        msg = 'joint stored state contradicts actual state/rate equations'
        raise ValueError(msg)

    def _holds(candidate, bound):
        defect = matrix @ candidate - rhs
        equation_scale = np.asarray(abs(matrix) @ abs(candidate)).ravel() + abs(rhs)
        # The residual was evaluated at the actual state: a row whose coordinates
        # are all fixed by storage keeps that evaluation's size as its only scale,
        # the same scale the nonlinear owner admits its algebraic state rows by.
        equation_scale[:n] += np.asarray(abs(jacobian) @ abs(state)).ravel()
        stored = matrix[n : 2 * n - algebraic.shape[0], :n]
        equation_scale[n : 2 * n - algebraic.shape[0]] += np.asarray(abs(stored) @ abs(state)).ravel()
        bar = row_bar(tolerance * equation_scale, equation_scale, system_resolution(equation_scale), rhs)
        return bool(np.all(np.isfinite(defect)) and np.all(abs(defect) <= bar + bound))

    unique = np.ones(n, dtype=bool)
    unique[~_rate_uniqueness(rate_directions)] = False
    # A joint correction that already meets EVERY original row at the bare bar needs no rate
    # repair: re-solving its rates re-makes the dependent body rows' rounding image (case_a
    # admission 338: the border met body 5 exactly, the re-solve put it at 2.1x its bar).
    if _holds(correction, 0.0):
        return correction, unique
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
            differentiated_allowance=admitted[2 * n - algebraic.shape[0] :],
        )
        correction[columns] += state_step
    except ValueError as error:
        msg = f'joint stored state contradicts actual state/rate equations: {error}'
        raise ValueError(msg) from error
    if not _holds(correction, admitted):
        msg = 'joint stored state contradicts actual state/rate equations'
        raise ValueError(msg)
    unique = np.ones(n, dtype=bool)
    unique[~_rate_uniqueness(rate_directions)] = False
    return correction, unique & rate_unique
