"""Small-border elimination with local roundoff bounds, not a global mixed-index SVD cutoff."""

import numpy as np
import scipy.sparse as sp

__all__ = ['border_affine_space', 'homogeneous_zero_coordinates']


def homogeneous_zero_coordinates(matrix, rhs, *, row_order=None, entry_errors=None):
    """Prove exact zero coordinates from original homogeneous support alone.

    A homogeneous row whose remaining support is ONE entry proves that coordinate zero -- if the entry is
    one. ``entry_errors`` bounds a COMPUTED matrix's entries (a Schur border's formation error), and an
    entry inside ``16 max(shape)`` of its bound (the eliminator's own reliability test) is a possible zero:
    the row may be ``0 = 0``, so it proves nothing and its column stays free.
    """
    original = sp.csr_matrix(matrix, copy=True)
    original.sum_duplicates()
    original.eliminate_zeros()
    rhs = np.asarray(rhs)
    if rhs.shape != (original.shape[0],) or not np.isfinite(rhs).all() or not np.isfinite(original.data).all():
        msg = 'homogeneous anchors require finite original equations'
        raise ValueError(msg)
    if entry_errors is None:
        reliable = None
    else:
        errors = np.asarray(entry_errors, dtype=float)
        if errors.shape != original.shape:
            msg = 'homogeneous anchors require entry error bounds aligned with the equations'
            raise ValueError(msg)
        rows_of = np.repeat(np.arange(original.shape[0]), np.diff(original.indptr))
        bound = 16 * max(original.shape)
        reliable = np.abs(original.data) > bound * errors[rows_of, original.indices]
    remaining = np.ones(original.shape[1], dtype=bool)
    consumed = set()
    order = range(original.shape[0]) if row_order is None else row_order
    while True:
        changed = False
        for row in order:
            if row in consumed or rhs[row] != 0:
                continue
            start, end = original.indptr[row], original.indptr[row + 1]
            columns = original.indices[start:end]
            live = remaining[columns]
            support = columns[live]
            if len(support) == 1 and (reliable is None or reliable[start:end][live][0]):
                remaining[support[0]] = False
                consumed.add(int(row))
                changed = True
        if not changed:
            return remaining, consumed


def border_affine_space(matrix, rhs, *, mandatory_rows=(), inherited_allowance=None, entry_errors=None):
    """Return a representative and explicit free directions of actual border equations.

    Re-equilibrate remaining rows after each elimination. A weak independent
    index constraint is not discarded just because another row has different
    units. Cancellation-only rows are removed using propagated arithmetic
    error bounds before any normalization can turn their noise into authority.
    ``entry_errors`` bounds a COMPUTED matrix's own entries (a Schur border's
    formation error); it seeds that propagation, so an entry inside its bound
    is never a pivot. Omitted, the entries are taken as exact.
    """
    matrix, rhs = np.asarray(matrix, dtype=float).copy(), np.asarray(rhs, dtype=float).copy()
    if (
        matrix.ndim != 2
        or not all(matrix.shape)
        or matrix.shape[1] > 512
        or rhs.shape != (matrix.shape[0],)
        or not np.isfinite(matrix).all()
        or not np.isfinite(rhs).all()
    ):
        msg = 'border affine equations require a finite bounded matrix and aligned forcing'
        raise ValueError(msg)
    rows, columns = matrix.shape
    allowance = np.zeros(rows) if inherited_allowance is None else np.asarray(inherited_allowance)
    if (
        allowance.shape != (rows,)
        or allowance.dtype.kind not in 'fiu'
        or any(isinstance(value, (bool, np.bool_)) for value in np.asarray(allowance, dtype=object).flat)
        or (
            inherited_allowance is not None
            and any(isinstance(value, (bool, np.bool_)) for value in np.asarray(inherited_allowance, dtype=object).flat)
        )
        or not np.isfinite(allowance).all()
        or np.any(allowance < 0)
    ):
        msg = 'inherited border allowance requires finite nonnegative actual row bounds'
        raise ValueError(msg)
    mandatory_rows = np.asarray(mandatory_rows)
    if (
        mandatory_rows.ndim != 1
        or (mandatory_rows.size and mandatory_rows.dtype.kind not in 'iu')
        or len(set(mandatory_rows.tolist())) != len(mandatory_rows)
        or np.any(mandatory_rows < 0)
        or np.any(mandatory_rows >= rows)
    ):
        msg = 'mandatory border constraints require unique actual equation rows'
        raise ValueError(msg)
    mandatory_rows = mandatory_rows.astype(int)
    seeded = np.zeros_like(matrix) if entry_errors is None else np.asarray(entry_errors, dtype=float)
    if seeded.shape != matrix.shape or not np.isfinite(seeded).all() or np.any(seeded < 0):
        msg = 'border entry error bounds require a finite nonnegative aligned matrix'
        raise ValueError(msg)
    # Original homogeneous singleton equations prove exact zeros. Propagate
    # only those proven columns, before driven-row arithmetic can contaminate
    # the representative or its free directions.
    ordered_rows = np.concatenate((mandatory_rows, np.setdiff1d(np.arange(rows), mandatory_rows)))
    remaining, consumed = homogeneous_zero_coordinates(matrix, rhs, row_order=ordered_rows, entry_errors=seeded)
    if consumed:
        support_rows = np.any(matrix[:, remaining] != 0, axis=1)
        eliminated_rows = (~support_rows) & np.any(matrix != 0, axis=1)
        if np.any(eliminated_rows & (abs(rhs) > allowance)):
            msg = 'joint stored state contradicts actual state/rate equations at exact homogeneous anchors'
            raise ValueError(msg)
        pending = [int(row) for row in mandatory_rows if row not in consumed]
        if any(not support_rows[row] for row in pending):
            msg = 'mandatory border constraints must retain independent actual equation authority'
            raise ValueError(msg)
        answer = np.zeros(columns)
        if not np.any(remaining):
            return answer, np.zeros((columns, 0))
        retained = np.flatnonzero(support_rows)
        if not retained.size:
            free = np.flatnonzero(remaining)
            directions = np.zeros((columns, len(free)))
            directions[free, np.arange(len(free))] = 1
            return answer, directions
        positions = {int(row): index for index, row in enumerate(retained)}
        reduced_answer, reduced_directions = border_affine_space(
            matrix[retained][:, remaining],
            rhs[retained],
            mandatory_rows=tuple(positions[row] for row in pending),
            inherited_allowance=allowance[retained],
            entry_errors=seeded[retained][:, remaining],
        )
        answer[remaining] = reduced_answer
        directions = np.zeros((columns, reduced_directions.shape[1]))
        directions[remaining] = reduced_directions
        return answer, directions
    order = np.concatenate((mandatory_rows, np.setdiff1d(np.arange(rows), mandatory_rows)))
    matrix, rhs = matrix[order], rhs[order]
    original_matrix, original_rhs = matrix.copy(), rhs.copy()
    row_ids = np.arange(rows)
    errors = seeded[order].copy()
    eps = np.finfo(float).eps
    bound = 16 * max(matrix.shape)
    pivots = []
    cursor = 0
    available = list(range(columns))
    while available and cursor < rows:
        active = matrix[cursor:, available]
        uncertainty = errors[cursor:, available]
        reliable = np.abs(active) > bound * uncertainty
        magnitude = np.max(np.where(reliable, np.abs(active), 0), axis=1)
        magnitude[magnitude == 0] = 1
        ratios_all = np.where(reliable, np.abs(active) / magnitude[:, None], 0)
        eligible = np.flatnonzero(np.max(ratios_all, axis=0) > eps * bound)
        if cursor < len(mandatory_rows) and np.max(ratios_all[0]) <= eps * bound:
            msg = 'mandatory border constraints must retain independent actual equation authority'
            raise ValueError(msg)
        if not eligible.size:
            break
        index = int(np.argmax(ratios_all[0])) if cursor < len(mandatory_rows) else int(eligible[0])
        column = available.pop(index)
        ratios = ratios_all[:, index]
        chosen = cursor if cursor < len(mandatory_rows) else cursor + int(np.argmax(ratios))
        if cursor >= len(mandatory_rows):
            # Exact homogeneous witnesses should not inherit driven-row cancellation.
            homogeneous = np.flatnonzero((rhs[cursor:] == 0) & (ratios > eps * bound))
            if homogeneous.size:
                chosen = cursor + int(homogeneous[np.argmax(ratios[homogeneous])])
        matrix[[cursor, chosen]] = matrix[[chosen, cursor]]
        errors[[cursor, chosen]] = errors[[chosen, cursor]]
        rhs[[cursor, chosen]] = rhs[[chosen, cursor]]
        row_ids[[cursor, chosen]] = row_ids[[chosen, cursor]]
        pivot = matrix[cursor, column]
        matrix[cursor] /= pivot
        errors[cursor] = errors[cursor] / abs(pivot) + eps * np.abs(matrix[cursor])
        rhs[cursor] /= pivot
        for row in range(rows):
            if row == cursor:
                continue
            factor = matrix[row, column]
            factor_error = errors[row, column]
            original = matrix[row].copy()
            matrix[row] -= factor * matrix[cursor]
            errors[row] += (
                abs(factor) * errors[cursor]
                + factor_error * np.abs(matrix[cursor])
                + eps * (np.abs(original) + abs(factor) * np.abs(matrix[cursor]))
            )
            rhs[row] -= factor * rhs[cursor]
            matrix[row, column] = 0
        pivots.append(column)
        cursor += 1
        if cursor == rows:
            break
    free_columns = np.setdiff1d(np.arange(columns), pivots)
    answer = np.zeros(columns)
    answer[pivots] = rhs[:cursor]
    if pivots:
        # Rank discovery is not the physical solve: reconstruct from the
        # original selected equations before elimination noise becomes a state.
        selected = row_ids[:cursor]
        subsystem = original_matrix[selected][:, pivots]
        answer[pivots] = np.linalg.solve(subsystem, original_rhs[selected])
        for _ in range(3):
            defect = original_rhs[selected] - original_matrix[selected] @ answer
            if not np.any(defect):
                break
            answer[pivots] += np.linalg.solve(subsystem, defect)
    directions = np.zeros((columns, len(free_columns)))
    directions[free_columns, np.arange(len(free_columns))] = 1
    directions[pivots] = -matrix[:cursor, free_columns]
    # Propagated cancellation uncertainty is the only zero authority. Never
    # erase a genuinely resolved free direction by a global physical cutoff.
    for row, pivot_column in enumerate(pivots):
        numerical_zero = np.abs(directions[pivot_column]) <= bound * errors[row, free_columns]
        directions[pivot_column, numerical_zero] = 0
    return answer, directions
