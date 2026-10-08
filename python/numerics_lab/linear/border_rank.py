"""Small-border elimination with local roundoff bounds, not a global mixed-index SVD cutoff."""

import numpy as np


def border_affine_space(matrix, rhs, *, mandatory_rows=()):
    """Return a representative and explicit free directions of actual border equations.

    Re-equilibrate remaining rows after each elimination. A weak independent
    index constraint is not discarded just because another row has different
    units. Cancellation-only rows are removed using propagated arithmetic
    error bounds before any normalization can turn their noise into authority.
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
    order = np.concatenate((mandatory_rows, np.setdiff1d(np.arange(rows), mandatory_rows)))
    matrix, rhs = matrix[order], rhs[order]
    errors = np.zeros_like(matrix)
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
        matrix[[cursor, chosen]] = matrix[[chosen, cursor]]
        errors[[cursor, chosen]] = errors[[chosen, cursor]]
        rhs[[cursor, chosen]] = rhs[[chosen, cursor]]
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
    directions = np.zeros((columns, len(free_columns)))
    directions[free_columns, np.arange(len(free_columns))] = 1
    directions[pivots] = -matrix[:cursor, free_columns]
    # Propagated cancellation uncertainty is the only zero authority. Never
    # erase a genuinely resolved free direction by a global physical cutoff.
    for row, pivot_column in enumerate(pivots):
        numerical_zero = np.abs(directions[pivot_column]) <= bound * errors[row, free_columns]
        directions[pivot_column, numerical_zero] = 0
    return answer, directions
