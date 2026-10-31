"""Sparse actual rate equations and exact disposable-coordinate dependency proofs."""

from fractions import Fraction

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, lsqr, splu

from numerics_lab.linear.arithmetic import EXACT_DEFECT_MAX_COLUMNS, rounded_defect

__all__ = ['normalize_sparse_rows', 'sparse_initial_rate']


def _independent_exact_columns(matrix, selected, *, dependent_start, general=True):
    """Quotient only exact sparse dependencies in explicitly disposable coordinates."""
    columns = matrix.tocsc()
    columns.sort_indices()
    if not general:
        representatives = {}
        affected, omitted = set(), set()
        for column in selected[selected >= dependent_start]:
            start, stop = columns.indptr[column : column + 2]
            key = (columns.indices[start:stop].tobytes(), columns.data[start:stop].tobytes())
            if key in representatives:
                affected.update((column, representatives[key]))
                omitted.add(column)
            else:
                representatives[key] = column
        return np.asarray([column for column in selected if column not in omitted], dtype=int), affected
    basis = {}
    affected, omitted = set(), set()
    for column in selected[selected >= dependent_start]:
        start, stop = columns.indptr[column : column + 2]
        vector = {
            int(row): Fraction(float(value))
            for row, value in zip(columns.indices[start:stop], columns.data[start:stop], strict=True)
            if value
        }
        support = {int(column): Fraction(1)}
        while vector:
            pivot = min(vector)
            if pivot not in basis:
                divisor = vector[pivot]
                basis[pivot] = (
                    {row: value / divisor for row, value in vector.items()},
                    {index: value / divisor for index, value in support.items()},
                )
                break
            previous, provenance = basis[pivot]
            multiplier = vector[pivot]
            for target, source in ((vector, previous), (support, provenance)):
                for index, value in source.items():
                    target[index] = target.get(index, Fraction(0)) - multiplier * value
                    if target[index] == 0:
                        del target[index]
        if not vector:
            affected.update(support)
            omitted.add(column)
    return np.asarray([column for column in selected if column not in omitted], dtype=int), affected


def _independent_exact_rows(matrix, mandatory):
    """Actual sparse row authority after LU failure, never a numerical rank cutoff.

    Mandatory stored-coordinate equations are considered first. Exact binary-float
    elimination may fill and grow rational integers; the ordinary sparse LU and
    caller-declared storage-border paths do not pay that fallback cost.
    """
    matrix = matrix.tocsr(copy=True)
    matrix.sum_duplicates()
    basis, selected = {}, []
    mandatory_set = set(mandatory.tolist())
    order = np.concatenate((mandatory, np.setdiff1d(np.arange(matrix.shape[0]), mandatory)))
    for row in order:
        start, stop = matrix.indptr[row : row + 2]
        vector = {
            int(column): Fraction(float(value))
            for column, value in zip(matrix.indices[start:stop], matrix.data[start:stop], strict=True)
            if value
        }
        while vector:
            pivot = min(vector)
            if pivot not in basis:
                divisor = vector[pivot]
                basis[pivot] = {column: value / divisor for column, value in vector.items()}
                selected.append(int(row))
                break
            multiplier = vector[pivot]
            for column, value in basis[pivot].items():
                vector[column] = vector.get(column, Fraction(0)) - multiplier * value
                if vector[column] == 0:
                    del vector[column]
        if not vector and row in mandatory_set:
            msg = 'mandatory stored-coordinate equations lack independent actual row authority'
            raise ValueError(msg)
        if len(selected) == matrix.shape[1]:
            return np.asarray(selected, dtype=int)
    msg = 'initial state/rate equations lack independent actual physical rows'
    raise ValueError(msg)


def _row_scale(matrix):
    """Each actual row's largest coefficient magnitude; an empty row keeps unit scale."""
    entries = matrix.tocoo()
    scale = np.zeros(matrix.shape[0])
    np.maximum.at(scale, entries.row, abs(entries.data))
    scale[scale == 0] = 1.0
    return scale


def normalize_sparse_rows(matrix, rhs):
    """Scale actual sparse rows without squaring or reciprocal overflow."""
    scale = _row_scale(matrix)
    normalized = matrix.copy()
    rows = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
    with np.errstate(over='ignore', under='ignore'):
        normalized.data = matrix.data / scale[rows]
        forcing = rhs / scale
    if (
        not np.isfinite(normalized.data).all()
        or not np.isfinite(forcing).all()
        or np.any((matrix.data != 0) & (normalized.data == 0))
        or np.any((rhs != 0) & (forcing == 0))
    ):
        msg = 'initial equation normalization requires representable finite operators and forcing'
        raise ValueError(msg)
    return normalized, forcing


def _refined_rate_values(factor, matrix, rhs, rows, selected):
    """Solve the factored square choice and refine it against its ORIGINAL rows.

    The factor holds the row-normalized coefficients, which are themselves rounded, so their exact
    solution is not the original rows' -- the defect is measured on the original rows and carried
    through the same row scale. Where it is priced the defect is the EXACT one: a working-precision
    residual rounds an answer's last ulp to zero and leaves it on a dependent row the square choice
    does not hold.
    """
    square, target = matrix[rows].tocsr(), rhs[rows]
    scale = _row_scale(square)
    exact = matrix.shape[1] <= EXACT_DEFECT_MAX_COLUMNS
    full = np.zeros(matrix.shape[1])
    with np.errstate(over='ignore', under='ignore'):
        values = factor.solve(target / scale)
    for _ in range(3):
        if not np.isfinite(values).all():
            msg = 'actual rate equations require a finite representable solution'
            raise ValueError(msg)
        full[selected] = values
        defect = rounded_defect(square, full, target) if exact else target - square @ full
        if not np.any(defect):
            break
        with np.errstate(over='ignore', under='ignore'):
            values = values + factor.solve(defect / scale)
    if not np.isfinite(values).all():
        msg = 'actual rate refinement requires a finite representable solution'
        raise ValueError(msg)
    return values


def _preconditioned_all_row_rate_values(factor, matrix, rhs, reduced, forcing, selected, *, tolerance, allowance):
    """Reuse a REAL square LU for all-row corrections, judged on ORIGINAL rows.

    The iterative kernel supplies no rank authority: this path requires the
    caller's existing full-column square factor. A failed bounded refinement
    returns None, so that factor can retire before the unchanged augmented
    fallback. No normal equations, regularization or zero snapping.
    """
    operator = LinearOperator(
        reduced.shape,
        matvec=lambda value: reduced @ factor.solve(value),
        rmatvec=lambda value: factor.solve(reduced.T @ value, trans='T'),
        dtype=np.float64,
    )
    answer = np.zeros(matrix.shape[1])
    scale = _row_scale(matrix)
    target = forcing
    for _ in range(4):
        solved = lsqr(operator, target, damp=0.0, atol=0.0, btol=0.0, conlim=0.0, iter_lim=128)
        answer[selected] += factor.solve(solved[0])
        defect = matrix @ answer - rhs
        bar = tolerance * (abs(matrix) @ abs(answer) + abs(rhs)) + allowance
        if not np.isfinite(answer).all() or not np.isfinite(defect).all() or not np.isfinite(bar).all():
            return None
        if np.all(abs(defect) <= bar):
            return answer[selected]
        target = -defect / scale
    return None


def _least_squares_rate_values(reduced, forcing):
    """Every row at once: the refined augmented system ``[I R; R' 0][r; x] = [f; 0]``.

    A square row choice can be singular while the whole overdetermined system
    has full column rank and one consistent answer (a dependent net-current
    row per solid body). This system is nonsingular exactly when ``R`` has
    full column rank; a rank-deficient one refuses. The answer is still judged
    by every original row at the caller's unchanged bar.
    """
    rows, size = reduced.shape
    kkt = sp.bmat([[sp.identity(rows), reduced], [reduced.T, None]], format='csc')
    try:
        factor = splu(kkt)
    except RuntimeError as error:
        msg = 'actual field-size rate equations lack full column rank'
        raise ValueError(msg) from error
    target = np.concatenate((forcing, np.zeros(size)))
    solution = factor.solve(target)
    for _ in range(4):
        if not np.isfinite(solution).all():
            msg = 'actual rate equations require a finite representable solution'
            raise ValueError(msg)
        solution += factor.solve(target - kkt @ solution)
    if not np.isfinite(solution).all():
        msg = 'actual rate refinement requires a finite representable solution'
        raise ValueError(msg)
    return solution[rows:]


def _exact_rate_values(matrix, rhs):
    """Reconstruct a bounded original subsystem when floating refinement loses its witnesses."""
    size = matrix.shape[1]
    if matrix.shape != (size, size) or size > 512:
        msg = 'exact rate reconstruction requires a bounded square physical subsystem'
        raise ValueError(msg)
    vectors = []
    forcing = [Fraction(float(value)) for value in rhs]
    for row in range(size):
        vector = {}
        start, end = matrix.indptr[row : row + 2]
        for column, value in zip(matrix.indices[start:end], matrix.data[start:end], strict=True):
            vector[int(column)] = vector.get(int(column), Fraction(0)) + Fraction(float(value))
        vectors.append(vector)
    for column in range(size):
        pivot = next((row for row in range(column, size) if vectors[row].get(column)), None)
        if pivot is None:
            msg = 'actual rate equations lack a nonsingular original physical subsystem'
            raise ValueError(msg)
        vectors[column], vectors[pivot] = vectors[pivot], vectors[column]
        forcing[column], forcing[pivot] = forcing[pivot], forcing[column]
        diagonal = vectors[column][column]
        for row in range(column + 1, size):
            if not vectors[row].get(column):
                continue
            multiplier = vectors[row].pop(column) / diagonal
            forcing[row] -= multiplier * forcing[column]
            for index, value in vectors[column].items():
                if index > column:
                    updated = vectors[row].get(index, Fraction(0)) - multiplier * value
                    if updated:
                        vectors[row][index] = updated
                    else:
                        vectors[row].pop(index, None)
    values = [Fraction(0)] * size
    for row in range(size - 1, -1, -1):
        tail = sum(value * values[column] for column, value in vectors[row].items() if column > row)
        values[row] = (forcing[row] - tail) / vectors[row][row]
    try:
        result = np.asarray([float(value) for value in values])
    except OverflowError as error:
        msg = 'actual exact rate requires a finite representable solution'
        raise ValueError(msg) from error
    if not np.isfinite(result).all():
        msg = 'actual exact rate requires a finite representable solution'
        raise ValueError(msg)
    return result


def _priority_row_matching(pattern):
    """Each column's matched row, the matched rows being the EARLIEST independent ones, else None.

    The rows of a column-perfect matching are the bases of a transversal matroid, so the
    greedy basis -- rows in order, each kept iff an augmenting path admits it -- is the unique
    minimum of any strictly increasing row weight (the former ``1 + row / (m + 1)`` min-weight
    full matching). A failed search leaves its visited columns permanently unreachable to a free
    column (MC21-style dead marking), so failures cost O(nnz) in total.
    """
    rows_of = sp.csr_matrix(pattern, dtype=bool)
    n_rows, n_cols = rows_of.shape
    if n_rows < n_cols:
        return None
    indptr, indices = rows_of.indptr.tolist(), rows_of.indices.tolist()
    mate = [-1] * n_cols
    dead = [False] * n_cols
    stamp = [-1] * n_cols
    look = indptr[:-1]
    matched = 0
    for root in range(n_rows):
        if matched == n_cols:
            break
        stack, via, visited, found = [[root, indptr[root]]], [], [], False
        while stack:
            frame = stack[-1]
            row = frame[0]
            stop = indptr[row + 1]
            cursor = look[row]
            while cursor < stop and mate[indices[cursor]] >= 0:
                cursor += 1
            look[row] = cursor
            if cursor < stop:
                via.append(indices[cursor])
                found = True
                break
            cursor = frame[1]
            while cursor < stop:
                column = indices[cursor]
                cursor += 1
                if dead[column] or stamp[column] == root:
                    continue
                stamp[column] = root
                visited.append(column)
                break
            else:
                stack.pop()
                if via:
                    via.pop()
                continue
            frame[1] = cursor
            via.append(column)
            stack.append([mate[column], indptr[mate[column]]])
        if found:
            for frame, column in zip(stack, via, strict=True):
                mate[column] = frame[0]
            matched += 1
        else:
            for column in visited:
                dead[column] = True
    return np.asarray(mate, dtype=int) if matched == n_cols else None


def sparse_initial_rate(matrix, rhs, *, tolerance=1e-10, residual_allowance=None):
    """Solve actual frozen-state rate equations; dependent rates stay unavailable.

    All rows are retained as physical witnesses. Sparse matching/LU is the
    ordinary path; exact disposable-column and row elimination is only a
    singular-factor or bounded original-validation retry, never a floating numerical rank threshold.
    """
    matrix = sp.csr_matrix(matrix, dtype=float, copy=True)
    rhs = np.asarray(rhs, dtype=float)
    allowance = np.zeros(matrix.shape[0]) if residual_allowance is None else np.asarray(residual_allowance, dtype=float)
    if (
        rhs.shape != (matrix.shape[0],)
        or not np.isfinite(rhs).all()
        or not np.isfinite(matrix.data).all()
        or not np.isfinite(tolerance)
        or tolerance <= 0
        or allowance.shape != rhs.shape
        or not np.isfinite(allowance).all()
        or np.any(allowance < 0)
    ):
        msg = 'initial rate equations require finite actual matrix and aligned right-hand side'
        raise ValueError(msg)
    matrix.eliminate_zeros()
    nonzero = np.asarray(matrix.getnnz(axis=0)).ravel() != 0
    selected, affected = _independent_exact_columns(matrix, np.flatnonzero(nonzero), dependent_start=0, general=False)
    normalized, forcing = normalize_sparse_rows(matrix, rhs)
    answer = np.zeros(matrix.shape[1])
    factor = None
    if len(selected):
        reduced = normalized[:, selected].tocsr()
        # Row order expresses the caller's numerical pivot priority, never
        # a rank declaration. Actual LU and all original equations prove it.
        rows = _priority_row_matching(reduced)
        if rows is None:
            rows = np.full(len(selected), -1, dtype=int)
        if np.all(rows >= 0):
            try:
                factor = splu(reduced[rows].tocsc())
            except RuntimeError:
                factor = None
        if factor is None and matrix.shape[1] > 512:
            answer[selected] = _least_squares_rate_values(reduced, forcing)
        elif factor is None:
            selected, affected = _independent_exact_columns(matrix, np.flatnonzero(nonzero), dependent_start=0)
            reduced = normalized[:, selected].tocsr()
            rows = _independent_exact_rows(matrix[:, selected], np.array([], dtype=int))
            try:
                factor = splu(reduced[rows].tocsc())
            except RuntimeError as error:
                msg = 'actual rate equations lack a nonsingular sparse physical subsystem'
                raise ValueError(msg) from error
        if factor is not None:
            answer[selected] = _refined_rate_values(factor, matrix, rhs, rows, selected)
    defect = matrix @ answer - rhs
    bar = tolerance * (abs(matrix) @ abs(answer) + abs(rhs)) + allowance
    if not np.isfinite(defect).all() or not np.isfinite(bar).all():
        msg_0 = 'actual rate physical validation requires finite defects and bounds'
        raise ValueError(msg_0)
    # A field-size square choice that factored but leaves witnesses violated is
    # retried over every row at once; the bar below is unchanged.
    if np.any(abs(defect) > bar) and factor is not None and matrix.shape[1] > 512:
        values = _preconditioned_all_row_rate_values(
            factor, matrix, rhs, reduced, forcing, selected, tolerance=tolerance, allowance=allowance
        )
        # A retained square factor preconditions every row, never overlaps KKT.
        del factor
        answer[selected] = _least_squares_rate_values(reduced, forcing) if values is None else values
        defect = matrix @ answer - rhs
        bar = tolerance * (abs(matrix) @ abs(answer) + abs(rhs)) + allowance
        if not np.isfinite(defect).all() or not np.isfinite(bar).all():
            msg_0 = 'actual rate physical validation requires finite defects and bounds'
            raise ValueError(msg_0)
    # Row scaling can make an exactly singular subsystem appear to factor.
    # Retry its original dependency witnesses, never a numerical rank cutoff.
    if np.any(abs(defect) > bar) and len(selected) and matrix.shape[1] <= 512:
        selected, affected = _independent_exact_columns(matrix, np.flatnonzero(nonzero), dependent_start=0)
        order = np.concatenate((np.flatnonzero(rhs == 0), np.flatnonzero(rhs != 0)))
        rows = order[_independent_exact_rows(matrix[order][:, selected], np.array([], dtype=int))]
        reduced = normalized[:, selected].tocsr()
        try:
            factor = splu(reduced[rows].tocsc())
        except RuntimeError as error:
            msg = 'actual rate equations lack a nonsingular sparse physical subsystem'
            raise ValueError(msg) from error
        answer[:] = 0.0
        answer[selected] = _refined_rate_values(factor, matrix, rhs, rows, selected)
        defect = matrix @ answer - rhs
        bar = tolerance * (abs(matrix) @ abs(answer) + abs(rhs)) + allowance
        if not np.isfinite(defect).all() or not np.isfinite(bar).all():
            msg = 'actual rate physical validation requires finite defects and bounds'
            raise ValueError(msg)
        if np.any(abs(defect) > bar):
            # Exact original coefficient/RHS reconstruction, not zero snapping:
            # rounded outputs still face every original row and its same bar.
            answer[selected] = _exact_rate_values(matrix[rows][:, selected].tocsr(), rhs[rows])
            defect = matrix @ answer - rhs
            bar = tolerance * (abs(matrix) @ abs(answer) + abs(rhs)) + allowance
            if not np.isfinite(defect).all() or not np.isfinite(bar).all():
                msg = 'actual exact rate physical validation requires finite defects and bounds'
                raise ValueError(msg)
    if np.any(abs(defect) > bar):
        msg = 'actual rate equations contradict original physical constraints'
        raise ValueError(msg)
    unique = nonzero.copy()
    unique[list(affected)] = False
    return answer, unique
