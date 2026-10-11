"""Sparse actual rate equations and exact disposable-coordinate dependency proofs."""

from fractions import Fraction

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.sparse.linalg import splu


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


def normalize_sparse_rows(matrix, rhs):
    """Scale actual sparse rows without squaring or reciprocal overflow."""
    entries = matrix.tocoo()
    scale = np.zeros(matrix.shape[0])
    np.maximum.at(scale, entries.row, abs(entries.data))
    scale[scale == 0] = 1.0
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


def _refined_rate_values(factor, reduced, rows, forcing):
    values = factor.solve(forcing[rows])
    for _ in range(3):
        if not np.isfinite(values).all():
            msg = 'actual rate equations require a finite representable solution'
            raise ValueError(msg)
        values += factor.solve(forcing[rows] - reduced[rows] @ values)
    if not np.isfinite(values).all():
        msg = 'actual rate refinement requires a finite representable solution'
        raise ValueError(msg)
    return values


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
    if len(selected):
        reduced = normalized[:, selected].tocsr()
        # Row order expresses the caller's numerical pivot priority, never
        # a rank declaration. Actual LU and all original equations prove it.
        pattern = reduced.tocoo()
        priority = sp.csr_matrix(
            (1.0 + pattern.row / (reduced.shape[0] + 1), (pattern.row, pattern.col)),
            shape=reduced.shape,
        )
        try:
            _columns, rows = min_weight_full_bipartite_matching(priority.T)
        except ValueError:
            rows = np.full(len(selected), -1, dtype=int)
        factor = None
        if np.all(rows >= 0):
            try:
                factor = splu(reduced[rows].tocsc())
            except RuntimeError:
                factor = None
        if factor is None:
            if matrix.shape[1] > 512:
                msg = 'singular field-size rate equations require an owned sparse subsystem, not unbounded exact elimination'
                raise ValueError(msg)
            selected, affected = _independent_exact_columns(matrix, np.flatnonzero(nonzero), dependent_start=0)
            reduced = normalized[:, selected].tocsr()
            rows = _independent_exact_rows(matrix[:, selected], np.array([], dtype=int))
            try:
                factor = splu(reduced[rows].tocsc())
            except RuntimeError as error:
                msg = 'actual rate equations lack a nonsingular sparse physical subsystem'
                raise ValueError(msg) from error
        answer[selected] = _refined_rate_values(factor, reduced, rows, forcing)
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
        answer[selected] = _refined_rate_values(factor, reduced, rows, forcing)
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
