"""Sparse bulk plus declared small-border mass rank decomposition."""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import scipy.sparse as sp
from scipy.linalg import qr
from scipy.sparse.linalg import splu

from numerics_lab.linear.resolution import FLOOR_SAFETY

__all__ = ['StorageRows', 'free_pinned_columns', 'storage_border_columns', 'storage_border_rows']


@dataclass(frozen=True)
class StorageRows:
    stored_rows: np.ndarray
    algebraic_rows: sp.csr_matrix


def _exact_border_dependencies(matrix):
    """Rank only the declared sparse border, with exact binary-float witnesses.

    Elimination is exact (rationals), but the ROWS are assembled floats: a stamped diagonal such as
    ``fl(Cgd + Cgs)`` sits one rounding off the exact sum its off-diagonals cancel. An entry left by
    elimination at or below the rounding of the operands summed into it (``FLOOR_SAFETY * eps`` of
    their absolute sum) is that assembly rounding, not storage, and is eliminated as zero. An entry
    that is small but not a cancellation keeps its full relative standing, so a tiny real capacitor
    is never read as noise.
    """
    floor = FLOOR_SAFETY * float(np.finfo(float).eps)
    basis, independent, dependencies = {}, [], []
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row : row + 2]
        vector = {
            int(column): Fraction(float(value))
            for column, value in zip(matrix.indices[start:end], matrix.data[start:end], strict=True)
            if value
        }
        operands = {column: abs(float(value)) for column, value in vector.items()}
        witness = {row: Fraction(1)}
        while vector:
            pivot = min(vector)
            if pivot not in basis:
                divisor = vector[pivot]
                basis[pivot] = (
                    {index: value / divisor for index, value in vector.items()},
                    {index: value / divisor for index, value in witness.items()},
                    {index: value / abs(float(divisor)) for index, value in operands.items()},
                )
                independent.append(row)
                break
            previous, provenance, magnitude = basis[pivot]
            multiplier = vector[pivot]
            for index, value in previous.items():
                vector[index] = vector.get(index, Fraction(0)) - multiplier * value
                operands[index] = operands.get(index, 0.0) + abs(float(multiplier)) * magnitude[index]
                if abs(float(vector[index])) <= floor * operands[index]:
                    del vector[index]
                    del operands[index]
            for index, value in provenance.items():
                witness[index] = witness.get(index, Fraction(0)) - multiplier * value
                if not witness[index]:
                    del witness[index]
        if not vector:
            dependencies.append(witness)
    null = np.zeros((len(dependencies), matrix.shape[0]))
    for row, witness in enumerate(dependencies):
        # Row scaling is arbitrary: normalize exactly before converting, so
        # extreme but finite original storage units do not overflow the basis.
        scale = max(map(abs, witness.values()))
        for column, value in witness.items():
            coefficient = float(value / scale)
            if not np.isfinite(coefficient) or coefficient == 0:
                msg = 'exact storage dependency has no representable finite null witness'
                raise ValueError(msg)
            null[row, column] = coefficient
    return np.asarray(independent, dtype=int), null


def storage_border_rows(mass, *, bulk_indices, border_rows):
    """Rank a declared invertible bulk block and its remaining border rows.

    The caller identifies the actual bulk coordinates/rows and the small border;
    all other rows must be exactly algebraic. Bulk rows may depend only on bulk
    rates. Dense work is border-by-state, never state-by-state. Singular bulk
    factors are refused, not regularized into invented storage.
    """
    mass = sp.csr_matrix(mass)
    mass.eliminate_zeros()
    bulk, border = np.asarray(bulk_indices), np.asarray(border_rows)
    n = mass.shape[0]
    if (
        mass.shape != (n, n)
        or not np.isfinite(mass.data).all()
        or any(value.ndim != 1 or value.dtype.kind not in 'iu' for value in (bulk, border))
        or len(set(np.concatenate((bulk, border)).tolist())) != len(bulk) + len(border)
        or any(np.any(value < 0) or np.any(value >= n) for value in (bulk, border))
    ):
        msg = 'storage border requires finite square mass and disjoint explicit bulk/border rows'
        raise ValueError(msg)
    remaining = np.setdiff1d(np.arange(n), bulk)
    algebraic_indices = np.setdiff1d(remaining, border)
    if mass[algebraic_indices].nnz or mass[bulk][:, remaining].nnz:
        msg = 'declared storage bulk/border omits actual mass rows or bulk rate dependencies'
        raise ValueError(msg)
    # Dense allocation is only the explicitly declared border width.
    sparse_tail = mass[border][:, remaining]
    supported = np.unique(sparse_tail.indices)
    independent, null_border = _exact_border_dependencies(sparse_tail[:, supported].tocsr())
    independent_border = border[independent]
    if len(bulk):
        try:
            factor = splu(mass[bulk][:, bulk].tocsc())
        except RuntimeError as error:
            msg = 'declared storage bulk is not numerically invertible'
            raise ValueError(msg) from error
        # E D^-1: transpose solve keeps the only dense dimension the border.
        correction = factor.solve(mass[border][:, bulk].T.toarray(), trans='T').T
    else:
        correction = np.zeros((len(border), 0))
    if not np.isfinite(correction).all():
        msg = 'storage bulk elimination requires finite representable operands'
        raise ValueError(msg)
    selectors = sp.eye(n, format='csr')
    algebraic = sp.vstack(
        [
            selectors[algebraic_indices],
            sp.csr_matrix(null_border) @ selectors[border] - sp.csr_matrix(null_border @ correction) @ selectors[bulk],
        ],
        format='csr',
    )
    if not np.isfinite(algebraic.data).all():
        msg = 'storage elimination requires finite original algebraic witnesses'
        raise ValueError(msg)
    return StorageRows(np.concatenate((bulk, independent_border)), algebraic)


def storage_border_columns(mass, *, bulk_indices, stored_rows):
    """Select an actual stored minor with only small-border dense pivot work.

    The caller's independent rows already establish the required rank. Column
    pivoting selects that many candidates, not a new floating rank threshold;
    an actual sparse LU proves the selected minor is nonsingular.
    """
    mass = sp.csr_matrix(mass)
    mass.eliminate_zeros()
    bulk, rows = np.asarray(bulk_indices), np.asarray(stored_rows)
    n = mass.shape[0]
    if (
        mass.shape != (n, n)
        or not np.isfinite(mass.data).all()
        or any(value.ndim != 1 or value.dtype.kind not in 'iu' for value in (bulk, rows))
        or any(len(np.unique(value)) != len(value) for value in (bulk, rows))
        or any(np.any(value < 0) or np.any(value >= n) for value in (bulk, rows))
        or not np.isin(bulk, rows).all()
    ):
        msg = 'stored columns require finite square mass and explicit unique bulk/stored indices'
        raise ValueError(msg)
    if not len(rows):
        return np.array([], dtype=int)
    remaining = np.setdiff1d(np.arange(mass.shape[1]), bulk)
    border_rows = rows[~np.isin(rows, bulk)]
    sparse_tail = mass[border_rows][:, remaining]
    supported = np.unique(sparse_tail.indices)
    if len(supported) < len(border_rows):
        msg = 'declared stored rows lack a nonsingular actual stored column minor'
        raise ValueError(msg)
    tail = sparse_tail[:, supported].toarray()
    _, _, pivots = qr(tail, mode='economic', pivoting=True)
    columns = np.concatenate((bulk, remaining[supported[pivots[: len(border_rows)]]]))
    try:
        splu(mass[rows][:, columns].tocsc())
    except RuntimeError as error:
        msg = 'declared stored rows and bulk require a nonsingular actual stored column minor'
        raise ValueError(msg) from error
    return columns


def free_pinned_columns(columns, stored_mass, pinned_rows):
    """Keep a derivative that a differentiated algebraic row fixes on the border.

    A row of ``algebraic @ jacobian`` supported only on stored columns fixes
    their rates (and, differentiated again, accelerations) directly. Eliminated through storage instead, such a rate is
    recovered by cancellation against a coupled free derivative and inherits
    its roundoff (r = 0 rebuilt as ulps of a 5e7 rate). Exchange it for a free
    column sharing its storage rows; the block LU still refuses a singular one.
    """
    columns = np.array(columns)
    stored_mass = sp.csc_matrix(stored_mass)
    pinned_rows = sp.csr_matrix(pinned_rows)
    pinned_rows.eliminate_zeros()
    for row in range(pinned_rows.shape[0]):
        span = slice(pinned_rows.indptr[row], pinned_rows.indptr[row + 1])
        support, weight = pinned_rows.indices[span], pinned_rows.data[span]
        if not support.size or not np.isin(support, columns).all():
            continue
        pinned = support[np.argmax(abs(weight))]
        position = int(np.flatnonzero(columns == pinned)[0])
        storage = stored_mass[:, pinned].nonzero()[0]
        coupled = abs(stored_mass[storage]).sum(axis=0).A1
        coupled[columns] = 0
        if np.any(coupled):
            columns[position] = int(np.argmax(coupled))
    return columns
