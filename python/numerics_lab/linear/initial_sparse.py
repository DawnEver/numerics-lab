"""Sparse initial pair over caller-proven independent storage/algebraic rows."""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.sparse.linalg import splu


@dataclass(frozen=True)
class SparseInitialPair:
    state: np.ndarray
    rate: np.ndarray
    rate_is_unique: np.ndarray

    def __post_init__(self) -> None:
        state, rate, unique = np.asarray(self.state), np.asarray(self.rate), np.asarray(self.rate_is_unique)
        if (
            state.ndim != 1
            or rate.shape != state.shape
            or unique.shape != state.shape
            or unique.dtype != np.dtype(bool)
        ):
            msg = 'initial pair requires aligned state/rate and explicit boolean uniqueness authority'
            raise ValueError(msg)
        if not np.isfinite(state).all() or not np.isfinite(rate[unique]).all():
            msg = 'initial pair requires finite physical state and every proven rate'
            raise ValueError(msg)
        for name, value in (('state', state), ('rate', rate), ('rate_is_unique', unique)):
            owned = value.copy()
            owned.setflags(write=False)
            object.__setattr__(self, name, owned)

    @property
    def physical_rate(self):
        """Proven component rates; unknown algebraic coordinates are explicitly unavailable."""
        value = np.where(self.rate_is_unique, self.rate, np.nan)
        value.setflags(write=False)
        return value


def _independent_exact_columns(matrix, selected, *, dependent_start):
    """Quotient only exact sparse dependencies in explicitly disposable coordinates."""
    columns = matrix.tocsc()
    columns.sort_indices()
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


def projected_rate_consistent(jacobian, rate, partial_time, *, algebraic_rows, tolerance):
    """Check each actual differentiated equation in its own projected units.

    Null-basis row scales are arbitrary numerical coordinates, not physics.
    A large constraint cannot hide a contradictory differently scaled row.
    """
    operator = algebraic_rows @ jacobian
    forcing = algebraic_rows @ partial_time
    defect = operator @ rate + forcing
    scale = np.abs(operator) @ np.abs(rate) + np.abs(forcing)
    return bool(np.all(np.abs(defect) <= tolerance * scale))


def sparse_initial_pair(
    jacobian,
    residual,
    mass,
    initial,
    *,
    algebraic_rows,
    stored_rows,
    partial_time,
    stored_increment=None,
    tolerance=1e-10,
):
    """Solve actual index-two equations without dense field-sized nullspaces.

    The caller owns the rank decomposition: ``stored_rows`` selects independent
    rows of mass and ``algebraic_rows`` spans its left nullspace. Free algebraic
    rates are marked, not presented as physical zero derivatives. No free state
    or stored rate is selected by a least-squares/minimum-norm convention.
    """
    initial = np.asarray(initial, dtype=float)
    residual, partial_time = np.asarray(residual, dtype=float), np.asarray(partial_time, dtype=float)
    jacobian, mass, algebraic = map(sp.csr_matrix, (jacobian, mass, algebraic_rows))
    n = initial.size
    stored_rows = np.asarray(stored_rows)
    stored_change = np.zeros(len(stored_rows)) if stored_increment is None else np.asarray(stored_increment)
    if (
        initial.shape != (n,)
        or residual.shape != (n,)
        or partial_time.shape != (n,)
        or jacobian.shape != (n, n)
        or mass.shape != (n, n)
        or algebraic.shape[1] != n
        or stored_rows.ndim != 1
        or stored_rows.dtype.kind not in 'iu'
        or len(set(stored_rows.tolist())) != len(stored_rows)
        or np.any(stored_rows < 0)
        or np.any(stored_rows >= n)
        or len(stored_rows) + algebraic.shape[0] != n
        or stored_change.shape != stored_rows.shape
        or not all(
            np.isfinite(value).all()
            for value in (initial, residual, partial_time, stored_change, jacobian.data, mass.data, algebraic.data)
        )
        or not np.isfinite(tolerance)
        or tolerance <= 0
    ):
        msg = 'sparse initial pair requires finite actual matrices and an explicit complete storage rank decomposition'
        raise ValueError(msg)
    defect = algebraic @ mass
    if np.linalg.norm(defect.data) > tolerance * np.linalg.norm(algebraic.data) * np.linalg.norm(mass.data):
        msg = 'declared algebraic rows are not in the actual mass left nullspace'
        raise ValueError(msg)
    stored = mass[stored_rows]
    matrix = sp.bmat(
        [
            [jacobian, mass],
            [stored, sp.csr_matrix(stored.shape)],
            [sp.csr_matrix(algebraic.shape), algebraic @ jacobian],
        ],
        format='csr',
    )
    rhs = np.concatenate((-residual, stored_change, -algebraic @ partial_time))
    matrix.eliminate_zeros()
    scale = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    scale[scale == 0] = 1.0
    matrix, rhs = sp.diags(1 / scale) @ matrix, rhs / scale
    # Only structurally zero rate columns may be omitted. Their value never
    # affects these equations; expose that fact to outgoing-mode consumers.
    nonzero = np.asarray(matrix.getnnz(axis=0)).ravel() != 0
    if not nonzero[:n].all():
        msg = 'initial state requires additional physical authority'
        raise ValueError(msg)
    selected = np.flatnonzero(nonzero)
    # Identical actual rate columns constrain their sum, not either rate.
    # Remove only this exact sparse dependency; state columns are never
    # quotiented and no numerical rank cutoff invents physical uniqueness.
    selected, duplicate_rates = _independent_exact_columns(matrix, selected, dependent_start=n)
    reduced = matrix[:, selected].tocsr()
    # Storage equations are mandatory, even when their increments are tiny
    # compared with the dynamic RHS. Match them first, then verify that none
    # was discarded by the structural reduction of free algebraic rates.
    mandatory = np.arange(n, n + len(stored_rows))
    row_order = np.concatenate((mandatory, np.arange(n + len(stored_rows), 2 * n), np.arange(n)))
    matched = maximum_bipartite_matching(reduced[row_order], perm_type='row')
    if np.any(matched < 0):
        msg = 'initial state/rate equations require additional physical authority'
        raise ValueError(msg)
    matched_rows = row_order[matched]
    if not np.isin(mandatory, matched_rows).all():
        msg = 'initial state/rate subsystem cannot discard actual stored-coordinate constraints'
        raise ValueError(msg)
    try:
        factor = splu(reduced[matched_rows].tocsc())
    except RuntimeError as error:
        msg = 'initial state/rate equations lack a nonsingular sparse physical constraint subsystem'
        raise ValueError(msg) from error
    pair = np.zeros(2 * n)
    subsystem = reduced[matched_rows]
    answer = factor.solve(rhs[matched_rows])
    # Sparse pivoting mixes a large dynamic RHS with exact stored-coordinate
    # anchors. Refine the actual selected equations before judging those
    # anchors; a tiny spurious state correction is not a physical Q change.
    for _ in range(3):
        answer += factor.solve(rhs[matched_rows] - subsystem @ answer)
    pair[selected] = answer
    bar = tolerance * (np.linalg.norm(matrix.data) * np.linalg.norm(pair) + np.linalg.norm(rhs))
    if np.linalg.norm(matrix @ pair - rhs) > bar:
        msg = 'stored initial state contradicts actual state/rate constraints'
        raise ValueError(msg)
    # Recover the increment of EVERY original storage row from the declared
    # independent rows and actual left nullspace. Finite-Q Newton corrections
    # carry nonzero increments; requiring E*delta=0 would reject their target.
    storage_coordinates = sp.vstack((sp.eye(n, format='csr')[stored_rows], algebraic), format='csc')
    try:
        storage_factor = splu(storage_coordinates)
    except RuntimeError as error:
        msg = 'declared storage rank decomposition does not provide complete independent coordinates'
        raise ValueError(msg) from error
    expected_storage = storage_factor.solve(np.concatenate((stored_change, np.zeros(algebraic.shape[0]))))
    storage_defect = mass @ pair[:n] - expected_storage
    storage_bar = tolerance * (np.linalg.norm(mass.data) * np.linalg.norm(pair[:n]) + np.linalg.norm(expected_storage))
    if np.linalg.norm(storage_defect) > storage_bar:
        msg = 'declared storage rank decomposition does not preserve actual stored state'
        raise ValueError(msg)
    unique = nonzero[n:].copy()
    unique[[column - n for column in duplicate_rates]] = False
    return SparseInitialPair(initial + pair[:n], pair[n:], unique)


def sparse_complete_initial_rate(
    jacobian,
    residual,
    mass,
    pair,
    *,
    algebraic_rows,
    partial_time,
    acceleration_remainder,
    storage_curvature=None,
    tolerance=1e-10,
):
    """Differentiate actual sparse DAE rows again without a dense field nullspace.

    Preserve proven rates, solve the actual acceleration equations, and retain
    unknown acceleration coordinates only as explicit non-authoritative values.
    Structurally free rate columns remain unknown, never guessed as physical zero.
    """
    jacobian, mass, algebraic = map(sp.csr_matrix, (jacobian, mass, algebraic_rows))
    residual, partial, remainder = map(np.asarray, (residual, partial_time, acceleration_remainder))
    n = pair.state.size
    storage_second = np.zeros(n) if storage_curvature is None else np.asarray(storage_curvature)
    if (
        jacobian.shape != (n, n)
        or mass.shape != (n, n)
        or algebraic.shape[1] != n
        or any(value.shape != (n,) for value in (residual, partial, remainder, storage_second))
        or not all(
            np.isfinite(value).all()
            for value in (jacobian.data, mass.data, algebraic.data, residual, partial, remainder, storage_second)
        )
        or not np.isfinite(tolerance)
        or tolerance <= 0
    ):
        msg = 'sparse differentiated rates require finite declared actual operators and curvature'
        raise ValueError(msg)
    zero = sp.csr_matrix((n, n))
    algebraic_zero = sp.csr_matrix(algebraic.shape)
    anchors = sp.eye(n, format='csr')[np.flatnonzero(pair.rate_is_unique)]
    matrix = sp.bmat(
        [
            [mass, zero],
            [algebraic @ jacobian, algebraic_zero],
            [jacobian, mass],
            [algebraic_zero, algebraic @ jacobian],
            [anchors, sp.csr_matrix(anchors.shape)],
        ],
        format='csr',
    )
    rhs = np.concatenate((
        -residual - mass @ pair.rate,
        -algebraic @ (jacobian @ pair.rate + partial),
        -partial - storage_second - jacobian @ pair.rate,
        -algebraic @ remainder,
        np.zeros(anchors.shape[0]),
    ))
    matrix.eliminate_zeros()
    scale = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    scale[scale == 0] = 1
    matrix, rhs = sp.diags(1 / scale) @ matrix, rhs / scale
    nonzero = np.asarray(matrix.getnnz(axis=0)).ravel() != 0
    available = nonzero.copy()
    # Proven physical rates are prescribed coordinates, not extra soft rows
    # that an overdetermined subsystem may discard. Their correction is zero.
    available[np.flatnonzero(pair.rate_is_unique)] = False
    selected = np.flatnonzero(available)
    # Acceleration coordinates are not physical initial conditions. Exact
    # duplicate acceleration columns must not prevent proving a current rate.
    selected, _ = _independent_exact_columns(matrix, selected, dependent_start=n)
    reduced = matrix[:, selected].tocsr()
    rows = maximum_bipartite_matching(reduced, perm_type='row')
    if np.any(rows < 0):
        msg = 'differentiated sparse rates need additional physical acceleration authority'
        raise NotImplementedError(msg)
    try:
        factor = splu(reduced[rows].tocsc())
    except RuntimeError as error:
        msg = 'differentiated sparse rates lack a proven nonsingular acceleration subsystem'
        raise NotImplementedError(msg) from error
    correction = np.zeros(2 * n)
    correction[selected] = factor.solve(rhs[rows])
    bar = tolerance * (np.linalg.norm(matrix.data) * np.linalg.norm(correction) + np.linalg.norm(rhs))
    if np.linalg.norm(matrix @ correction - rhs) > bar:
        msg = 'declared curvature contradicts actual sparse differentiated DAE constraints'
        raise ValueError(msg)
    return SparseInitialPair(pair.state, pair.rate + correction[:n], nonzero[:n])
