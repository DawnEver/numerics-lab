"""Sparse initial pair over caller-proven independent storage/algebraic rows."""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.sparse.linalg import splu

from numerics_lab.formulation import ResidualAssemblyBound
from numerics_lab.linear.arithmetic import affine_residual_bound, certified_sparse_product, transport_residual_bound
from numerics_lab.linear.initial_rate import (
    _independent_exact_columns,
    _independent_exact_rows,
    normalize_sparse_rows,
)
from numerics_lab.linear.initial_rate_border import complete_rate_border
from numerics_lab.linear.initial_storage_border import stored_initial_correction
from numerics_lab.linear.resolution import row_bar, system_resolution
from numerics_lab.linear.storage_border import free_pinned_columns

__all__ = ['SparseInitialPair', 'projected_rate_consistent', 'sparse_complete_initial_rate', 'sparse_initial_pair']


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


def projected_rate_consistent(
    jacobian, rate, partial_time, *, algebraic_rows, tolerance, differentiated_evaluation=None
):
    """Check each actual differentiated equation in its own projected units.

    Null-basis row scales are arbitrary numerical coordinates, not physics.
    A large constraint cannot hide a contradictory differently scaled row: each row keeps its own
    relative bar; only a row whose own operands are under the SYSTEM's resolution (the full
    ``|J| . |rate| + |partial|``) is noise and judged as zero, so free rates that are rounding on a
    homogeneous constraint are not read as a 100% defect.
    """
    operator = algebraic_rows @ jacobian
    forcing = algebraic_rows @ partial_time
    defect = operator @ rate + forcing
    scale = np.abs(operator) @ np.abs(rate) + np.abs(forcing)
    resolution = system_resolution(abs(jacobian) @ np.abs(rate) + np.abs(partial_time))
    allowance = np.zeros_like(defect)
    if differentiated_evaluation is not None:
        actual = np.asarray(differentiated_evaluation.residual)
        bound = differentiated_evaluation.bound
        if (
            actual.shape != defect.shape
            or actual.dtype.kind not in 'fiu'
            or not np.isfinite(actual).all()
            or not isinstance(bound, ResidualAssemblyBound)
            or bound.values.shape != defect.shape
        ):
            msg = 'differentiated replay requires its finite aligned actual projected value and bound'
            raise ValueError(msg)
        # This certificate owns Y*(J*v+Ft), not the separately reassociated
        # (Y*J)*v + Y*Ft. Keep the original physical scale, but replay its
        # actual performed value rather than attaching it to another sum.
        defect, allowance = actual, bound.allowance()
    return bool(np.all(np.abs(defect) <= row_bar(tolerance * scale, scale, resolution, forcing) + allowance))


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
    stored_columns=None,
    rate_jacobian=None,
    sparse_state_indices=None,
    tolerance=1e-10,
    residual_allowance=None,
    stored_allowance=None,
    differentiated_allowance=None,
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
    rate_tangent = jacobian if rate_jacobian is None else sp.csr_matrix(rate_jacobian)
    n = initial.size
    stored_rows = np.asarray(stored_rows)
    stored_change = np.zeros(len(stored_rows)) if stored_increment is None else np.asarray(stored_increment)
    derivative_bound = (
        np.zeros(algebraic.shape[0]) if differentiated_allowance is None else np.asarray(differentiated_allowance)
    )
    if (
        derivative_bound.shape != (algebraic.shape[0],)
        or derivative_bound.dtype.kind not in 'fiu'
        or not np.isfinite(derivative_bound).all()
        or np.any(derivative_bound < 0)
    ):
        msg = 'differentiated allowance requires finite nonnegative original projected-row bounds'
        raise ValueError(msg)
    if (
        initial.shape != (n,)
        or residual.shape != (n,)
        or partial_time.shape != (n,)
        or jacobian.shape != (n, n)
        or mass.shape != (n, n)
        or rate_tangent.shape != (n, n)
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
            for value in (
                initial,
                residual,
                partial_time,
                stored_change,
                jacobian.data,
                mass.data,
                algebraic.data,
                rate_tangent.data,
            )
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
    if stored_columns is None and stored_allowance is not None and np.any(stored_allowance):
        msg = 'certified stored increments require the declared sparse storage elimination'
        raise NotImplementedError(msg)
    if stored_columns is not None:
        correction, unique = stored_initial_correction(
            jacobian,
            residual,
            mass,
            algebraic,
            stored_rows,
            stored_columns,
            stored_change,
            partial_time,
            tolerance=tolerance,
            state=initial,
            rate_tangent=rate_tangent,
            sparse_state_indices=sparse_state_indices,
            residual_allowance=residual_allowance,
            stored_allowance=stored_allowance,
            differentiated_allowance=differentiated_allowance,
        )
        return SparseInitialPair(initial + correction[:n], correction[n:], unique)
    matrix = sp.bmat(
        [
            [jacobian, mass],
            [stored, sp.csr_matrix(stored.shape)],
            [sp.csr_matrix(algebraic.shape), certified_sparse_product(algebraic, rate_tangent)],
        ],
        format='csr',
    )
    rhs = np.concatenate((-residual, stored_change, -algebraic @ partial_time))
    matrix.eliminate_zeros()
    nonzero = np.asarray(matrix.getnnz(axis=0)).ravel() != 0
    if not nonzero[:n].all():
        msg = 'initial state requires additional physical authority'
        raise ValueError(msg)
    actual_matrix = matrix
    actual_rhs = rhs.copy()
    selected, duplicate_rates = _independent_exact_columns(
        actual_matrix, np.flatnonzero(nonzero), dependent_start=n, general=False
    )
    matrix, rhs = normalize_sparse_rows(matrix, rhs)
    # Zero and identical rate columns need no rational elimination. A general
    # exact quotient is attempted only if the sparse physical factor is singular.
    reduced = matrix[:, selected].tocsr()
    # Storage equations are mandatory, even when their increments are tiny
    # compared with the dynamic RHS. Match them first, then verify that none
    # was discarded by the structural reduction of free algebraic rates.
    mandatory = np.arange(n, n + len(stored_rows))
    row_order = np.concatenate((mandatory, np.arange(n + len(stored_rows), 2 * n), np.arange(n)))
    matched = maximum_bipartite_matching(reduced[row_order], perm_type='row')
    matched_rows = row_order[matched]
    if not np.any(matched < 0) and not np.isin(mandatory, matched_rows).all():
        msg = 'initial state/rate subsystem cannot discard actual stored-coordinate constraints'
        raise ValueError(msg)
    factor = None
    factor_error = None
    if not np.any(matched < 0):
        try:
            factor = splu(reduced[matched_rows].tocsc())
        except RuntimeError as error:
            factor_error = error
    if factor is None:
        selected, duplicate_rates = _independent_exact_columns(
            actual_matrix, np.flatnonzero(nonzero), dependent_start=n
        )
        reduced = matrix[:, selected].tocsr()
        matched = maximum_bipartite_matching(reduced[row_order], perm_type='row')
        if np.any(matched < 0):
            msg = 'initial state/rate equations require additional physical authority'
            raise ValueError(msg) from factor_error
        matched_rows = row_order[matched]
        if not np.isin(mandatory, matched_rows).all():
            msg = 'initial state/rate subsystem cannot discard actual stored-coordinate constraints'
            raise ValueError(msg) from factor_error
        try:
            factor = splu(reduced[matched_rows].tocsc())
        except RuntimeError:
            # Structural matching proves only a nonzero pattern, not actual row
            # independence. Keep every stored equation and choose an exact basis
            # of the original unscaled rows; all omitted equations are checked below.
            matched_rows = _independent_exact_rows(actual_matrix[:, selected], mandatory)
            try:
                factor = splu(reduced[matched_rows].tocsc())
            except RuntimeError as basis_error:
                gauged = _gauge_free_algebraic_rates(matrix, selected, row_order, mandatory, mass, n)
                if gauged is None:
                    msg = 'initial state/rate equations lack a nonsingular sparse physical constraint subsystem'
                    raise ValueError(msg) from basis_error
                selected, gauge_columns, row_order = gauged
                nonzero[gauge_columns] = False  # marked free, exactly as a structurally free rate is
                reduced, matched_rows, factor = _matched_factor(matrix, selected, row_order, mandatory)
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
    compatible = np.linalg.norm(matrix @ pair - rhs) <= bar
    if not compatible and differentiated_allowance is not None:
        allowance = np.concatenate((np.zeros(n + len(stored_rows)), derivative_bound))
        scale = abs(actual_matrix) @ abs(pair) + abs(actual_rhs)
        compatible = np.all(abs(actual_matrix @ pair - actual_rhs) <= tolerance * scale + allowance)
    if not compatible:
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


def _matched_factor(matrix, selected, row_order, mandatory):
    """The square subsystem a row matching picks for ``selected`` columns, and its LU factor."""
    reduced = matrix[:, selected].tocsr()
    matched = maximum_bipartite_matching(reduced[row_order], perm_type='row')
    if np.any(matched < 0):
        msg = 'initial state/rate equations require additional physical authority'
        raise ValueError(msg)
    matched_rows = row_order[matched]
    if not np.isin(mandatory, matched_rows).all():
        msg = 'initial state/rate subsystem cannot discard actual stored-coordinate constraints'
        raise ValueError(msg)
    return reduced, matched_rows, splu(reduced[matched_rows].tocsc())


#: The largest matched subsystem the exact-nullspace check below will take a dense SVD of.
_GAUGE_DENSE_LIMIT = 2000


def _gauge_free_algebraic_rates(matrix, selected, row_order, mandatory, mass, n):
    """Fix the gauge of a rate no stored state reads, or ``None`` when the singularity is anything else.

    THE MEASURED CASE: inverter_3ph_grid_tied's grid star point ``n`` has no reference, so the whole
    grid cluster {ma, ga, mb, gb, mc, gc, n} floats behind the three inductors. Its potential is
    fixed only by the DIFFERENTIATED KCL (sum di/dt = 0), so its RATE is an index-two quantity the
    first differentiation cannot fix: the matched subsystem has one exact null vector, uniform over
    exactly those seven node-voltage rate columns (singular value 5.9e-15 against 1.7). No mass
    column reads those rates, so no stored state and no march depends on them.

    Only that shape is admitted: every exact null vector must live on rate columns whose mass column
    is zero. One such column per null vector is dropped and every rate on the vector is marked free
    (never asserted as a physical value), and the caller re-matches; the caller's full residual check still judges the
    answer against every equation, the dropped one included.
    """
    reduced = matrix[:, selected].tocsr()
    matched = maximum_bipartite_matching(reduced[row_order], perm_type='row')
    if np.any(matched < 0) or len(selected) > _GAUGE_DENSE_LIMIT:
        return None
    rows = row_order[matched]
    block = reduced[rows].toarray()
    u, sigma, vt = np.linalg.svd(block)
    exact = sigma <= 1e-12 * sigma[0]
    null = vt[exact]
    if not len(null):
        return None
    massless = {n + c for c in range(n) if not np.any(mass[:, [c]].toarray())}
    drop, free = [], []
    for vector in null:
        support = np.flatnonzero(np.abs(vector) > 1e-9 * np.max(np.abs(vector)))
        columns = [int(selected[i]) for i in support]
        if not set(columns) <= massless:
            return None
        drop.append(columns[int(np.argmax(np.abs(vector[support])))])
        # EVERY rate on the null vector is fixed only relative to the gauge, so all of them are
        # free, not just the dropped one -- a later differentiation may move them together.
        free.extend(columns)
    # The matched rows are dependent along the LEFT null vectors: the equation each one makes
    # redundant leaves the matching too (it is still judged by the caller's full residual check).
    redundant = {int(rows[int(np.argmax(np.abs(left)))]) for left in u[:, exact].T}
    if set(redundant) & set(mandatory.tolist()):
        return None
    kept_rows = np.asarray([r for r in row_order if int(r) not in redundant], dtype=int)
    return np.asarray([c for c in selected if c not in drop], dtype=int), sorted(set(free)), kept_rows


def sparse_complete_initial_rate(
    jacobian,
    residual,
    mass,
    pair,
    *,
    algebraic_rows,
    partial_time,
    partial_time_bound=None,
    acceleration_remainder,
    acceleration_jacobian=None,
    storage_curvature=None,
    stored_rows=None,
    stored_columns=None,
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
    if partial_time_bound is not None and (
        not isinstance(partial_time_bound, ResidualAssemblyBound) or partial_time_bound.values.shape != (n,)
    ):
        msg = 'partial time assembly bound must cover the actual differentiated equations'
        raise ValueError(msg)
    second = jacobian if acceleration_jacobian is None else sp.csr_matrix(acceleration_jacobian)
    if second.shape != (n, n) or second.dtype.kind not in 'fiu' or not np.isfinite(second.data).all():
        msg = 'second acceleration operator requires finite real actual state dimensions'
        raise ValueError(msg)
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
            [algebraic_zero, algebraic @ second],
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
    equation_bound = ResidualAssemblyBound(np.zeros(rhs.size), np.zeros(rhs.size, dtype=bool))
    if partial_time_bound is not None:
        # These are the performed RHS expressions, in their original order.
        # Ft authority belongs only to the two differentiated equations which
        # actually consume Ft; F, curvature and preserved-rate rows get none.
        rate_partial = jacobian @ pair.rate + partial
        rate_partial_bound = affine_residual_bound(jacobian, pair.rate, partial, partial_time_bound)
        projected = transport_residual_bound(-algebraic, rate_partial, rate_partial_bound)
        storage_partial = -partial - storage_second
        storage_partial_bound = affine_residual_bound(
            -sp.eye(n, format='csr'), storage_second, -partial, partial_time_bound
        )
        acceleration = affine_residual_bound(-jacobian, pair.rate, storage_partial, storage_partial_bound)
        equation_bound = ResidualAssemblyBound(
            np.concatenate((
                np.zeros(n),
                projected.values,
                acceleration.values,
                np.zeros(algebraic.shape[0] + anchors.shape[0]),
            )),
            np.concatenate((
                np.zeros(n, dtype=bool),
                projected.known,
                acceleration.known,
                np.zeros(algebraic.shape[0] + anchors.shape[0], dtype=bool),
            )),
            np.concatenate((
                np.zeros(n),
                projected.law,
                acceleration.law,
                np.zeros(algebraic.shape[0] + anchors.shape[0]),
            )),
        )
    matrix.eliminate_zeros()
    scale = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    scale[scale == 0] = 1
    equation_bound = transport_residual_bound(sp.diags(1 / scale), rhs, equation_bound)
    matrix, rhs = sp.diags(1 / scale) @ matrix, rhs / scale
    if stored_rows is not None or stored_columns is not None:
        if stored_rows is None or stored_columns is None:
            msg = 'stored acceleration requires both actual independent rows and columns'
            raise ValueError(msg)
        evaluation_scale = (
            np.concatenate((
                # The admitted residual includes evaluated algebraic cancellations
                # (including zero-mass linkage rows). Its own tiny remainder is
                # not the backward-error scale of that physical equation.
                abs(residual) + abs(jacobian) @ abs(pair.state) + abs(mass) @ abs(pair.rate),
                abs(algebraic) @ (abs(jacobian) @ abs(pair.rate) + abs(partial)),
                abs(partial) + abs(storage_second) + abs(jacobian) @ abs(pair.rate),
                abs(algebraic) @ abs(remainder),
                np.zeros(anchors.shape[0]),
            ))
            / scale
        )
        correction, unique = complete_rate_border(
            matrix,
            rhs,
            mass=mass,
            jacobian=jacobian,
            acceleration_rhs=-partial - storage_second - jacobian @ pair.rate,
            rate_is_unique=pair.rate_is_unique,
            stored_rows=stored_rows,
            stored_columns=free_pinned_columns(stored_columns, mass[stored_rows], algebraic @ second),
            evaluation_scale=evaluation_scale,
            equation_bound=equation_bound,
            tolerance=tolerance,
            acceleration_equation_start=n + algebraic.shape[0],
        )
        return SparseInitialPair(pair.state, pair.rate + correction[:n], unique)
    nonzero = np.asarray(matrix.getnnz(axis=0)).ravel() != 0
    available = nonzero.copy()
    # Proven physical rates are prescribed coordinates, not extra soft rows
    # that an overdetermined subsystem may discard. Their correction is zero.
    available[np.flatnonzero(pair.rate_is_unique)] = False
    selected = np.flatnonzero(available)
    # Acceleration coordinates are not physical initial conditions. Exact
    # duplicate acceleration columns must not prevent proving a current rate.
    selected, _ = _independent_exact_columns(matrix, selected, dependent_start=n, general=False)
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
    if np.linalg.norm(np.maximum(abs(matrix @ correction - rhs) - equation_bound.allowance(), 0)) > bar:
        msg = 'declared curvature contradicts actual sparse differentiated DAE constraints'
        raise ValueError(msg)
    return SparseInitialPair(pair.state, pair.rate + correction[:n], nonzero[:n])
