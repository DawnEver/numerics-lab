"""Linear DAE initial constraints, with no minimum-norm guess for preferred states."""

import numpy as np
from scipy.linalg import null_space
from scipy.sparse import csr_matrix

from numerics_lab.linear.initial_sparse import SparseInitialPair

__all__ = ['complete_dae_initial_rate', 'initial_subspaces', 'linear_dae_initial_pair', 'linear_dae_initial_state']


def initial_subspaces(mass, extra_rate_columns):
    """Numerically scaled algebraic rows and differential-state coordinates."""
    mass = csr_matrix(mass).toarray()
    augmented = np.column_stack((mass, np.asarray(extra_rate_columns, dtype=float)))
    norms = np.linalg.norm(augmented, axis=1)
    scale = np.where(norms == 0.0, 1.0, norms)
    algebraic = null_space((augmented / scale[:, None]).T).T / scale[None, :]
    norms = np.linalg.norm(mass, axis=1)
    stored = mass / np.where(norms == 0.0, 1.0, norms)[:, None]
    return algebraic, stored


def linear_dae_initial_state(
    jacobian,
    residual,
    mass,
    initial,
    *,
    extra_rate_columns,
    preferred_indices,
    tolerance=1e-10,
    allow_preferred_changes=True,
):
    """Preserve stored state and solve algebraic consistency over actual DAE rows.

    Extra rate columns participate in algebraic admission, not stored-state ICs.
    Preferred initial components remain unchanged whenever consistent. If the
    constraints force changes, only uniquely determined components may change;
    a least-norm split of free preferred components is never an initial condition.
    """
    initial = np.asarray(initial, dtype=float)
    jacobian, mass = csr_matrix(jacobian).toarray(), csr_matrix(mass).toarray()
    rate = np.asarray(extra_rate_columns, dtype=float)
    residual = np.asarray(residual, dtype=float)
    n = initial.size
    indices = np.asarray(preferred_indices, dtype=int)
    if (
        initial.shape != (n,)
        or residual.shape != (n,)
        or jacobian.shape != (n, n)
        or mass.shape != (n, n)
        or rate.ndim != 2
        or rate.shape[0] != n
        or len(set(indices)) != len(indices)
        or np.any(indices < 0)
        or np.any(indices >= n)
        or not all(np.isfinite(value).all() for value in (initial, jacobian, mass, rate, residual))
        or not np.isfinite(tolerance)
        or tolerance <= 0
    ):
        msg = 'linear initial projection requires finite actual DAE matrices and valid preferred indices'
        raise ValueError(msg)

    algebraic, stored = initial_subspaces(mass, rate)
    matrix = np.vstack((algebraic @ jacobian, stored))
    rhs = np.concatenate((algebraic @ (jacobian @ initial - residual), stored @ initial))
    row_norm = np.linalg.norm(matrix, axis=1)
    scale = np.where(row_norm == 0.0, 1.0, row_norm)
    matrix, rhs = matrix / scale[:, None], rhs / scale

    def project(fixed):
        anchors = np.eye(n)[fixed]
        a, b = np.vstack((matrix, anchors)), np.concatenate((rhs, initial[fixed]))
        delta, *_ = np.linalg.lstsq(a, b - a @ initial, rcond=None)
        state = initial + delta
        # AN ANCHOR IS AN EXACT CONSTRAINT, so its coordinate is handed back exactly: the lstsq
        # solve returns it within an ulp, and a caller re-admitting this state against the same
        # exact value (a REST winding current of 0.0) would refuse the projection's own output.
        # Whether the snapped state is still consistent is judged below, on the snapped state.
        state[fixed] = initial[fixed]
        # Refine against the snapped anchors: the coordinates they DETERMINE (a series current
        # equal to a declared stored one) inherit the lstsq rounding otherwise, and a downstream
        # exact-anchor elimination reads that ulp as a contradicted declaration.
        for _ in range(3):
            defect = b - a @ state
            if not np.any(defect):
                break
            correction, *_ = np.linalg.lstsq(a, defect, rcond=None)
            state += correction
            state[fixed] = initial[fixed]
        bar = tolerance * (np.linalg.norm(a) * np.linalg.norm(state) + np.linalg.norm(b))
        return state, np.linalg.norm(a @ state - b) <= bar, a

    state, consistent, matrix_with_anchors = project(indices)
    if not consistent:
        if not allow_preferred_changes:
            msg = 'declared preferred initial states contradict the actual algebraic DAE constraints'
            raise ValueError(msg)
        _state, possible, _ = project(np.array([], dtype=int))
        if not possible:
            msg = 'declared stored initial state contradicts the actual algebraic DAE constraints'
            raise ValueError(msg)
        free = null_space(matrix)
        # The rank decision is floating-point algebra, not an electrical-current
        # threshold. A row with any free direction cannot choose its own value.
        rank_bar = np.finfo(float).eps * max(matrix.shape) * 16
        free_preferred = indices[np.linalg.norm(free[indices], axis=1) > rank_bar]
        state, consistent, matrix_with_anchors = project(free_preferred)
        if not consistent:
            msg = 'initial preferred states need explicit values; algebraic constraints do not determine their split'
            raise ValueError(msg)
    remaining = null_space(matrix_with_anchors)
    if np.any(np.linalg.norm(remaining[indices], axis=1) > np.finfo(float).eps * max(matrix_with_anchors.shape) * 16):
        msg = 'initial preferred DAE states are not uniquely determined after preserving declared states'
        raise ValueError(msg)
    return state


def linear_dae_initial_pair(jacobian, residual, mass, initial, *, partial_time, tolerance=1e-10):
    """Resolve algebraic state and stored-state rate for an index-two linear DAE.

    The caller supplies the actual partial time derivative of its residual at
    zero rate. Differential state is preserved; algebraic constraints are also
    differentiated, so an algebraic series-current equality constrains both
    currents and their rates. Free algebraic rates are harmless, but a free
    state or stored-state rate is not selected by a minimum-norm convention.
    """
    initial = np.asarray(initial, dtype=float)
    jacobian, mass = csr_matrix(jacobian).toarray(), csr_matrix(mass).toarray()
    residual, partial_time = np.asarray(residual, dtype=float), np.asarray(partial_time, dtype=float)
    n = initial.size
    if (
        initial.shape != (n,)
        or residual.shape != (n,)
        or partial_time.shape != (n,)
        or jacobian.shape != (n, n)
        or mass.shape != (n, n)
        or not all(np.isfinite(value).all() for value in (initial, jacobian, mass, residual, partial_time))
        or not np.isfinite(tolerance)
        or tolerance <= 0
    ):
        msg = 'initial state/rate consistency requires finite actual DAE matrices and time derivative'
        raise ValueError(msg)
    mass_scale = np.linalg.norm(mass, axis=1)
    mass_scale = np.where(mass_scale == 0.0, 1.0, mass_scale)
    algebraic = null_space((mass / mass_scale[:, None]).T).T / mass_scale[None, :]
    zero = np.zeros_like(mass)
    matrix = np.vstack((np.column_stack((jacobian, mass)), np.column_stack((mass, zero))))
    matrix = np.vstack((matrix, np.column_stack((np.zeros_like(algebraic), algebraic @ jacobian))))
    rhs = np.concatenate((-residual, np.zeros(n), -algebraic @ partial_time))
    scale = np.linalg.norm(matrix, axis=1)
    scale = np.where(scale == 0.0, 1.0, scale)
    matrix, rhs = matrix / scale[:, None], rhs / scale
    pair, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    bar = tolerance * (np.linalg.norm(matrix) * np.linalg.norm(pair) + np.linalg.norm(rhs))
    if np.linalg.norm(matrix @ pair - rhs) > bar:
        msg = 'stored initial state contradicts the actual state/rate DAE constraints'
        raise ValueError(msg)
    free = null_space(matrix)
    rank_bar = np.finfo(float).eps * max(matrix.shape) * 16
    normalized_mass = mass / mass_scale[:, None]
    if np.linalg.norm(free[:n]) > rank_bar or np.linalg.norm(normalized_mass @ free[n:]) > rank_bar:
        msg = 'initial state or stored-state rate requires additional physical initial authority'
        raise ValueError(msg)
    unique_rate = np.linalg.norm(free[n:], axis=1) <= rank_bar
    return SparseInitialPair(initial + pair[:n], pair[n:], unique_rate)


def complete_dae_initial_rate(
    jacobian, residual, mass, pair, *, partial_time, acceleration_remainder, storage_curvature=None, tolerance=1e-10
):
    """Use declared second differentiated constraints, never choose free rates.

    This small-system owner eliminates the actual acceleration block. The
    caller owns residual Hessians/time chains; no finite difference is used.
    """
    jacobian, mass = csr_matrix(jacobian).toarray(), csr_matrix(mass).toarray()
    residual, partial = np.asarray(residual), np.asarray(partial_time)
    remainder = np.asarray(acceleration_remainder)
    n = pair.state.size
    storage_second = np.zeros(n) if storage_curvature is None else np.asarray(storage_curvature)
    if (
        jacobian.shape != (n, n)
        or mass.shape != (n, n)
        or any(value.shape != (n,) for value in (residual, partial, remainder, storage_second))
        or not all(np.isfinite(value).all() for value in (jacobian, mass, residual, partial, remainder, storage_second))
        or not np.isfinite(tolerance)
        or tolerance <= 0
    ):
        msg = 'differentiated rate consistency requires finite declared actual operators'
        raise ValueError(msg)
    norms = np.linalg.norm(mass, axis=1)
    norms[norms == 0] = 1
    algebraic = null_space((mass / norms[:, None]).T).T / norms[None, :]
    acceleration = np.vstack((mass, algebraic @ jacobian))
    scale = np.linalg.norm(acceleration, axis=1)
    scale[scale == 0] = 1
    left = null_space((acceleration / scale[:, None]).T).T / scale[None, :]
    matrix = np.vstack((mass, algebraic @ jacobian, left[:, :n] @ jacobian))
    rhs = np.concatenate((
        -residual,
        -algebraic @ partial,
        -left @ np.concatenate((partial + storage_second, algebraic @ remainder)),
    ))
    scale = np.linalg.norm(matrix, axis=1)
    scale[scale == 0] = 1
    matrix, rhs = matrix / scale[:, None], rhs / scale
    rate, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    if np.linalg.norm(matrix @ rate - rhs) > tolerance * (
        np.linalg.norm(matrix) * np.linalg.norm(rate) + np.linalg.norm(rhs)
    ):
        msg = 'declared source/device curvature contradicts actual differentiated DAE constraints'
        raise ValueError(msg)
    free = null_space(matrix)
    unique = np.linalg.norm(free, axis=1) <= np.finfo(float).eps * max(matrix.shape) * 16
    if np.any(pair.rate_is_unique & ~unique):
        msg = 'higher differentiated constraints lost previously proven rate authority'
        raise ValueError(msg)
    proven = pair.rate_is_unique
    if np.linalg.norm(rate[proven] - pair.rate[proven]) > tolerance * (
        np.linalg.norm(rate) + np.linalg.norm(pair.rate[proven])
    ):
        msg = 'higher differentiated constraints changed previously proven physical rates'
        raise ValueError(msg)
    return SparseInitialPair(pair.state, np.where(proven, pair.rate, rate), unique)
