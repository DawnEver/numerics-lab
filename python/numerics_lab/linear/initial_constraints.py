"""Linear DAE initial constraints, with no minimum-norm guess for preferred states."""

import numpy as np
from scipy.linalg import null_space
from scipy.sparse import csr_matrix


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

    def normalized(matrix):
        norms = np.linalg.norm(matrix, axis=1)
        return matrix / np.where(norms == 0.0, 1.0, norms)[:, None]

    augmented_rate = np.column_stack((mass, rate))
    rate_norm = np.linalg.norm(augmented_rate, axis=1)
    rate_scale = np.where(rate_norm == 0.0, 1.0, rate_norm)
    augmented_rate = augmented_rate / rate_scale[:, None]
    algebraic = null_space(augmented_rate.T).T / rate_scale[None, :]
    stored = normalized(mass)
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
