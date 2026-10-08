"""Initial/right-state projection over an actual finite storage map.

No physical constitutive law lives here. Callers supply Q, DQ, the residual,
and a proven invariant left-nullspace of DQ. A changing storage nullspace
requires its own coordinate authority rather than a constant-mass surrogate.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from motronics.gauss.linear.initial_sparse import sparse_initial_pair


class StorageRateAuthorityRequired(ValueError):
    """Finite stored coordinates and undifferentiated constraints do not fix every state."""


def finite_storage_right_state(
    initial, *, residual_at, jacobian_at, mass_at, storage_at, storage_rows, max_newton=30, tolerance=1e-10
):
    """Preserve actual finite stored coordinates, not DQ(initial) times state."""
    initial = np.asarray(initial, dtype=float)
    if (
        initial.ndim != 1
        or not np.isfinite(initial).all()
        or max_newton < 1
        or not np.isfinite(tolerance)
        or tolerance <= 0
    ):
        msg = 'finite storage projection requires finite state and positive iteration controls'
        raise ValueError(msg)
    algebraic, selected = storage_rows.algebraic_rows, storage_rows.stored_rows
    target = np.asarray(storage_at(initial))[selected]
    state = initial.copy()
    for _iteration in range(max_newton):
        mass = sp.csr_matrix(mass_at(state))
        defect = algebraic @ mass
        if np.linalg.norm(defect.data) > tolerance * np.linalg.norm(algebraic.data) * np.linalg.norm(mass.data):
            msg = 'finite storage nullspace changed; its actual nonlinear coordinate owner is required'
            raise ValueError(msg)
        jacobian = sp.csr_matrix(jacobian_at(state))
        residual = np.asarray(residual_at(state))
        stored = np.asarray(storage_at(state))[selected]
        matrix = sp.vstack((algebraic @ jacobian, mass[selected]), format='csr')
        rhs = np.concatenate((-algebraic @ residual, target - stored))
        scale = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
        scale[scale == 0] = 1
        normalized, normalized_rhs = sp.diags(1 / scale) @ matrix, rhs / scale
        try:
            correction = splu(normalized.tocsc()).solve(normalized_rhs)
        except RuntimeError as error:
            msg = 'finite stored-coordinate projection lacks a nonsingular physical algebraic subsystem'
            raise StorageRateAuthorityRequired(msg) from error
        state += correction
        if not np.isfinite(state).all():
            msg = 'finite storage projection produced a nonfinite physical state'
            raise ValueError(msg)
        actual = np.concatenate((algebraic @ residual_at(state), np.asarray(storage_at(state))[selected] - target))
        reference = np.linalg.norm(normalized.data) * np.linalg.norm(state) + np.linalg.norm(normalized_rhs)
        if np.linalg.norm(actual / scale) <= tolerance * reference:
            final_mass = sp.csr_matrix(mass_at(state))
            final_defect = algebraic @ final_mass
            if np.linalg.norm(final_defect.data) <= tolerance * np.linalg.norm(algebraic.data) * np.linalg.norm(
                final_mass.data
            ):
                return state
    msg = 'actual finite stored-coordinate projection did not converge'
    raise ValueError(msg)


def finite_storage_initial_pair(
    initial,
    *,
    residual_at,
    jacobian_at,
    mass_at,
    storage_at,
    partial_time_at,
    storage_rows,
    max_newton=30,
    tolerance=1e-10,
):
    """Actual state/rate pair for finite Q with its invariant algebraic row space."""
    state = np.asarray(initial, dtype=float).copy()
    zero = np.zeros_like(state)
    algebraic, selected = storage_rows.algebraic_rows, storage_rows.stored_rows
    target = np.asarray(storage_at(initial))[storage_rows.stored_rows]
    if max_newton < 1 or not np.isfinite(tolerance) or tolerance <= 0:
        msg = 'finite storage initial pair requires positive iteration controls'
        raise ValueError(msg)
    for _iteration in range(max_newton):
        mass = sp.csr_matrix(mass_at(state))
        defect = algebraic @ mass
        if np.linalg.norm(defect.data) > tolerance * np.linalg.norm(algebraic.data) * np.linalg.norm(mass.data):
            msg = 'finite storage initial pair requires its invariant actual algebraic row space'
            raise ValueError(msg)
        jacobian, residual = sp.csr_matrix(jacobian_at(state)), np.asarray(residual_at(state, zero))
        pair = sparse_initial_pair(
            jacobian,
            residual,
            mass,
            state,
            algebraic_rows=algebraic,
            stored_rows=selected,
            stored_increment=target - np.asarray(storage_at(state))[selected],
            partial_time=partial_time_at(state),
            tolerance=tolerance,
        )
        state = pair.state
        actual_mass, actual_jacobian = sp.csr_matrix(mass_at(state)), sp.csr_matrix(jacobian_at(state))
        actual = np.asarray(storage_at(state))[selected]
        residual_defect = np.asarray(residual_at(state, pair.rate))
        rate_defect = algebraic @ (actual_jacobian @ pair.rate + partial_time_at(state))
        state_scale = np.linalg.norm(actual_jacobian.data) * np.linalg.norm(state) + np.linalg.norm(
            actual_mass @ pair.rate
        )
        storage_scale = (
            np.linalg.norm(actual)
            + np.linalg.norm(target)
            + np.linalg.norm(actual_mass[selected].data) * np.linalg.norm(state)
        )
        rate_scale = np.linalg.norm(actual_jacobian.data) * np.linalg.norm(pair.rate) + np.linalg.norm(
            partial_time_at(state)
        )
        if (
            np.linalg.norm(residual_defect) <= tolerance * state_scale
            and np.linalg.norm(actual - target) <= tolerance * storage_scale
            and np.linalg.norm(rate_defect) <= tolerance * rate_scale
        ):
            final_defect = algebraic @ actual_mass
            if np.linalg.norm(final_defect.data) <= tolerance * np.linalg.norm(algebraic.data) * np.linalg.norm(
                actual_mass.data
            ):
                return pair
    msg = 'actual finite storage state/rate projection did not converge'
    raise ValueError(msg)
