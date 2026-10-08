"""Nonlinear algebraic IC projection on invariant actual DAE storage subspaces."""

import numpy as np
from scipy.linalg import null_space
from scipy.sparse import csr_matrix

from motronics.gauss.linear.initial_constraints import initial_subspaces, linear_dae_initial_state
from motronics.gauss.nonlinear import AssembledState, NewtonStrategy


def nonlinear_dae_initial_state(
    residual,
    jacobian,
    mass,
    initial,
    *,
    extra_rate_columns,
    preferred_indices,
    max_iter,
    tolerance=1e-10,
    allow_preferred_changes=True,
    stored_state=None,
):
    """Use the shared Newton/line-search owner; never guess a free preferred IC.

    The algebraic rate space must remain invariant during projection. Without
    stored_state the storage coordinates must also be invariant; with that
    explicit primitive its actual Jacobian is mass, never a frozen surrogate.
    """
    initial = np.asarray(initial, dtype=float)
    initial_mass = csr_matrix(mass(initial)).toarray()
    initial_jacobian = csr_matrix(jacobian(initial)).toarray()
    linear_dae_initial_state(
        initial_jacobian,
        residual(initial),
        initial_mass,
        initial,
        extra_rate_columns=extra_rate_columns,
        preferred_indices=preferred_indices,
        tolerance=tolerance,
        allow_preferred_changes=allow_preferred_changes,
    )
    algebraic, stored = initial_subspaces(initial_mass, extra_rate_columns)
    matrix = np.vstack((algebraic @ initial_jacobian, stored))
    norms = np.linalg.norm(matrix, axis=1)
    matrix = matrix / np.where(norms == 0.0, 1.0, norms)[:, None]
    indices = np.asarray(preferred_indices, dtype=int)
    if allow_preferred_changes:
        free = null_space(matrix)
        rank_bar = np.finfo(float).eps * max(matrix.shape) * 16
        indices = indices[np.linalg.norm(free[indices], axis=1) > rank_bar]
    anchors = np.eye(initial.size)[indices]
    system = _InitialConstraints(
        residual, jacobian, mass, initial, algebraic, stored, anchors, extra_rate_columns, stored_state
    )
    result = NewtonStrategy().solve(system, max_iter=max_iter, rtol_energy_pct=tolerance * 100)
    if not result.converged:
        msg = 'nonlinear stored initial state did not close its actual algebraic DAE constraints'
        raise ValueError(msg)
    state = system.assemble(result.x)
    reference = np.linalg.norm(state.system_matrix) * np.linalg.norm(result.x) + np.linalg.norm(stored @ initial)
    if system.residual_norm(result.x, state) > tolerance * reference:
        msg = 'nonlinear initial constraints did not satisfy the measured final-state residual scale'
        raise ValueError(msg)
    free = null_space(state.system_matrix)
    indices = np.asarray(preferred_indices, dtype=int)
    if np.any(np.linalg.norm(free[indices], axis=1) > np.finfo(float).eps * max(state.system_matrix.shape) * 16):
        msg = 'nonlinear initial preferred states require additional physical initial authority'
        raise ValueError(msg)
    return result.x


class _InitialConstraints:
    is_linear = False

    def __init__(
        self, residual, jacobian, mass, initial, algebraic, stored, anchors, extra_rate_columns, stored_state
    ) -> None:
        self.evaluate, self.evaluate_jacobian, self.evaluate_mass = residual, jacobian, mass
        self.initial = initial
        self.algebraic, self.stored, self.anchors = algebraic, stored, anchors
        self.extra_rate_columns = extra_rate_columns
        self.storage_free = null_space(stored)
        self.evaluate_storage = stored_state
        norms = np.linalg.norm(csr_matrix(mass(initial)).toarray(), axis=1)
        self.storage_scale = np.where(norms == 0.0, 1.0, norms)
        self.initial_storage = None if stored_state is None else self._storage(initial)

    def _storage(self, state):
        value = np.asarray(self.evaluate_storage(state), dtype=float)
        if value.shape != self.initial.shape or not np.isfinite(value).all():
            msg = 'finite initial storage must cover every actual equation with finite values'
            raise ValueError(msg)
        return value

    def initial_guess(self):
        return self.initial.copy()

    def assemble(self, x):
        mass = csr_matrix(self.evaluate_mass(x)).toarray()
        augmented = np.column_stack((mass, self.extra_rate_columns))
        norms = np.linalg.norm(mass, axis=1)
        actual_stored = mass / np.where(norms == 0.0, 1.0, norms)[:, None]
        if (
            np.linalg.norm(self.algebraic @ augmented)
            > 1e-10 * np.linalg.norm(self.algebraic) * np.linalg.norm(augmented)
            or (
                self.evaluate_storage is None
                and np.linalg.norm(actual_stored @ self.storage_free) > 1e-10 * np.linalg.norm(actual_stored)
            )
            or np.linalg.matrix_rank(actual_stored) != np.linalg.matrix_rank(self.stored)
        ):
            msg = 'nonlinear initial projection needs an invariant storage/rate space or an explicit stored-state map'
            raise ValueError(msg)
        jacobian = csr_matrix(self.evaluate_jacobian(x)).toarray()
        stored = self.stored if self.evaluate_storage is None else mass / self.storage_scale[:, None]
        increment = (
            self.stored @ (x - self.initial)
            if self.evaluate_storage is None
            else (self._storage(x) - self.initial_storage) / self.storage_scale
        )
        matrix = np.vstack((self.algebraic @ jacobian, stored, self.anchors))
        residual = np.concatenate((
            self.algebraic @ self.evaluate(x),
            increment,
            self.anchors @ (x - self.initial),
        ))
        norms = np.linalg.norm(matrix, axis=1)
        scale = np.where(norms == 0.0, 1.0, norms)
        matrix, residual = matrix / scale[:, None], residual / scale
        return AssembledState(matrix, 0.5 * float(residual @ residual), payload=residual)

    def residual(self, x, state):
        return state.payload

    def residual_norm(self, x, state):
        return float(np.linalg.norm(state.payload))

    def tangent(self, x, state):
        return state.system_matrix

    def solve_system(self, matrix, rhs):
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]

    def solve_linear(self, state):
        msg = 'initial algebraic constraints require the declared nonlinear Newton owner'
        raise TypeError(msg)

    def on_iteration(self, i, state, err, alpha, increment) -> None:
        pass
