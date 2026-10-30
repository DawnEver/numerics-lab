"""Nonlinear algebraic IC projection on invariant actual DAE storage subspaces."""

import numpy as np
from scipy.linalg import null_space
from scipy.sparse import csr_matrix

from numerics_lab.linear.border_rank import border_affine_space, rank_bar
from numerics_lab.linear.initial_constraints import initial_subspaces, linear_dae_initial_state
from numerics_lab.nonlinear import AssembledState, NewtonStrategy

__all__ = ['nonlinear_dae_initial_state']


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
    affine_constraints=None,
):
    """Use the shared Newton/line-search owner; never guess a free preferred IC.

    The algebraic rate space must remain invariant during projection. Without
    stored_state the storage coordinates must also be invariant; with that
    explicit primitive its actual Jacobian is mass, never a frozen surrogate.

    affine_constraints, when supplied, must be owner-declared equations valid
    throughout the search, not a sampled tangent certificate. They restrict the
    search coordinates; acceptance still checks the original nonlinear DAE,
    stored-state equations, and final preferred-coordinate uniqueness.
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
        bar = rank_bar(matrix.shape)
        indices = indices[np.linalg.norm(free[indices], axis=1) > bar]
    anchored = indices
    anchors = np.eye(initial.size)[anchored]
    system = _InitialConstraints(
        residual, jacobian, mass, initial, algebraic, stored, anchors, extra_rate_columns, stored_state
    )
    reduced = system if affine_constraints is None else _AffineInitialConstraints(system, affine_constraints, tolerance)
    result = NewtonStrategy().solve(reduced, max_iter=max_iter, rtol_energy_pct=tolerance * 100)
    if not result.converged:
        msg = 'nonlinear stored initial state did not close its actual algebraic DAE constraints'
        raise ValueError(msg)
    value = result.x if affine_constraints is None else reduced.full_state(result.x)
    # AN ANCHOR IS AN EXACT CONSTRAINT, so its coordinate is handed back exactly, as the linear
    # projection does: Newton closes it within an ulp (a REST wye current came back 1e-35), and a
    # caller re-checking the same exact value refuses the projection's own output. The snapped
    # state is what the residual and uniqueness checks below judge.
    value[anchored] = initial[anchored]
    state = system.assemble(value)
    reference = np.linalg.norm(state.system_matrix) * np.linalg.norm(value) + np.linalg.norm(stored @ initial)
    if system.residual_norm(value, state) > tolerance * reference:
        msg = 'nonlinear initial constraints did not satisfy the measured final-state residual scale'
        raise ValueError(msg)
    free = (
        null_space(state.system_matrix)
        if affine_constraints is None
        else border_affine_space(state.system_matrix, np.zeros(state.system_matrix.shape[0]))[1]
    )
    indices = np.asarray(preferred_indices, dtype=int)
    if (
        np.any(free[indices] != 0)
        if affine_constraints is not None
        else np.any(np.linalg.norm(free[indices], axis=1) > rank_bar(state.system_matrix.shape))
    ):
        msg = 'nonlinear initial preferred states require additional physical initial authority'
        raise ValueError(msg)
    return value


class _AffineInitialConstraints:
    """Newton in caller-owned exact affine coordinates, with original constraints retained."""

    is_linear = False

    def __init__(self, original, constraints, tolerance) -> None:
        self.original = original
        self.tolerance = tolerance
        self.matrix, self.rhs = (np.asarray(value, dtype=float) for value in constraints)
        if self.matrix.ndim != 2 or self.matrix.shape[1] != original.initial.size:
            msg = 'affine initial constraints must cover the actual full state'
            raise ValueError(msg)
        self.origin, self.directions = border_affine_space(self.matrix, self.rhs)

    def full_state(self, value):
        state = self.origin + self.directions @ value
        defect = self.matrix @ state - self.rhs
        scale = np.abs(self.matrix) @ np.abs(state) + np.abs(self.rhs)
        if (
            not np.isfinite(state).all()
            or not np.isfinite(scale).all()
            or np.any(np.abs(defect) > self.tolerance * scale)
        ):
            msg = 'affine initial coordinates contradict their original physical equations'
            raise ValueError(msg)
        return state

    def initial_guess(self):
        # This selects only a numerical search seed, never a physical free state.
        return np.linalg.lstsq(self.directions, self.original.initial - self.origin, rcond=None)[0]

    def assemble(self, value):
        state = self.original.assemble(self.full_state(value))
        return AssembledState(state.system_matrix @ self.directions, state.total_energy, payload=state.payload)

    def residual(self, value, state):
        return state.payload

    def residual_norm(self, value, state):
        return float(np.linalg.norm(state.payload))

    def rhs_norm(self, state):
        """The ORIGINAL system's source norm, on the reduced tangent this Newton actually solves.

        Without it the Newton stop fell back to the SEED residual, so a seed already at the root
        (case_l `network_controlled` REST, 1.6e-14) was asked for 1e-10 of
        its own roundoff and refused after 40 iterations at ratio 1.0.
        """
        return self.original.rhs_norm(state)

    def tangent(self, value, state):
        return state.system_matrix

    def solve_system(self, matrix, rhs):
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]

    def on_iteration(self, iteration, state, error, alpha, increment):
        pass


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

    def rhs_norm(self, state):
        """The scale the Newton stop is judged on: the one the final acceptance check uses.

        Without it the stop was relative to the SEED residual, so a seed near the root (a cold
        Shockley bridge whose currents are already ~0) demanded an answer tighter than roundoff
        and never converged -- the `_FiniteRightSystem` sibling owns the same source norm.
        """
        return float(np.linalg.norm(state.system_matrix) * np.linalg.norm(self.initial)) + float(
            np.linalg.norm(self.stored @ self.initial)
        )

    def tangent(self, x, state):
        return state.system_matrix

    def solve_system(self, matrix, rhs):
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]

    def solve_linear(self, state):
        msg = 'initial algebraic constraints require the declared nonlinear Newton owner'
        raise TypeError(msg)

    def on_iteration(self, i, state, err, alpha, increment) -> None:
        pass
