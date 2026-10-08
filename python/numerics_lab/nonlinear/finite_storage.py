"""Initial/right-state projection over an actual finite storage map.

No physical constitutive law lives here. Callers supply Q, DQ, the residual,
and a proven invariant left-nullspace of DQ. A changing storage nullspace
requires its own coordinate authority rather than a constant-mass surrogate.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from motronics.gauss.linear.algebraic_bulk_initial import algebraic_bulk_initial_pair
from motronics.gauss.linear.initial_sparse import projected_rate_consistent, sparse_initial_pair
from motronics.gauss.nonlinear import AssembledState, NewtonStrategy


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
    system = _FiniteRightSystem(initial, residual_at, jacobian_at, mass_at, storage_at, algebraic, selected, tolerance)
    result = NewtonStrategy().solve(system, max_iter=max_newton, rtol_energy_pct=tolerance * 100)
    final = system.assemble(result.x)
    reference = np.linalg.norm(final.system_matrix.data) * np.linalg.norm(result.x) + system.rhs_norm(final)
    if not result.converged or system.residual_norm(result.x, final) > tolerance * reference:
        msg = 'actual finite stored-coordinate projection did not converge'
        raise ValueError(msg)
    return result.x


class _FiniteRightSystem:
    """Physical constraints supplied to the sole shared line-search Newton owner."""

    is_linear = False

    def __init__(self, initial, residual, jacobian, mass, storage, algebraic, selected, tolerance) -> None:
        self.initial, self.evaluate_residual, self.evaluate_jacobian = initial, residual, jacobian
        self.evaluate_mass, self.evaluate_storage = mass, storage
        self.algebraic, self.selected, self.tolerance = algebraic, selected, tolerance
        self.target = np.asarray(storage(initial))[selected]
        matrix = self._matrix(initial)
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
        self.scale = np.where(norms == 0, 1, norms)
        self.source_norm = np.linalg.norm((sp.diags(1 / self.scale) @ matrix).data) * np.linalg.norm(
            initial
        ) + np.linalg.norm(self._residual(initial) / self.scale)

    def _matrix(self, state):
        mass = sp.csr_matrix(self.evaluate_mass(state))
        defect = self.algebraic @ mass
        if np.linalg.norm(defect.data) > self.tolerance * np.linalg.norm(self.algebraic.data) * np.linalg.norm(
            mass.data
        ):
            msg = 'finite storage nullspace changed; its actual nonlinear coordinate owner is required'
            raise ValueError(msg)
        return sp.vstack(
            (self.algebraic @ sp.csr_matrix(self.evaluate_jacobian(state)), mass[self.selected]), format='csr'
        )

    def _residual(self, state):
        return np.concatenate((
            self.algebraic @ self.evaluate_residual(state),
            np.asarray(self.evaluate_storage(state))[self.selected] - self.target,
        ))

    def initial_guess(self):
        return self.initial.copy()

    def assemble(self, state):
        matrix = sp.diags(1 / self.scale) @ self._matrix(state)
        residual = self._residual(state) / self.scale
        return AssembledState(matrix, 0.5 * float(residual @ residual), payload=residual)

    def residual(self, state, assembled):
        return assembled.payload

    def residual_norm(self, state, assembled):
        return float(np.linalg.norm(assembled.payload))

    def rhs_norm(self, assembled):
        return self.source_norm

    def tangent(self, state, assembled):
        return assembled.system_matrix

    def solve_system(self, matrix, rhs):
        try:
            return splu(matrix.tocsc()).solve(rhs)
        except RuntimeError as error:
            msg = 'finite stored-coordinate projection lacks a nonsingular physical algebraic subsystem'
            raise StorageRateAuthorityRequired(msg) from error

    def solve_linear(self, assembled):
        msg = 'finite stored coordinates require the actual nonlinear Newton owner'
        raise TypeError(msg)

    def on_iteration(self, index, assembled, error, alpha, increment):
        pass


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
    algebraic_bulk_size=None,
    stored_columns=None,
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
        pair_solver = sparse_initial_pair if algebraic_bulk_size is None else algebraic_bulk_initial_pair
        coordinates = (
            {'algebraic_rows': algebraic, 'stored_rows': selected, 'stored_columns': stored_columns}
            if algebraic_bulk_size is None
            else {'bulk_size': algebraic_bulk_size}
        )
        pair = pair_solver(
            jacobian,
            residual,
            mass,
            state,
            **coordinates,
            stored_increment=target - np.asarray(storage_at(state))[selected],
            partial_time=partial_time_at(state),
            tolerance=tolerance,
        )
        state = pair.state
        actual_mass, actual_jacobian = sp.csr_matrix(mass_at(state)), sp.csr_matrix(jacobian_at(state))
        actual = np.asarray(storage_at(state))[selected]
        residual_defect = np.asarray(residual_at(state, pair.rate))
        state_scale = np.linalg.norm(actual_jacobian.data) * np.linalg.norm(state) + np.linalg.norm(
            actual_mass @ pair.rate
        )
        storage_scale = (
            np.linalg.norm(actual)
            + np.linalg.norm(target)
            + np.linalg.norm(actual_mass[selected].data) * np.linalg.norm(state)
        )
        state_admitted = (
            np.linalg.norm(residual_defect) <= tolerance * state_scale
            and np.linalg.norm(actual - target) <= tolerance * storage_scale
        )
        # Algebraic rows cannot subsequently be repaired by a rate correction.
        # Do not let an unrelated large circuit coordinate hide their state
        # constraint error, especially when the stored linkage is near zero.
        zero_mass_rows = np.asarray(actual_mass.getnnz(axis=1)).ravel() == 0
        equation_scale = np.asarray(abs(actual_jacobian) @ abs(state)).ravel() + abs(residual_defect)
        algebraic_state_admitted = np.all(
            abs(residual_defect[zero_mass_rows]) <= tolerance * equation_scale[zero_mass_rows]
        )
        if stored_columns is not None:
            state_admitted = state_admitted and algebraic_state_admitted
        if state_admitted and algebraic_bulk_size is not None:
            # Re-evaluate the true constitutive tangent at the now-admitted
            # finite-Q state, not at the preceding Newton loading iterate.
            # Rate consistency must not perturb a state whose storage is fixed.
            pair = algebraic_bulk_initial_pair(
                actual_jacobian,
                residual_at(state, zero),
                actual_mass,
                state,
                bulk_size=algebraic_bulk_size,
                partial_time=partial_time_at(state),
                stored_increment=np.zeros(len(selected)),
                tolerance=tolerance,
                fixed_state=True,
            )
            residual_defect = np.asarray(residual_at(state, pair.rate))
        if (
            state_admitted
            and np.linalg.norm(residual_defect) <= tolerance * state_scale
            and projected_rate_consistent(
                actual_jacobian, pair.rate, partial_time_at(state), algebraic_rows=algebraic, tolerance=tolerance
            )
        ):
            final_defect = algebraic @ actual_mass
            if np.linalg.norm(final_defect.data) <= tolerance * np.linalg.norm(algebraic.data) * np.linalg.norm(
                actual_mass.data
            ):
                return pair
    msg = 'actual finite storage state/rate projection did not converge'
    raise ValueError(msg)
