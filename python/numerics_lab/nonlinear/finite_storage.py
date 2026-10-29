"""Initial/right-state projection over an actual finite storage map.

No physical constitutive law lives here. Callers supply Q, DQ, the residual,
and a proven invariant left-nullspace of DQ. A changing storage nullspace
requires its own coordinate authority rather than a constant-mass surrogate.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from numerics_lab.formulation import ResidualAssemblyBound
from numerics_lab.line_search import backtracking, sufficient_decrease
from numerics_lab.linear.algebraic_bulk_initial import algebraic_bulk_initial_pair
from numerics_lab.linear.arithmetic import affine_residual_bound, certified_sparse_product, transport_residual_bound
from numerics_lab.linear.index_two import (
    certified_null_directions,
    index_two_frozen_rate,
)
from numerics_lab.linear.initial_rate import sparse_initial_rate
from numerics_lab.linear.initial_sparse import (
    SparseInitialPair,
    projected_rate_consistent,
    sparse_initial_pair,
)
from numerics_lab.linear.initial_storage_border import joint_coordinate_sparse_correction
from numerics_lab.linear.resolution import ADMISSION_BAR, system_resolution
from numerics_lab.nonlinear import AssembledState, NewtonStrategy
from numerics_lab.nonlinear.initial_joint import joint_finite_storage_initial_pair

__all__ = [
    'StorageRateAuthorityRequired',
    'finite_storage_actual_rate_initial_pair',
    'finite_storage_initial_pair',
    'finite_storage_right_state',
]


@dataclass(frozen=True)
class _ProjectedDerivativeEvaluation:
    residual: np.ndarray
    bound: ResidualAssemblyBound


class StorageRateAuthorityRequired(ValueError):
    """Finite stored coordinates and undifferentiated constraints do not fix every state."""

    def __init__(self, message, *, state=None, directions=None) -> None:
        super().__init__(message)
        self.state, self.directions = state, directions


def finite_storage_right_state(
    initial, *, residual_at, jacobian_at, mass_at, storage_at, storage_rows, max_newton=30, tolerance=ADMISSION_BAR
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
    residual = system.residual_norm(result.x, final)
    # JUDGED ON THE EQUATION, not on the Newton's stop flag: its increment test can stay unmet at the
    # roundoff floor while the constraints already hold -- measured on push_pull@ccm, whose every
    # march (the case's own included) died here with converged=False at residual 1.9e-15 against a
    # 7.5e-8 bar, inside its first period.
    if residual > tolerance * reference:
        msg = (
            f'actual finite stored-coordinate projection did not converge (newton converged={result.converged}, '
            f'residual {residual:.3e} against {tolerance * reference:.3e})'
        )
        raise ValueError(msg)
    # A SEED ALREADY AT THE ROOT RUNS NO SOLVE (the Newton owner returns it untouched), so the
    # singular-subsystem refusal in `solve_system` never fired and an index-two state -- a voltage
    # source clamping a capacitor, whose current only the source RATE fixes -- came back with its
    # undetermined current left at the seed's 0.0. Determinacy is a property of the subsystem, not
    # of how many iterations it took, so it is checked once on the final tangent.
    system.solve_system(final.system_matrix, np.zeros(final.system_matrix.shape[0]))
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
    tolerance=ADMISSION_BAR,
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


def finite_storage_actual_rate_initial_pair(
    initial,
    *,
    residual_at,
    jacobian_at,
    mass_at,
    storage_at,
    partial_time_at,
    storage_rows_at,
    rate_constraint_jacobian_at,
    rate_constraint_state_columns_at=None,
    partial_time_evaluation_at=None,
    residual_evaluation_at=None,
    storage_evaluation_at=None,
    target_storage_bound=None,
    joint_coordinates_at=None,
    differentiated_residual_at=None,
    differentiated_residual_evaluation_at=None,
    target_storage=None,
    stored_columns_at=None,
    sparse_state_indices=None,
    max_newton=30,
    tolerance=ADMISSION_BAR,
):
    """Admit actual finite Q and full differentiated equations at the actual rate.

    The owner supplies F_u(u,v), F_t(u,v), and the exact rate tangent of
    F_u(u,v)v+F_t(u,v). A changing left nullspace is only an elimination
    basis: differentiating F=0 first removes its acceleration without
    manufacturing null-basis derivatives. Sparse bulk/small-border rank
    construction remains the caller's actual storage owner.
    """

    def construction_allowance(rate, mass, residual, evaluation):
        if evaluation is None:
            return np.zeros_like(residual)
        bound = evaluation.bound
        bound = affine_residual_bound(mass, rate, -residual, bound)
        return bound.allowance()

    state = np.asarray(initial, dtype=float).copy()
    if (
        state.ndim != 1
        or not np.isfinite(state).all()
        or not isinstance(max_newton, (int, np.integer))
        or isinstance(max_newton, (bool, np.bool_))
        or max_newton < 1
        or not np.isfinite(tolerance)
        or tolerance <= 0
    ):
        msg = 'actual-rate finite storage admission requires finite state and positive iteration controls'
        raise ValueError(msg)
    rate = np.zeros_like(state)  # Search seed, never returned as derivative authority without proof.

    def checked_matrix(owner):
        def evaluate(value, velocity):
            matrix = sp.csr_matrix(owner(value, velocity), dtype=float)
            if matrix.shape != (len(state), len(state)) or not np.isfinite(matrix.data).all():
                msg = 'actual-rate admission requires finite actual matrices with the square state shape'
                raise ValueError(msg)
            return matrix

        return evaluate

    def checked_vector(owner):
        def evaluate(value, velocity):
            vector = np.asarray(owner(value, velocity), dtype=float)
            if vector.shape != state.shape or not np.isfinite(vector).all():
                msg = 'actual-rate admission requires finite actual vectors with the state shape'
                raise ValueError(msg)
            return vector

        return evaluate

    mass_at, jacobian_at, rate_constraint_jacobian_at = map(
        checked_matrix, (mass_at, jacobian_at, rate_constraint_jacobian_at)
    )
    residual_at, partial_time_at = map(checked_vector, (residual_at, partial_time_at))

    def evaluate_partial(value, velocity):
        if partial_time_evaluation_at is None:
            return partial_time_at(value, velocity), None
        sample = partial_time_evaluation_at(value, velocity)
        original = np.asarray(sample.residual)
        bound = sample.bound
        if (
            np.iscomplexobj(original)
            or original.dtype.kind not in 'fiu'
            or original.shape != state.shape
            or not np.isfinite(original).all()
            or not isinstance(bound, ResidualAssemblyBound)
            or bound.values.shape != state.shape
        ):
            msg = 'explicit time evaluation requires finite aligned actual partial values and row bounds'
            raise ValueError(msg)
        return np.asarray(original, dtype=float), bound

    def projected_evaluation(jacobian, velocity, partial, bound, algebraic):
        if bound is None:
            return None
        derivative = jacobian @ velocity + partial
        result = transport_residual_bound(
            algebraic, derivative, affine_residual_bound(jacobian, velocity, partial, bound)
        )
        return _ProjectedDerivativeEvaluation(algebraic @ derivative, result)

    def projected_bound(jacobian, velocity, partial, bound, algebraic):
        result = projected_evaluation(jacobian, velocity, partial, bound, algebraic)
        return None if result is None else result.bound

    def forcing_allowance(hidden, velocity, derivative, bound):
        if bound is None:
            return None
        return affine_residual_bound(hidden, velocity, -derivative, bound).allowance()

    sampled: dict[bytes, object] = {}

    def storage_sample(value):
        """The supplied storage producer at *value*, evaluated ONCE: the charge and its allowance read one sample."""
        key = np.asarray(value, dtype=float).tobytes()
        if key not in sampled:
            sampled.clear()
            sampled[key] = storage_evaluation_at(value)
        return sampled[key]

    def charge_at(value):
        charge = np.asarray(
            storage_at(value) if storage_evaluation_at is None else storage_sample(value).storage, dtype=float
        )
        if charge.shape != state.shape or not np.isfinite(charge).all():
            msg = 'actual-rate admission requires finite full charge witnesses with the state vector shape'
            raise ValueError(msg)
        return charge

    target = charge_at(state).copy() if target_storage is None else np.asarray(target_storage, dtype=float).copy()
    if target.shape != state.shape or not np.isfinite(target).all():
        msg = 'actual-rate admission requires an aligned finite full storage target'
        raise ValueError(msg)
    if target_storage_bound is None and target_storage is None and storage_evaluation_at is not None:
        target_storage_bound = storage_sample(state).bound
    if target_storage_bound is not None and (
        not isinstance(target_storage_bound, ResidualAssemblyBound) or target_storage_bound.values.shape != state.shape
    ):
        msg = 'target storage arithmetic requires aligned actual row provenance'
        raise ValueError(msg)

    def storage_allowance(value):
        if storage_evaluation_at is None or target_storage_bound is None:
            return np.zeros_like(state)
        sample = storage_sample(value)
        if sample.storage.shape != state.shape or sample.bound.values.shape != state.shape:
            msg = 'storage arithmetic requires aligned actual row provenance'
            raise ValueError(msg)
        producers = ResidualAssemblyBound(
            target_storage_bound.values + sample.bound.values,
            target_storage_bound.known & sample.bound.known,
            target_storage_bound.law + sample.bound.law,
        )
        # The only new arithmetic is target-Q. Neither DQ*state nor residual/source
        # magnitudes certify how either finite-storage producer evaluated its Q.
        return affine_residual_bound(-sp.eye(len(state), format='csr'), sample.storage, target, producers).allowance()

    # A HOMOGENEOUS system (its own forcing, the residual at rest F(0, 0, t), exactly zero) whose target
    # storage is zero under the floor no system resolves is AT REST: (0, 0) satisfies F = 0 and Q = 0
    # exactly, so the search starts there. Seeded from the incoming state, it walked down into gradual
    # underflow instead (1.6e-11 A -> 2.8e-309 A after a pulse, 2026-10-14) and judged noise as forcing.
    # Uniqueness is still certified by the admission below, from this seed.
    rest = np.zeros_like(state)
    if np.any(state) and np.all(np.abs(target) <= system_resolution()) and not np.any(residual_at(rest, rest)):
        state, target = rest, rest.copy()

    if joint_coordinates_at is not None:
        return joint_finite_storage_initial_pair(
            state,
            target,
            joint_coordinates_at,
            residual_at,
            jacobian_at,
            mass_at,
            charge_at,
            lambda value, velocity: evaluate_partial(value, velocity)[0],
            storage_rows_at,
            max_newton,
            tolerance,
            residual_evaluation_at=residual_evaluation_at,
            sparse_state_indices=sparse_state_indices,
            differentiated_residual_at=differentiated_residual_at,
            differentiated_residual_evaluation_at=differentiated_residual_evaluation_at,
            storage_allowance_at=storage_allowance,
        )

    # First separate state authority wherever Q and the actual algebraic
    # equations already determine it. A rate RHS must not introduce numerical
    # corrections into an exactly admitted homogeneous storage/linkage state.
    # An index-two state that needs source-rate authority takes the coupled
    # physical equations below instead, not a guessed algebraic state.
    try:
        state = _actual_finite_storage_state(
            state,
            target,
            residual_at,
            jacobian_at,
            mass_at,
            charge_at,
            storage_rows_at,
            max_newton,
            tolerance,
            stored_columns_at,
            sparse_state_indices,
            residual_evaluation_at,
        )
    except StorageRateAuthorityRequired as authority:
        if authority.directions is None and rate_constraint_state_columns_at is not None:
            msg = 'mixed boundary state columns require an issued bounded state range'
            raise NotImplementedError(msg) from authority
        if authority.directions is not None:
            state = authority.state
            directions = authority.directions
            for _iteration in range(max_newton):
                mass, jacobian = mass_at(state, rate), jacobian_at(state, rate)
                partial, partial_bound = evaluate_partial(state, rate)
                tangent = rate_constraint_jacobian_at(state, rate)
                storage = storage_rows_at(state, rate)
                evaluation = None if residual_evaluation_at is None else residual_evaluation_at(state, rate)
                residual = residual_at(state, rate) if evaluation is None else evaluation.residual
                derivative = storage.algebraic_rows @ (jacobian @ rate + partial)
                derivative_bound = projected_bound(jacobian, rate, partial, partial_bound, storage.algebraic_rows)
                # The CERTIFIED hidden rows, as on the frozen branch: a left-null row against a
                # column in the range of the mass cancels to residue, and the border would pivot on it.
                hidden = certified_sparse_product(storage.algebraic_rows, tangent)
                difference = sp.csr_matrix(tangent) - sp.csr_matrix(jacobian)
                difference.eliminate_zeros()
                if not difference.nnz:
                    # These actual rate columns ARE J. Their absolute-rate
                    # equation is Y*J*v_new = -Y*Ft, not the subtraction of
                    # separately rounded Y*(J*v+Ft) and (Y*J)*v. The latter
                    # manufactures driven forcing on genuinely homogeneous
                    # rate constraints. No source value is probed or inferred.
                    hidden_rhs = -np.asarray(storage.algebraic_rows @ partial)
                    forcing_bound = (
                        None
                        if partial_bound is None
                        else transport_residual_bound(storage.algebraic_rows, partial, partial_bound)
                    )
                    hidden_allowance = None if forcing_bound is None else forcing_bound.allowance()
                else:
                    hidden_rhs = -derivative + hidden @ rate
                    hidden_allowance = forcing_allowance(hidden, rate, derivative, derivative_bound)
                if hidden_allowance is None:
                    hidden_allowance = np.zeros(storage.algebraic_rows.shape[0])
                mixed = sp.csr_matrix((len(state), directions.shape[1]))
                if rate_constraint_state_columns_at is not None:
                    supplied = rate_constraint_state_columns_at(state, rate, directions)
                    entries = supplied.data if sp.issparse(supplied) else np.asarray(supplied, dtype=object)
                    mixed = sp.csr_matrix(supplied)
                    if (
                        mixed.shape != (len(state), directions.shape[1])
                        or mixed.dtype.kind not in 'fiu'
                        or not np.isfinite(mixed.data).all()
                        or any(isinstance(value, (bool, np.bool_)) for value in np.asarray(entries).flat)
                    ):
                        msg = 'mixed boundary state columns require finite real actual state-range dimensions'
                        raise ValueError(msg)
                mixed_hidden = certified_sparse_product(storage.algebraic_rows, mixed)
                equations = sp.bmat(
                    [
                        [jacobian @ directions, mass],
                        [
                            mixed_hidden,
                            hidden,
                        ],
                    ],
                    format='csr',
                )
                rhs = np.concatenate((-residual + mass @ rate, hidden_rhs))
                algebraic_state = np.asarray(mass.getnnz(axis=1)).ravel() == 0
                state_allowance = tolerance * (
                    abs(jacobian) @ abs(state) + abs(residual_at(state, np.zeros_like(rate)))
                )
                state_allowance[~algebraic_state] = 0.0
                state_allowance += construction_allowance(rate, mass, residual, evaluation)
                if sparse_state_indices is not None and len(sparse_state_indices):
                    stored = mass[storage.stored_rows]
                    original = sp.bmat(
                        [
                            [jacobian, mass],
                            [stored, sp.csr_matrix(stored.shape)],
                            [sp.csr_matrix(storage.algebraic_rows.shape), hidden],
                        ],
                        format='csr',
                    )
                    actual_charge = charge_at(state)
                    charge_scale = abs(actual_charge) + abs(target) + abs(mass) @ abs(state)
                    if np.any(abs(actual_charge - target) > tolerance * charge_scale):
                        msg = 'boundary particular contradicts original finite charge witnesses'
                        raise ValueError(msg)
                    # Preserve this already admitted particular. The remaining
                    # correction lies in homogeneous stored-coordinate directions;
                    # the unchanged target still owns the final full-Q guard.
                    forcing = np.concatenate((
                        rhs[: len(state)],
                        np.zeros(len(storage.stored_rows)),
                        rhs[len(state) :],
                    ))
                    correction, unique = joint_coordinate_sparse_correction(
                        original,
                        forcing,
                        mass=mass,
                        algebraic_rows=storage.algebraic_rows,
                        stored_rows=storage.stored_rows,
                        indices=sparse_state_indices,
                        state_directions=directions,
                        state_direction_columns=sp.vstack(
                            (sp.csr_matrix((len(state) + len(storage.stored_rows), directions.shape[1])), mixed_hidden),
                            format='csr',
                        ),
                        residual_allowance=np.concatenate((
                            state_allowance,
                            np.zeros(len(storage.stored_rows)),
                            hidden_allowance,
                        )),
                        row_scale=np.concatenate((
                            abs(jacobian) @ abs(state),
                            abs(stored) @ abs(state),
                            np.zeros(storage.algebraic_rows.shape[0]),
                        )),
                        tolerance=tolerance,
                    )
                    candidate = state + correction[: len(state)]
                    rate = correction[len(state) :]
                    rate_unique = unique[len(state) :]
                else:
                    solution, unique = sparse_initial_rate(
                        equations,
                        rhs,
                        tolerance=tolerance,
                        residual_allowance=np.concatenate((state_allowance, hidden_allowance)),
                    )
                    count = directions.shape[1]
                    if not unique[:count].all():
                        msg = 'original boundary state freedom lacks unique source-selected authority'
                        raise ValueError(msg)
                    candidate = state + directions @ solution[:count]
                    rate = solution[count:]
                    rate_unique = unique[count:]
                charge = charge_at(candidate)
                charge_scale = abs(charge) + abs(target) + abs(mass_at(candidate, rate)) @ abs(candidate)
                if np.any(abs(charge - target) > tolerance * charge_scale):
                    msg = 'boundary state direction contradicts original finite charge witnesses'
                    raise ValueError(msg)
                state = candidate
                mass, jacobian = mass_at(state, rate), jacobian_at(state, rate)
                residual = residual_at(state, rate)
                partial, partial_bound = evaluate_partial(state, rate)
                scale = (
                    abs(jacobian) @ abs(state) + abs(mass) @ abs(rate) + abs(residual_at(state, np.zeros_like(rate)))
                )
                storage = storage_rows_at(state, rate)
                if np.all(abs(residual) <= tolerance * scale) and projected_rate_consistent(
                    jacobian,
                    rate,
                    partial,
                    algebraic_rows=storage.algebraic_rows,
                    tolerance=tolerance,
                    differentiated_evaluation=projected_evaluation(
                        jacobian, rate, partial, partial_bound, storage.algebraic_rows
                    ),
                ):
                    return SparseInitialPair(state, rate, rate_unique)
            msg = 'original boundary state/rate equations did not converge'
            raise ValueError(msg)
    else:
        state_mass = sp.csr_matrix(mass_at(state, rate))
        actual_charge = charge_at(state)
        charge_scale = abs(actual_charge) + abs(target) + np.asarray(abs(state_mass) @ abs(state)).ravel()
        if np.any(abs(actual_charge - target) > tolerance * charge_scale):
            msg = 'actual finite storage state candidate contradicts original full charge witnesses'
            raise ValueError(msg)
        for _iteration in range(max_newton):
            mass = sp.csr_matrix(mass_at(state, rate))
            jacobian = sp.csr_matrix(jacobian_at(state, rate))
            partial, partial_bound = evaluate_partial(state, rate)
            tangent = sp.csr_matrix(rate_constraint_jacobian_at(state, rate))
            storage = storage_rows_at(state, rate)
            evaluation = None if residual_evaluation_at is None else residual_evaluation_at(state, rate)
            residual = np.asarray(residual_at(state, rate) if evaluation is None else evaluation.residual)
            derivative = storage.algebraic_rows @ (jacobian @ rate + partial)
            derivative_bound = projected_bound(jacobian, rate, partial, partial_bound, storage.algebraic_rows)
            # Exact zero-mass rows are already admitted STATE equations, not
            # dynamic rate forcing. Preserve their same physical state bar;
            # differentiated rows keep only their independently owned bound.
            mass.eliminate_zeros()
            algebraic_state = np.asarray(mass.getnnz(axis=1)).ravel() == 0
            state_allowance = tolerance * (
                np.asarray(abs(jacobian) @ abs(state)).ravel()
                + abs(np.asarray(residual_at(state, np.zeros_like(rate))))
            )
            state_allowance[~algebraic_state] = 0.0
            state_allowance += construction_allowance(rate, mass, residual, evaluation)
            # The hidden rows are the CERTIFIED product: a left-null row against a column in the
            # range of the mass cancels to rounding residue, and a pivot on that residue absorbs a
            # junk rate (case_a at its magnetostatic rest: 1.2e21 on the solid bodies' voltages).
            # A solid body's voltage is index two -- no undifferentiated row fixes it, so a frozen
            # state carried from the seed contradicts the hidden constraint and is re-solved with
            # the rates, exactly as the joint path's frozen solve does.
            hidden = certified_sparse_product(storage.algebraic_rows, tangent)
            columns, state_step, rate, _unique = index_two_frozen_rate(
                storage.algebraic_rows,
                hidden,
                mass,
                jacobian,
                hidden @ rate - derivative,
                mass @ rate - residual,
                tolerance=tolerance,
                allowance=state_allowance,
                differentiated_allowance=forcing_allowance(hidden, rate, derivative, derivative_bound),
            )
            if len(columns):
                state = state.copy()
                state[columns] += state_step
            final_mass = sp.csr_matrix(mass_at(state, rate))
            final_jacobian = sp.csr_matrix(jacobian_at(state, rate))
            final_partial, final_partial_bound = evaluate_partial(state, rate)
            final_tangent = sp.csr_matrix(rate_constraint_jacobian_at(state, rate))
            final_storage = storage_rows_at(state, rate)
            final_residual = np.asarray(residual_at(state, rate))
            residual_scale = (
                np.asarray(abs(final_jacobian) @ abs(state)).ravel()
                + np.asarray(abs(final_mass) @ abs(rate)).ravel()
                + abs(np.asarray(residual_at(state, np.zeros_like(rate))))
            )
            # Judged to the allowance the frozen rate was solved to: the bare bar alone refused rows the
            # solve already admitted, and the stage became a fixed point (sdm k = 2 hysteretic, 2026-10-19).
            admitted = np.all(abs(final_residual) <= tolerance * residual_scale) or (
                residual_evaluation_at is not None
                and _within_allowance(
                    final_residual,
                    tolerance * residual_scale,
                    construction_allowance(rate, final_mass, final_residual, residual_evaluation_at(state, rate)),
                )
            )
            if admitted and projected_rate_consistent(
                final_jacobian,
                rate,
                final_partial,
                algebraic_rows=final_storage.algebraic_rows,
                tolerance=tolerance,
                differentiated_evaluation=projected_evaluation(
                    final_jacobian, rate, final_partial, final_partial_bound, final_storage.algebraic_rows
                ),
            ):
                # State search used a frozen-nullspace quasi-Newton tangent.
                # Its nonsingularity is not the final physical state proof:
                # Re-factor the homogeneous perturbation equations at this
                # actual pair. Re-admitting nonzero rate forcing here would
                # introduce cancellation noise into an already frozen state.
                sparse_initial_pair(
                    final_jacobian,
                    np.zeros_like(state),
                    final_mass,
                    np.zeros_like(state),
                    algebraic_rows=final_storage.algebraic_rows,
                    stored_rows=final_storage.stored_rows,
                    partial_time=np.zeros_like(state),
                    stored_increment=np.zeros(len(final_storage.stored_rows)),
                    rate_jacobian=final_tangent,
                    stored_columns=None if stored_columns_at is None else stored_columns_at(state, rate, final_storage),
                    sparse_state_indices=sparse_state_indices,
                    tolerance=tolerance,
                )
                _, unique = sparse_initial_rate(
                    sp.vstack(
                        (final_mass, certified_sparse_product(final_storage.algebraic_rows, final_tangent)),
                        format='csr',
                    ),
                    np.zeros(len(state) + final_storage.algebraic_rows.shape[0]),
                    tolerance=tolerance,
                )
                return SparseInitialPair(state, rate, unique)
        msg = 'actual-rate finite storage frozen-state rate projection did not converge'
        raise ValueError(msg)

    for _iteration in range(max_newton):
        mass = sp.csr_matrix(mass_at(state, rate))
        jacobian = sp.csr_matrix(jacobian_at(state, rate))
        partial, partial_bound = evaluate_partial(state, rate)
        rate_tangent = sp.csr_matrix(rate_constraint_jacobian_at(state, rate))
        storage = storage_rows_at(state, rate)
        evaluation = None if residual_evaluation_at is None else residual_evaluation_at(state, rate)
        residual = np.asarray(residual_at(state, rate) if evaluation is None else evaluation.residual)
        linear_partial = jacobian @ rate + partial - rate_tangent @ rate
        linear_bound = None
        if partial_bound is not None:
            derivative_bound = affine_residual_bound(jacobian, rate, partial, partial_bound)
            linear_bound = affine_residual_bound(-rate_tangent, rate, jacobian @ rate + partial, derivative_bound)
            linear_bound = transport_residual_bound(storage.algebraic_rows, linear_partial, linear_bound).allowance()
        # Quasi-Newton search: freeze the state dependence of the projected
        # derivative equation, but use its EXACT rate tangent. Acceptance
        # below never substitutes this search linearization for physics.
        # The residual's certified rounding travels with it into the joint
        # elimination: a solid body's dependent net-current row receives the
        # image of its conducting field rows' rounding, which grows with the
        # excitation (case_a dual-star PWM: 4.5e-8 against a 8.6e-9 bar).
        try:
            pair = sparse_initial_pair(
                jacobian,
                residual - mass @ rate,
                mass,
                state,
                algebraic_rows=storage.algebraic_rows,
                stored_rows=storage.stored_rows,
                partial_time=linear_partial,
                stored_increment=(target - charge_at(state))[storage.stored_rows],
                stored_columns=None if stored_columns_at is None else stored_columns_at(state, rate, storage),
                sparse_state_indices=sparse_state_indices,
                rate_jacobian=rate_tangent,
                tolerance=tolerance,
                residual_allowance=construction_allowance(rate, mass, residual, evaluation),
                stored_allowance=storage_allowance(state)[storage.stored_rows],
                differentiated_allowance=linear_bound,
            )
        except ValueError as error:
            msg = 'actual-rate finite storage equations lack a compatible physical initial pair'
            raise ValueError(msg) from error
        state, rate = pair.state, pair.rate
        mass = sp.csr_matrix(mass_at(state, rate))
        jacobian = sp.csr_matrix(jacobian_at(state, rate))
        partial, partial_bound = evaluate_partial(state, rate)
        storage = storage_rows_at(state, rate)
        actual_q = charge_at(state)
        residual = np.asarray(residual_at(state, rate))
        # ALL original charge witnesses, not merely a locally independent
        # subset: e.g. Q=(x,x²) cannot exchange the incoming sign branch.
        charge_scale = abs(actual_q) + abs(target) + np.asarray(abs(mass) @ abs(state)).ravel()
        residual_scale = (
            np.asarray(abs(jacobian) @ abs(state)).ravel()
            + np.asarray(abs(mass) @ abs(rate)).ravel()
            + abs(np.asarray(residual_at(state, np.zeros_like(rate))))
        )
        annihilation = storage.algebraic_rows @ mass
        if (
            np.all(abs(actual_q - target) <= tolerance * charge_scale)
            and np.all(abs(residual) <= tolerance * residual_scale)
            and np.linalg.norm(annihilation.data)
            <= tolerance * np.linalg.norm(storage.algebraic_rows.data) * np.linalg.norm(mass.data)
            and projected_rate_consistent(
                jacobian,
                rate,
                partial,
                algebraic_rows=storage.algebraic_rows,
                tolerance=tolerance,
                differentiated_evaluation=projected_evaluation(
                    jacobian, rate, partial, partial_bound, storage.algebraic_rows
                ),
            )
        ):
            final_tangent = sp.csr_matrix(rate_constraint_jacobian_at(state, rate))
            # Uniqueness asks for homogeneous directions at this accepted pair,
            # not a second nonhomogeneous representative from rounded defects.
            # The original full F/Q/G guards above remain the admission owner.
            final_pair = sparse_initial_pair(
                jacobian,
                np.zeros_like(residual),
                mass,
                state,
                algebraic_rows=storage.algebraic_rows,
                stored_rows=storage.stored_rows,
                partial_time=np.zeros_like(partial),
                stored_increment=np.zeros(len(storage.stored_rows)),
                stored_columns=None if stored_columns_at is None else stored_columns_at(state, rate, storage),
                sparse_state_indices=sparse_state_indices,
                rate_jacobian=final_tangent,
                tolerance=tolerance,
            )
            return type(pair)(state, rate, final_pair.rate_is_unique)
    msg = 'actual-rate finite storage state/rate projection did not converge'
    raise ValueError(msg)


def _actual_finite_storage_state(
    initial,
    target,
    residual_at,
    jacobian_at,
    mass_at,
    charge_at,
    storage_rows_at,
    max_newton,
    tolerance,
    stored_columns_at,
    sparse_state_indices,
    residual_evaluation_at=None,
):
    """Quasi-Newton state search; full Q and actual algebraic rows decide admission.

    A row is admitted to its tolerance PLUS its residual's LAW term (`ResidualAssemblyBound.law`): a row whose
    law is certified only to the inverse's bar cannot be asked below it -- the joint and pair stages read the
    same evaluation. This stage counted no arithmetic allowance before the law term existed and counts none now.
    """
    state = initial.copy()
    zero = np.zeros_like(state)
    for _iteration in range(max_newton):
        mass = sp.csr_matrix(mass_at(state, zero))
        jacobian = sp.csr_matrix(jacobian_at(state, zero))
        storage = storage_rows_at(state, zero)
        charge = charge_at(state)
        residual = np.asarray(residual_at(state, zero))
        projected = storage.algebraic_rows @ residual
        columns = None if stored_columns_at is None else stored_columns_at(state, zero, storage)
        base = np.zeros_like(state)
        mapping = sp.eye(len(state), format='csr')
        if columns is not None and len(columns):
            columns = np.asarray(columns)
            free = np.setdiff1d(np.arange(len(state)), columns)
            stored = mass[storage.stored_rows]
            block = stored[:, columns].tocsc()
            storage_factor = splu(block)
            base[columns] = storage_factor.solve((target - charge)[storage.stored_rows])
            coupling = stored[:, free].tocsc()
            active = np.flatnonzero(np.asarray(coupling.getnnz(axis=0)).ravel())
            # Do not build a field-by-field response to structurally zero
            # stored/free coupling. Only actual nonzero boundary columns
            # require a sparse stored solve.
            response = sp.csr_matrix((len(columns), len(free)))
            if len(active):
                solved = storage_factor.solve(-coupling[:, active].toarray())
                response = sp.csr_matrix(solved) @ sp.eye(len(free), format='csr')[active]
            selectors = sp.eye(len(state), format='csr')
            mapping = selectors[:, free] + selectors[:, columns] @ response
            left, right = storage.algebraic_rows, (jacobian @ mapping).tocsc()
            matrix = (left @ right).tocsc()
            rhs = -projected - storage.algebraic_rows @ jacobian @ base
        else:
            left = sp.block_diag((storage.algebraic_rows, sp.eye(len(storage.stored_rows))), format='csr')
            right = sp.vstack((jacobian, mass[storage.stored_rows]), format='csc')
            matrix = sp.vstack((storage.algebraic_rows @ jacobian, mass[storage.stored_rows]), format='csc')
            rhs = -np.concatenate((projected, (charge - target)[storage.stored_rows]))
            free = np.arange(len(state))
        if sparse_state_indices is not None:
            boundary = np.flatnonzero(~np.isin(free, sparse_state_indices))
            if 0 < len(boundary) <= 512:
                # The boundary's freedom is the CERTIFIED null space: a solid body's voltage column
                # cancels to rounding residue against the left-null rows, and two bodies sharing a
                # field node cancel only as a pair, so an exact rational dependency misses both.
                null = certified_null_directions(left, right[:, boundary])
                if null.shape[1]:
                    msg = 'original boundary state equations require actual differentiated rate authority'
                    charge_scale = abs(charge) + abs(target) + abs(mass) @ abs(state)
                    projected_scale = abs(storage.algebraic_rows) @ (abs(jacobian) @ abs(state) + abs(residual))
                    admitted = np.all(abs(charge - target) <= tolerance * charge_scale) and np.all(
                        abs(projected) <= tolerance * projected_scale
                    )
                    if not admitted:
                        # The certified boundary nullspace is surviving freedom,
                        # not permission to leave independent field/state equations
                        # unsolved. Obtain a numerical particular first, retaining
                        # every original F/Q equation as the nonlinear validator.
                        # Only the bounded boundary product receives its existing
                        # contraction certificate; other state columns stay intact.
                        actual = matrix.tolil(copy=True)
                        actual[:, boundary] = certified_sparse_product(left, right[:, boundary])
                        # This is a numerical state correction, not a physical
                        # rate datum. Its already-admitted cancellation rows keep
                        # the ORIGINAL F/Q accuracy, not a new relative demand
                        # against their tiny correction RHS. The nonlinear F/Q
                        # validator below still decides the candidate's admission.
                        allowance = tolerance * projected_scale
                        if columns is None or not len(columns):
                            allowance = np.concatenate((allowance, tolerance * charge_scale[storage.stored_rows]))
                        particular, _unique = sparse_initial_rate(
                            actual.tocsr(), rhs, tolerance=tolerance, residual_allowance=allowance
                        )
                        step = base + mapping @ particular
                        state = _backtracked_state(state, step, target, residual_at, charge_at, storage, zero)
                        continue
                    directions = sp.csr_matrix(mapping[:, boundary] @ null)
                    raise StorageRateAuthorityRequired(msg, state=state, directions=directions)
        try:
            factor = splu(matrix)
        except RuntimeError as error:
            msg = 'finite stored coordinates require actual differentiated rate authority to determine state'
            raise StorageRateAuthorityRequired(msg) from error
        charge_scale = abs(charge) + abs(target) + np.asarray(abs(mass) @ abs(state)).ravel()
        projected_scale = abs(storage.algebraic_rows) @ (np.asarray(abs(jacobian) @ abs(state)).ravel() + abs(residual))
        if np.all(abs(charge - target) <= tolerance * charge_scale) and (
            np.all(abs(projected) <= tolerance * projected_scale)
            or _within_law(residual_evaluation_at, state, zero, storage, projected, tolerance * projected_scale)
        ):
            return state
        step = base + mapping @ factor.solve(rhs)
        state = _backtracked_state(state, step, target, residual_at, charge_at, storage, zero)
    msg = 'actual finite storage state equations did not converge with all original charge witnesses'
    raise ValueError(msg)


def _within_law(residual_evaluation_at, state, rate, storage, projected, bar) -> bool:
    """Whether every projected row meets ``bar`` plus its LAW term -- read only where the plain bar refused.

    A law-free evaluation (every term zero) answers as the plain bar did, so a closed-form law's admission is
    bit-identical; without an evaluation there is no term.
    """
    if residual_evaluation_at is None:
        return False
    law = residual_evaluation_at(state, rate).bound.law
    if not np.any(law):
        return False
    return bool(np.all(abs(projected) <= bar + np.asarray(abs(storage.algebraic_rows) @ law).ravel()))


def _within_allowance(residual, bar, allowance) -> bool:
    """Whether every row meets ``bar`` plus its certified construction ``allowance``; a zero allowance is the bar."""
    return bool(np.all(abs(np.asarray(residual)) <= bar + allowance))


def _backtracked_state(state, step, target, residual_at, charge_at, storage, zero, *, shrink=0.5, armijo=1e-4):
    """``state + lam * step`` at the first ``lam`` in ``1, shrink, ...`` that decreases the state equations.

    The same Armijo rule as :class:`~numerics_lab.nonlinear._strategies_newton.NewtonStrategy`, on
    the norm of what this search solves: the projected algebraic rows and the stored-charge mismatch.
    A full step on a saturating field can overshoot into a two-cycle that never converges; near the
    root ``lam = 1`` is accepted, so the search is Newton there. When no ``lam`` decreases the norm
    (the roundoff floor) the full step is taken, as plain Newton would.
    """

    def norm(value):
        mismatch = (charge_at(value) - target)[storage.stored_rows]
        return float(np.linalg.norm(np.concatenate((storage.algebraic_rows @ residual_at(value, zero), mismatch))))

    current = norm(state)
    decreases = sufficient_decrease(current, -current, c1=armijo)
    search = backtracking(
        lambda lam: (state + lam * step, norm(state + lam * step)),
        lambda lam, t: decreases(lam, t[1]),
        tries=13,
        shrink=shrink,
    )
    return search.trial[0] if search.accepted else state + step
