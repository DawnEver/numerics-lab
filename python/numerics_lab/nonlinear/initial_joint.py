"""Explicit joint-coordinate finite-storage initial admission."""

import numpy as np
import scipy.sparse as sp

from numerics_lab.formulation import InitialJointEvaluation, ResidualAssemblyBound
from numerics_lab.linear.arithmetic import transport_residual_bound
from numerics_lab.linear.initial_rate import sparse_initial_rate
from numerics_lab.linear.initial_sparse import SparseInitialPair
from numerics_lab.linear.initial_storage_border import joint_coordinate_sparse_correction

__all__ = ['joint_finite_storage_initial_pair']


def joint_finite_storage_initial_pair(
    state,
    target,
    coordinates_at,
    residual_at,
    jacobian_at,
    mass_at,
    charge_at,
    partial_time_at,
    storage_rows_at,
    max_newton,
    tolerance,
    *,
    residual_evaluation_at=None,
    sparse_state_indices=None,
    differentiated_residual_at=None,
    differentiated_residual_evaluation_at=None,
    storage_allowance_at=None,
):
    """One sparse chart for search and the final homogeneous physical proof.

    In particular, the owner forms joint columns before cancellation-sensitive
    contractions. Neither a numerical nullspace nor ``J + M T`` supplies them.
    Full stored-state rows remain witnesses even when they are dependent.
    """
    n = len(state)
    independent = np.zeros(n)
    if differentiated_residual_at is None and differentiated_residual_evaluation_at is None:
        msg = 'joint initial coordinates require original lifted differentiation authority'
        raise NotImplementedError(msg)

    def solve(matrix, defect, mass, storage, storage_allowance=None, differentiated_allowance=None):
        allowance = None
        if storage_allowance is not None or differentiated_allowance is not None:
            allowance = np.concatenate((
                np.zeros(n),
                np.zeros(n) if storage_allowance is None else storage_allowance,
                np.zeros(matrix.shape[0] - 2 * n) if differentiated_allowance is None else differentiated_allowance,
            ))
        if sparse_state_indices is None:
            return sparse_initial_rate(matrix, -defect, tolerance=tolerance, residual_allowance=allowance)
        # Independent storage rows select the solve, never replace the full Q
        # witnesses checked below. All residual/differentiated rows survive.
        selected = np.concatenate((np.arange(n), n + storage.stored_rows, np.arange(2 * n, matrix.shape[0])))
        return joint_coordinate_sparse_correction(
            matrix[selected],
            -defect[selected],
            mass=mass,
            algebraic_rows=storage.algebraic_rows,
            stored_rows=storage.stored_rows,
            indices=sparse_state_indices,
            residual_allowance=None if allowance is None else allowance[selected],
        )

    def evaluate(value, velocity):
        sample = coordinates_at(value, velocity)
        if not isinstance(sample, InitialJointEvaluation) or sample.physical_rate.shape != (n,):
            msg = 'joint coordinates require the declared evaluation and physical state shape'
            raise ValueError(msg)
        matrices = []
        for name in (
            'residual_state_columns',
            'residual_rate_columns',
            'differentiated_state_columns',
            'differentiated_rate_columns',
            'physical_rate_state_map',
            'physical_rate_rate_map',
        ):
            owned = getattr(sample, name)
            if owned is None:
                msg = 'joint coordinates lack actual differentiated column authority'
                raise NotImplementedError(msg)
            matrix = sp.csr_matrix(owned, copy=True)
            if np.iscomplexobj(matrix.data):
                msg = 'joint coordinate columns require real physical authority'
                raise ValueError(msg)
            matrix = matrix.astype(float, copy=False)
            if matrix.shape != (n, n) or not np.isfinite(matrix.data).all():
                msg = 'joint coordinate columns require finite actual square state matrices'
                raise ValueError(msg)
            matrix.eliminate_zeros()
            matrices.append(matrix)
        if sample.differentiated_value is None or sample.physical_acceleration_offset is None:
            msg = 'joint initial coordinates lack issued differentiated value and acceleration lift'
            raise NotImplementedError(msg)
        return (
            sample.physical_rate,
            matrices,
            sample.differentiated_value,
            sample.physical_acceleration_offset,
            sample.differentiated_bound,
        )

    def equations(value, velocity):
        physical, columns, differentiated, acceleration_offset, differentiated_bound = evaluate(value, velocity)
        state_columns, rate_columns, derivative_state, derivative_rate, _, _ = columns
        mass, jacobian = mass_at(value, physical), jacobian_at(value, physical)
        storage = storage_rows_at(value, physical)
        evaluation = None if residual_evaluation_at is None else residual_evaluation_at(value, physical)
        residual = residual_at(value, physical) if evaluation is None else evaluation.residual
        partial = partial_time_at(value, physical)
        derivative = storage.algebraic_rows @ differentiated
        original_bound = None
        if differentiated_residual_evaluation_at is None:
            original_derivative = np.asarray(differentiated_residual_at(value, physical, acceleration_offset))
        else:
            original = differentiated_residual_evaluation_at(value, physical, acceleration_offset)
            original_derivative, original_bound = np.asarray(original.residual), original.bound
            if not isinstance(original_bound, ResidualAssemblyBound) or original_bound.values.shape != (n,):
                msg = 'original differentiated evaluation requires aligned actual row bounds'
                raise ValueError(msg)
        if (
            original_derivative.shape != (n,)
            or original_derivative.dtype.kind not in 'fiu'
            or not np.isfinite(original_derivative).all()
        ):
            msg = 'original lifted differentiation requires a finite aligned physical value'
            raise ValueError(msg)
        matrix = sp.bmat(
            [
                [state_columns, rate_columns],
                [mass, sp.csr_matrix((n, n))],
                [storage.algebraic_rows @ derivative_state, storage.algebraic_rows @ derivative_rate],
            ],
            format='csr',
        )
        defect = np.concatenate((residual, charge_at(value) - target, derivative))
        return (
            physical,
            columns,
            mass,
            jacobian,
            storage,
            matrix,
            defect,
            partial,
            original_derivative,
            acceleration_offset,
            (
                None
                if differentiated_bound is None
                else transport_residual_bound(storage.algebraic_rows, differentiated, differentiated_bound).allowance()
            ),
            (
                np.zeros(storage.algebraic_rows.shape[0])
                if original_bound is None
                else transport_residual_bound(storage.algebraic_rows, original_derivative, original_bound).allowance()
            ),
        )

    for _iteration in range(max_newton):
        (
            physical,
            columns,
            mass,
            jacobian,
            storage,
            matrix,
            defect,
            partial,
            original_derivative,
            acceleration_offset,
            chart_allowance,
            original_allowance,
        ) = equations(state, independent)
        # Q and chart differentiation each keep their own producer certificate;
        # neither inherits the other equation's construction allowance.
        storage_allowance = None if storage_allowance_at is None else storage_allowance_at(state)
        change, unique = solve(matrix, defect, mass, storage, storage_allowance, chart_allowance)
        if not unique[:n].all():
            msg = 'joint initial coordinates lack unique original physical state authority'
            raise ValueError(msg)
        state = state + change[:n]
        independent = independent + change[n:]
        (
            physical,
            columns,
            mass,
            jacobian,
            storage,
            matrix,
            defect,
            partial,
            original_derivative,
            acceleration_offset,
            chart_allowance,
            original_allowance,
        ) = equations(state, independent)
        residual, charge = defect[:n], defect[n : 2 * n] + target
        # The source reference belongs to this exact evaluated residual. A
        # second callback at zero rate can be a different future source value.
        residual_scale = abs(jacobian) @ abs(state) + abs(mass) @ abs(physical) + abs(residual - mass @ physical)
        charge_scale = abs(charge) + abs(target) + abs(mass) @ abs(state)
        if (
            np.any(abs(residual) > tolerance * residual_scale)
            or np.any(abs(charge - target) > tolerance * charge_scale)
            or np.any(
                abs(storage.algebraic_rows @ original_derivative)
                > tolerance
                * (
                    abs(storage.algebraic_rows)
                    @ (abs(jacobian) @ abs(physical) + abs(mass) @ abs(acceleration_offset) + abs(partial))
                )
                + original_allowance
            )
        ):
            continue
        # Re-evaluate and factor the same actual chart, never the rounded raw
        # state/rate matrix whose spurious binary rank motivated this owner.
        _, unique = solve(matrix, np.zeros(matrix.shape[0]), mass, storage)
        if not unique[:n].all():
            msg = 'final joint coordinates lack unique original physical state authority'
            raise ValueError(msg)
        rate_state, rate_rate = columns[-2:]
        unavailable = np.asarray(abs(rate_state) @ (~unique[:n]).astype(int)).ravel()
        unavailable += np.asarray(abs(rate_rate) @ (~unique[n:]).astype(int)).ravel()
        return SparseInitialPair(state, physical.copy(), unavailable == 0)
    msg = 'original joint finite storage state/rate equations did not converge'
    raise ValueError(msg)
