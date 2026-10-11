"""Explicit joint-coordinate finite-storage initial admission."""

import numpy as np
import scipy.sparse as sp

from motronics.contracts.runtime.initial_coordinates import InitialJointEvaluation
from motronics.gauss.linear.initial_rate import sparse_initial_rate
from motronics.gauss.linear.initial_sparse import SparseInitialPair, projected_rate_consistent


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
):
    """One sparse chart for search and the final homogeneous physical proof.

    In particular, the owner forms joint columns before cancellation-sensitive
    contractions. Neither a numerical nullspace nor ``J + M T`` supplies them.
    Full stored-state rows remain witnesses even when they are dependent.
    """
    n = len(state)
    independent = np.zeros(n)

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
        return sample.physical_rate, matrices

    def equations(value, velocity):
        physical, columns = evaluate(value, velocity)
        state_columns, rate_columns, derivative_state, derivative_rate, _, _ = columns
        mass, jacobian = mass_at(value, physical), jacobian_at(value, physical)
        storage = storage_rows_at(value, physical)
        evaluation = None if residual_evaluation_at is None else residual_evaluation_at(value, physical)
        residual = residual_at(value, physical) if evaluation is None else evaluation.residual
        partial = partial_time_at(value, physical)
        derivative = storage.algebraic_rows @ (jacobian @ physical + partial)
        matrix = sp.bmat(
            [
                [state_columns, rate_columns],
                [mass, sp.csr_matrix((n, n))],
                [storage.algebraic_rows @ derivative_state, storage.algebraic_rows @ derivative_rate],
            ],
            format='csr',
        )
        defect = np.concatenate((residual, charge_at(value) - target, derivative))
        return physical, columns, mass, jacobian, storage, matrix, defect, partial

    for _iteration in range(max_newton):
        physical, columns, mass, jacobian, storage, matrix, defect, partial = equations(state, independent)
        # No construction allowance is transferred to Q or differentiated rows.
        change, unique = sparse_initial_rate(matrix, -defect, tolerance=tolerance)
        if not unique[:n].all():
            msg = 'joint initial coordinates lack unique original physical state authority'
            raise ValueError(msg)
        state = state + change[:n]
        independent = independent + change[n:]
        physical, columns, mass, jacobian, storage, matrix, defect, partial = equations(state, independent)
        residual, charge = defect[:n], defect[n : 2 * n] + target
        # The source reference belongs to this exact evaluated residual. A
        # second callback at zero rate can be a different future source value.
        residual_scale = abs(jacobian) @ abs(state) + abs(mass) @ abs(physical) + abs(residual - mass @ physical)
        charge_scale = abs(charge) + abs(target) + abs(mass) @ abs(state)
        if (
            np.any(abs(residual) > tolerance * residual_scale)
            or np.any(abs(charge - target) > tolerance * charge_scale)
            or not projected_rate_consistent(
                jacobian,
                physical,
                partial,
                algebraic_rows=storage.algebraic_rows,
                tolerance=tolerance,
            )
        ):
            continue
        # Re-evaluate and factor the same actual chart, never the rounded raw
        # state/rate matrix whose spurious binary rank motivated this owner.
        _, unique = sparse_initial_rate(matrix, np.zeros(matrix.shape[0]), tolerance=tolerance)
        if not unique[:n].all():
            msg = 'final joint coordinates lack unique original physical state authority'
            raise ValueError(msg)
        rate_state, rate_rate = columns[-2:]
        unavailable = np.asarray(abs(rate_state) @ (~unique[:n]).astype(int)).ravel()
        unavailable += np.asarray(abs(rate_rate) @ (~unique[n:]).astype(int)).ravel()
        return SparseInitialPair(state, physical.copy(), unavailable == 0)
    msg = 'original joint finite storage state/rate equations did not converge'
    raise ValueError(msg)
