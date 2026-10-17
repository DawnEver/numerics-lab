"""Frozen rates inherit only already-validated original algebraic state bars."""

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear.initial_storage_border import _validate_joint_correction


def _validate(residual, partial=None, *, stored=False):
    jacobian = sp.csr_matrix([[1.0, 1.0], [0.0, 0.0]])
    mass = sp.csr_matrix((2, 2))
    algebraic = sp.eye(2, format='csr')
    matrix = sp.bmat([[jacobian, mass], [mass, jacobian]], format='csr')
    rhs = np.r_[-residual, -np.zeros(2) if partial is None else -partial]
    if stored:
        matrix = sp.vstack((matrix, sp.csr_matrix([[1, 0, 0, 0]])), format='csr')
        rhs = np.r_[rhs, 0.0]
    return _validate_joint_correction(
        np.array([1.0, -1.0, 0.0, 0.0]),
        np.empty((2, 0)),
        matrix,
        rhs,
        jacobian,
        residual,
        mass,
        algebraic,
        np.zeros(2) if partial is None else partial,
        1e-10,
        np.zeros(2),
        jacobian,
    )


def test_validated_original_state_bar_survives_the_frozen_zero_mass_rate_rows():
    correction, _unique = _validate(np.array([1e-12, 0.0]))
    np.testing.assert_array_equal(correction[:2], [1.0, -1.0])


def test_invalid_original_zero_mass_state_is_not_admitted_by_rate_transport():
    with pytest.raises(ValueError, match='contradict'):
        _validate(np.array([1e-6, 0.0]))


def test_independent_tiny_differentiated_constraint_gets_no_state_allowance():
    with pytest.raises(ValueError, match='contradict'):
        _validate(np.array([1e-12, 0.0]), np.array([0.0, 1e-30]))


def test_finite_storage_witness_remains_original_after_state_allowance_transport():
    with pytest.raises(ValueError, match='contradict'):
        _validate(np.array([1e-12, 0.0]), stored=True)
