"""Authentic joint residual-rate and physical stored-state minors are separate."""

import numpy as np
import pytest
import scipy.sparse as sp

from motronics.gauss.linear.initial_storage_border import joint_coordinate_sparse_correction


def _original_joint_rows():
    # F-rate coefficient10 and DQ coefficient2 are supplied independent owners;
    # differentiated G also has an explicitly owned state column7.
    matrix = sp.csr_matrix([[4.0, 0.0, 10.0, 0.0], [0.0, 5.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [7.0, 0.0, 0.0, 5.0]])
    rhs = np.array([34.0, 10.0, 2.0, 27.0])
    mass = sp.diags([2.0, 0.0], format='csr')
    return matrix, rhs, mass


def test_actual_distinct_sparse_bulk_operators_and_differentiated_state_are_retained():
    matrix, rhs, mass = _original_joint_rows()
    answer, unique = joint_coordinate_sparse_correction(
        matrix,
        rhs,
        mass=mass,
        algebraic_rows=sp.csr_matrix([[0.0, 1.0]]),
        stored_rows=np.array([0]),
        indices=np.arange(2),
    )
    np.testing.assert_array_equal(answer, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(matrix @ answer, rhs)
    assert unique.all()


def test_sparse_joint_closure_refuses_unknown_rate_support_on_declared_algebraic_coordinates():
    matrix, rhs, mass = _original_joint_rows()
    matrix = matrix.tolil()
    matrix[1, 2] = 1
    with pytest.raises(ValueError, match='stored/zero-rate coordinate support'):
        joint_coordinate_sparse_correction(
            matrix.tocsr(),
            rhs,
            mass=mass,
            algebraic_rows=sp.csr_matrix([[0.0, 1.0]]),
            stored_rows=np.array([0]),
            indices=np.arange(2),
        )


def test_original_sparse_homogeneous_anchors_cannot_hide_tiny_driven_conflict():
    matrix = sp.csr_matrix([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match='exact homogeneous anchors'):
        joint_coordinate_sparse_correction(
            matrix,
            np.array([1e-30, 0.0, 0.0, 0.0]),
            mass=sp.diags([1.0, 0.0], format='csr'),
            algebraic_rows=sp.csr_matrix([[0.0, 1.0]]),
            stored_rows=np.array([0]),
            indices=np.arange(2),
        )


def test_only_owned_top_row_roundoff_can_transport_an_already_fixed_particular():
    matrix = sp.csr_matrix([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    rhs = np.array([1e-23, 0.0, 0.0, 0.0, 0.0])
    answer, unique = joint_coordinate_sparse_correction(
        matrix,
        rhs,
        mass=sp.diags([1.0, 0.0], format='csr'),
        algebraic_rows=sp.csr_matrix([[0.0, 1.0]]),
        stored_rows=np.array([0]),
        indices=np.arange(2),
        residual_allowance=np.array([2e-23, 0.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(answer, np.zeros(4))
    assert unique.all()


def test_owned_correlated_state_range_requires_no_binary_representable_annihilator():
    matrix, rhs, mass = _original_joint_rows()
    rhs[1] = 5.0 / 3.0
    direction = sp.csr_matrix([[1.0], [1.0 / 3.0]])
    answer, unique = joint_coordinate_sparse_correction(
        matrix,
        rhs,
        mass=mass,
        algebraic_rows=sp.csr_matrix([[0.0, 1.0]]),
        stored_rows=np.array([0]),
        indices=np.arange(2),
        state_directions=direction,
    )
    np.testing.assert_array_equal(answer[:2], direction.toarray().ravel())
    np.testing.assert_allclose(matrix @ answer, rhs, rtol=1e-14, atol=0)
    assert unique.all()


def test_large_owned_fixed_range_retains_sparse_bulk_and_only_densifies_actual_border(monkeypatch):
    n = 1000
    mass = sp.diags(np.r_[np.ones(n - 2), 0.0, 0.0], format='csr')
    algebraic = sp.eye(n, format='csr')[-2:]
    stored = np.arange(n - 2)
    matrix = sp.bmat(
        [
            [sp.eye(n, format='csr'), mass],
            [mass[stored], sp.csr_matrix((n - 2, n))],
            [sp.csr_matrix((2, n)), algebraic],
        ],
        format='csr',
    )
    rhs = np.r_[np.zeros(n - 2), 1.0, 2.0, np.zeros(n)]
    directions = sp.csr_matrix(([1.0, 2.0], ([n - 2, n - 1], [0, 0])), shape=(n, 1))
    original = sp.csr_matrix.toarray

    def bounded(matrix, *args, **kwargs):
        assert max(matrix.shape) <= 512, 'only actual supported border data may densify'
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(sp.csr_matrix, 'toarray', bounded)
    answer, unique = joint_coordinate_sparse_correction(
        matrix,
        rhs,
        mass=mass,
        algebraic_rows=algebraic,
        stored_rows=stored,
        indices=stored,
        state_directions=directions,
    )
    np.testing.assert_array_equal(answer[:n], np.r_[np.zeros(n - 2), 1.0, 2.0])
    assert unique.all()


@pytest.mark.parametrize('forcing', [1.0, 1e-30])
def test_owned_state_range_cannot_discard_a_driven_fixed_coordinate(forcing):
    matrix, rhs, mass = _original_joint_rows()
    rhs[1] = forcing
    with pytest.raises(ValueError, match='joint stored state contradicts'):
        joint_coordinate_sparse_correction(
            matrix,
            rhs,
            mass=mass,
            algebraic_rows=sp.csr_matrix([[0.0, 1.0]]),
            stored_rows=np.array([0]),
            indices=np.arange(2),
            state_directions=sp.csr_matrix([[1.0], [0.0]]),
        )


@pytest.mark.parametrize('directions', [[[True], [0.0]], [[1.0], [False]], [[1j], [0.0]], [[np.nan], [0.0]]])
def test_owned_state_range_refuses_nonphysical_numeric_authority(directions):
    matrix, rhs, mass = _original_joint_rows()
    with pytest.raises(ValueError, match='finite real bounded actual direction columns'):
        joint_coordinate_sparse_correction(
            matrix,
            rhs,
            mass=mass,
            algebraic_rows=sp.csr_matrix([[0.0, 1.0]]),
            stored_rows=np.array([0]),
            indices=np.arange(2),
            state_directions=directions,
        )
