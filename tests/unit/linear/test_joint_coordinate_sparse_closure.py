"""Authentic joint residual-rate and physical stored-state minors are separate."""

from fractions import Fraction

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear import initial_storage_border
from numerics_lab.linear.arithmetic import matrix_vector_roundoff
from numerics_lab.linear.initial_sparse import sparse_initial_pair
from numerics_lab.linear.initial_storage_border import joint_coordinate_sparse_correction


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


@pytest.mark.parametrize('allowance', [None, 0.5e-23, 2e-23])
def test_only_certified_storage_increment_roundoff_can_meet_an_exact_anchor(allowance):
    kwargs = {} if allowance is None else {'stored_allowance': np.array([allowance])}

    def admit():
        return sparse_initial_pair(
            sp.csr_matrix([[1.0, 1.0], [1.0, 0.0]]),
            np.zeros(2),
            sp.diags([1.0, 0.0], format='csr'),
            np.zeros(2),
            algebraic_rows=sp.csr_matrix([[0.0, 1.0]]),
            stored_rows=np.array([0]),
            partial_time=np.zeros(2),
            stored_increment=np.array([1e-23]),
            stored_columns=np.array([0]),
            sparse_state_indices=np.array([0]),
            **kwargs,
        )

    if allowance is None or allowance < 1e-23:
        with pytest.raises(ValueError, match='homogeneous anchors'):
            admit()
    else:
        pair = admit()
        np.testing.assert_array_equal(pair.state, np.zeros(2))
        np.testing.assert_array_equal(pair.rate, np.zeros(2))


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
    # Original Q fixes x=1. The computed homogeneous range is not authority
    # for the last bit of y: the original source row supplies 5*y=fl(5/3).
    assert answer[0] == 1.0
    original_bound = matrix_vector_roundoff(matrix, answer, -rhs)
    assert original_bound.known.all()
    for row, target, allowance in zip(matrix.toarray(), rhs, original_bound.values, strict=True):
        defect = sum(Fraction(float(a)) * Fraction(float(x)) for a, x in zip(row, answer, strict=True))
        assert abs(defect - Fraction(float(target))) <= Fraction(float(allowance))
    # Independently bound the discrepancy to the range's actual binary64
    # product/subtraction, rather than granting an arbitrary relative tolerance.
    parameter = np.array([1.0])
    range_bound = matrix_vector_roundoff(direction, parameter, -answer[:2])
    assert range_bound.known.all()
    for row, value, allowance in zip(direction.toarray(), answer[:2], range_bound.values, strict=True):
        reference = sum(Fraction(float(a)) * Fraction(float(x)) for a, x in zip(row, parameter, strict=True))
        assert abs(Fraction(float(value)) - reference) <= Fraction(float(allowance))
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


def test_the_computed_schur_border_carries_a_bound_on_its_own_formation_error(monkeypatch):
    # The border handed to the eliminator is fl(A_rr - L B^-1 C); its entries' distance
    # from the exact rational Schur complement must sit inside the bound handed with it,
    # or a cancellation remainder (a solid body's dependent net-current row) reads as a pivot.

    captured = {}
    real = initial_storage_border.border_affine_space

    def spy(matrix, rhs, **kwargs):
        captured.update(matrix=np.array(matrix), errors=kwargs.get('entry_errors'))
        return real(matrix, rhs, **kwargs)

    monkeypatch.setattr(initial_storage_border, 'border_affine_space', spy)
    matrix = sp.csr_matrix([[0.3, 0.7, 0.1, 0.0], [0.7, 1.9, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0], [0.0, 0.0, 0.7, 1.9]])
    rhs = matrix @ np.array([1.0, 2.0, 3.0, 4.0])
    joint_coordinate_sparse_correction(
        matrix,
        rhs,
        mass=sp.diags([0.1, 0.0], format='csr'),
        algebraic_rows=sp.csr_matrix([[0.0, 1.0]]),
        stored_rows=np.array([0]),
        indices=np.array([0]),
    )
    dense = [[Fraction(float(value)) for value in row] for row in matrix.toarray()]
    selected_rows, selected_columns, remaining_rows, remaining_columns = (0, 2), (0, 2), (1, 3), (1, 3)
    b = [[dense[r][c] for c in selected_columns] for r in selected_rows]
    det = b[0][0] * b[1][1] - b[0][1] * b[1][0]
    inverse = [[b[1][1] / det, -b[0][1] / det], [-b[1][0] / det, b[0][0] / det]]
    for i, r in enumerate(remaining_rows):
        for j, c in enumerate(remaining_columns):
            image = sum(
                dense[r][selected_columns[k]] * inverse[k][m] * dense[selected_rows[m]][c]
                for k in range(2)
                for m in range(2)
            )
            exact = dense[r][c] - image
            computed = Fraction(float(captured['matrix'][i, j]))
            assert abs(computed - exact) <= Fraction(float(captured['errors'][i, j]))
    assert captured['errors'] is not None and np.all(captured['errors'] >= 0)
