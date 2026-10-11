"""Joint stored-coordinate elimination keeps algebraically fixed quantities exact."""

import numpy as np
import pytest
import scipy.sparse as sp

from motronics.gauss.linear.initial_storage_border import stored_initial_correction
from motronics.gauss.linear.storage_border import free_pinned_columns


def _large_sparse_joint_equations():
    count = 300
    n = count + 2
    jacobian = sp.lil_matrix((n, n))
    jacobian[:count, :count] = sp.eye(count)
    jacobian[0, count] = -1
    jacobian[count, count] = 5
    jacobian[count + 1, 0] = 1
    jacobian[count + 1, count + 1] = -1
    jacobian = jacobian.tocsr()
    mass = sp.lil_matrix((n, n))
    mass[0, 0] = 2
    mass[count, count + 1] = 1
    mass = mass.tocsr()
    algebraic = sp.eye(n, format='csr')[np.r_[np.arange(1, count), count + 1]]
    residual = np.zeros(n)
    residual[count] = -7
    return count, jacobian, residual, mass, algebraic


def _large_correction(jacobian, residual, mass, algebraic, indices):
    count = mass.shape[0] - 2
    n = mass.shape[0]
    return stored_initial_correction(
        jacobian,
        residual,
        mass,
        algebraic,
        np.array([0, count]),
        np.array([0, count + 1]),
        np.zeros(2),
        np.zeros(n),
        tolerance=1e-10,
        state=np.zeros(n),
        rate_tangent=jacobian,
        sparse_state_indices=indices,
    )


def test_declared_large_sparse_state_rate_block_is_not_a_dense_free_border():
    count, jacobian, residual, mass, algebraic = _large_sparse_joint_equations()
    n = mass.shape[0]
    correction, unique = stored_initial_correction(
        jacobian,
        residual,
        mass,
        algebraic,
        np.array([0, count]),
        np.array([0, count + 1]),
        np.zeros(2),
        np.zeros(n),
        tolerance=1e-10,
        state=np.zeros(n),
        rate_tangent=jacobian,
        sparse_state_indices=np.arange(count),
    )
    np.testing.assert_array_equal(correction[:count], np.zeros(count))
    assert correction[count] == pytest.approx(7 / 5.5, abs=1e-14)
    assert correction[n] == pytest.approx(correction[count] / 2, abs=1e-14)
    assert correction[n + count + 1] == pytest.approx(correction[n], abs=1e-14)
    assert not unique[count]
    assert unique[:count].all() and unique[count + 1]


@pytest.mark.parametrize('indices', [np.array([0, 0]), np.array([-1]), np.array([0.5])])
def test_sparse_elimination_refuses_invalid_coordinate_declarations(indices):
    _count, jacobian, residual, mass, algebraic = _large_sparse_joint_equations()
    with pytest.raises(ValueError, match='coordinate indices'):
        _large_correction(jacobian, residual, mass, algebraic, indices)


def test_sparse_elimination_cannot_discard_a_tiny_original_charge_contradiction():
    count, jacobian, residual, mass, algebraic = _large_sparse_joint_equations()
    residual[count + 1] = 1e-30
    with pytest.raises(ValueError, match='contradicts actual state/rate'):
        _large_correction(jacobian, residual, mass, algebraic, np.arange(count))


def test_sparse_elimination_refuses_an_unowned_mass_coupling_to_air_coordinates():
    count, jacobian, residual, mass, algebraic = _large_sparse_joint_equations()
    mass = mass.tolil()
    mass[0, 1] = 1
    with pytest.raises(ValueError, match='exact stored/zero-mass'):
        _large_correction(jacobian, residual, mass.tocsr(), algebraic, np.arange(count))


def test_sparse_joint_elimination_never_densifies_the_large_state_block(monkeypatch):
    count, jacobian, residual, mass, algebraic = _large_sparse_joint_equations()
    original = sp.csr_matrix.toarray

    def bounded_dense(matrix, *args, **kwargs):
        # Only actual small-border responses or the small Schur block may be dense.
        assert min(matrix.shape) <= 4
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(sp.csr_matrix, 'toarray', bounded_dense)
    correction, _unique = _large_correction(jacobian, residual, mass, algebraic, np.arange(count))
    assert correction[count] == pytest.approx(7 / 5.5, abs=1e-14)


def test_sparse_joint_elimination_refuses_a_singular_actual_state_minor():
    count, jacobian, residual, mass, algebraic = _large_sparse_joint_equations()
    jacobian = jacobian.tolil()
    jacobian[1, 1] = 0
    with pytest.raises(ValueError, match='sparse'):
        _large_correction(jacobian.tocsr(), residual, mass, algebraic, np.arange(count))


def test_large_algebraic_state_block_does_not_invent_a_nonempty_storage_minor():
    count, jacobian, residual, mass, _algebraic = _large_sparse_joint_equations()
    n = mass.shape[0]
    mass = mass.tolil()
    mass[0, 0] = 0
    algebraic = sp.eye(n, format='csr')[np.r_[np.arange(count), count + 1]]
    correction, unique = stored_initial_correction(
        jacobian,
        residual,
        mass.tocsr(),
        algebraic,
        np.array([count]),
        np.array([count + 1]),
        np.zeros(1),
        np.zeros(n),
        tolerance=1e-10,
        state=np.zeros(n),
        rate_tangent=jacobian,
        sparse_state_indices=np.arange(count),
    )
    np.testing.assert_array_equal(correction[:n], np.zeros(n))
    assert correction[n] == pytest.approx(7.0, abs=1e-14)
    assert correction[n + count] == pytest.approx(7.0, abs=1e-14)
    assert unique.all()


def test_fully_determined_large_sparse_block_needs_no_retained_border():
    count = 300
    jacobian = sp.eye(count, format='csr')
    mass = sp.diags(np.r_[2.0, np.zeros(count - 1)], format='csr')
    algebraic = sp.eye(count, format='csr')[1:]
    residual = np.r_[7.0, np.zeros(count - 1)]
    correction, unique = stored_initial_correction(
        jacobian,
        residual,
        mass,
        algebraic,
        np.array([0]),
        np.array([0]),
        np.zeros(1),
        np.zeros(count),
        tolerance=1e-10,
        state=np.zeros(count),
        rate_tangent=jacobian,
        sparse_state_indices=np.arange(count),
    )
    np.testing.assert_array_equal(correction[:count], np.zeros(count))
    assert correction[count] == pytest.approx(-3.5, abs=1e-14)
    assert unique.all()


def test_an_algebraically_pinned_stored_rate_survives_cancellation_against_a_large_free_rate():
    # Stored row: m (r0 - r1) + 2.5e-10 x0 + x2 + R0 = 0, with m = 1 nF-like mass.
    # Differentiated algebraic row pins r0 = 0, so the free rate r1 = (x2 + R0) / m
    # is ~ -2.5e7 and r0 is recovered from it by cancellation of that size.
    # Measured on the six-flag production kernel (row 203: r96 = 0 against r98 = -5e7).
    mass = 1e-9
    stored_residual = -0.07499159179807384
    jacobian = sp.csr_matrix([[2.5e-10, 0, 1], [0, 0, 1], [-1, 0, 0]])
    correction, unique = stored_initial_correction(
        jacobian,
        np.array([stored_residual, -0.05, 0.0]),
        sp.csr_matrix([[mass, -mass, 0], [0, 0, 0], [0, 0, 0]]),
        sp.csr_matrix([[0, 1.0, 0], [0, 0, 1.0]]),
        np.array([0]),
        np.array([0]),
        np.zeros(1),
        np.zeros(3),
        tolerance=1e-10,
        state=np.zeros(3),
        rate_tangent=jacobian,  # Constant mass: the differentiated residual has this exact tangent.
    )
    state, rate = correction[:3], correction[3:]
    assert unique.all()
    assert state == pytest.approx([0.0, 0.0, 0.05], abs=1e-15)
    free_rate = (0.05 + stored_residual) / mass
    assert rate[1] == pytest.approx(free_rate, rel=1e-12)
    # The pinned rate stays on the border: exactly zero, not ulps of the free rate.
    assert rate[0] == 0
    assert rate[2] == 0


def test_an_algebraic_row_on_fixed_stored_states_is_judged_at_the_state_it_was_evaluated():
    # Linkage row x1 = 0.1 x0 touches only stored coordinates, whose storage is
    # fixed, so its evaluation roundoff cannot be repaired by any correction.
    # Measured on the plain conducting-interface kernel: residual 1.32e-23.
    state = np.array([0.3608613664297655, 0.03608613664297655, 0.0])
    residual = np.array([0.0, 0.0, 1.3234889800848443e-23])
    jacobian = sp.csr_matrix([[0, 0, 1.0], [0, 0, 0], [-0.1, 1.0, 0]])
    mass = sp.csr_matrix([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 0]])
    correction, _ = stored_initial_correction(
        jacobian,
        residual,
        mass,
        sp.csr_matrix([[0, 0, 1.0]]),
        np.array([0, 1]),
        np.array([0, 1]),
        np.zeros(2),
        np.zeros(3),
        tolerance=1e-10,
        state=state,
        rate_tangent=jacobian,
    )
    assert np.array_equal(correction[:2], [0, 0])
    # A residual of physical size on the same row is still a contradiction.
    with pytest.raises(ValueError, match='contradicts'):
        stored_initial_correction(
            jacobian,
            np.array([0.0, 0.0, 1e-3]),
            mass,
            sp.csr_matrix([[0, 0, 1.0]]),
            np.array([0, 1]),
            np.array([0, 1]),
            np.zeros(2),
            np.zeros(3),
            tolerance=1e-10,
            state=state,
            rate_tangent=jacobian,
        )


def test_a_pinned_stored_column_is_exchanged_for_its_storage_partner():
    # Shared by the rate and acceleration eliminations: both differentiate
    # the same algebraic rows, so both must keep the pinned coordinate free.
    mass = sp.csr_matrix([[1e-9, -1e-9, 0], [0, 0, 0], [0, 0, 0]])
    pinned = sp.csr_matrix([[0, 0, 1.0], [-1.0, 0, 0]])
    assert free_pinned_columns(np.array([0]), mass[[0]], pinned).tolist() == [1]
    assert free_pinned_columns(np.array([1]), mass[[0]], pinned).tolist() == [1]
