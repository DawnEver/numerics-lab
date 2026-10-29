"""Joint stored-coordinate elimination keeps algebraically fixed quantities exact."""

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear import initial_storage_border as owner
from numerics_lab.linear.initial_storage_border import stored_initial_correction
from numerics_lab.linear.storage_border import free_pinned_columns


def _large_sparse_joint_equations(count=300):
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


def test_declared_sparse_coordinates_are_consumed_below_dense_border_ceiling(monkeypatch):
    count, jacobian, residual, mass, algebraic = _large_sparse_joint_equations(3)
    original = owner._sparse_joint_border
    consumed = []

    def actual_sparse_closure(*args, **kwargs):
        consumed.append(args[-1].copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, '_sparse_joint_border', actual_sparse_closure)
    indices = np.arange(count)
    correction, unique = _large_correction(jacobian, residual, mass, algebraic, indices)
    assert len(consumed) == 1
    np.testing.assert_array_equal(consumed[0], indices)
    np.testing.assert_array_equal(correction[:count], np.zeros(count))
    assert correction[count] == pytest.approx(7 / 5.5, abs=1e-14)
    assert unique[:count].all()


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


@pytest.mark.parametrize('storage', [2.0, 3.0, 7.0])
def test_sparse_retained_border_cannot_invent_a_unique_physical_state(storage):
    count, jacobian, residual, mass, algebraic = _large_sparse_joint_equations()
    jacobian = jacobian.tolil()
    mass = mass.tolil()
    mass[0, 0] = storage
    jacobian[count + 1, 0] = storage
    jacobian[count, count] = -1
    residual[:] = 0
    # Every current i is admissible: A=0, lambda=0, A'=i/storage,
    # lambda'=i. The selected sparse minors are nonsingular; the retained
    # original KVL is identical to the differentiated linkage constraint.
    with pytest.raises(ValueError, match='physical state authority'):
        _large_correction(jacobian.tocsr(), residual, mass.tocsr(), algebraic, np.arange(count))


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
    assert rate[1] == pytest.approx(free_rate, rel=1e-12, abs=0.0)
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


@pytest.mark.parametrize('voltage', [0.1, -0.1, 0.3, 1e-12])
def test_homogeneous_stored_field_keeps_exact_circuit_constraints_at_startup(voltage):
    # Actual three-field-coordinate conducting/MNA topology: source, node KCL,
    # series inductance and independent physical linkage. Zero full Q fixes the
    # field, series current and linkage; the source drives a 100 A/s rate only.
    n = 12
    jacobian = np.zeros((n, n))
    stiffness = 1.0 / (4e-7 * np.pi)
    jacobian[:3, :3] = stiffness * np.array([[1, 0, -0.5], [0, 1, -0.5], [-0.5, -0.5, 1]])
    jacobian[:3, 10] = -1 / 3
    for row, entries in {
        3: {7: 1, 8: 1},
        4: {4: 1, 5: -1, 8: -1},
        5: {4: -1, 5: 1, 10: 1},
        6: {9: -1, 10: -1},
        7: {3: -1},
        8: {3: -1, 4: 1},
        9: {6: 1},
        10: {5: -1, 6: 1, 10: 3.448e-6},
        11: {0: -1 / 300, 1: -1 / 300, 2: -1 / 300, 11: 1},
    }.items():
        for column, value in entries.items():
            jacobian[row, column] = value
    mass = np.zeros((n, n))
    mass[:3, :3] = (5 / 12) * (np.eye(3) + np.ones((3, 3)))
    mass[8, 8], mass[10, 11] = 0.001, 1.0
    state = np.zeros(n)
    state[3] = np.nextafter(voltage, 0.0)
    residual = jacobian @ state
    residual[7] += voltage
    jacobian, mass = sp.csr_matrix(jacobian), sp.csr_matrix(mass)
    correction, _unique = stored_initial_correction(
        jacobian,
        residual,
        mass,
        sp.eye(n, format='csr')[[3, 4, 5, 6, 7, 9, 11]],
        np.array([0, 1, 2, 8, 10]),
        np.array([0, 1, 2, 8, 11]),
        np.zeros(5),
        np.zeros(n),
        tolerance=1e-10,
        state=state,
        rate_tangent=jacobian,
    )
    np.testing.assert_array_equal(correction[[0, 1, 2, 8, 11]], np.zeros(5))
    np.testing.assert_array_equal(correction[n : n + 3], np.zeros(3))
    assert correction[n + 8] == pytest.approx(voltage / 0.001, rel=1e-12, abs=0.0)


def test_the_residuals_certified_rounding_reaches_the_sparse_elimination_and_its_validator(monkeypatch):
    # A solid body's net-current row depends on its conducting field rows, so it keeps the image
    # of their certified rounding (case_a dual-star PWM: 4.5e-8 against an 8.6e-9 tolerance bar).
    count, jacobian, residual, mass, algebraic = _large_sparse_joint_equations(3)
    n = mass.shape[0]
    border, validator = owner._sparse_joint_border, owner._validate_joint_correction
    seen = {}

    def forwarded_border(*args, **kwargs):
        seen['allowance'] = kwargs['residual_allowance']
        result = border(*args, **kwargs)
        seen['admitted'] = result[2]
        return result

    def forwarded_validator(*args, **kwargs):
        seen['validated'] = kwargs['admitted']
        return validator(*args, **kwargs)

    monkeypatch.setattr(owner, '_sparse_joint_border', forwarded_border)
    monkeypatch.setattr(owner, '_validate_joint_correction', forwarded_validator)
    bound = np.full(n, 1e-13)
    stored_initial_correction(
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
        residual_allowance=bound,
    )
    np.testing.assert_array_equal(seen['allowance'][:n], bound)
    assert not seen['allowance'][n:].any()
    assert seen['validated'] is seen['admitted']
    assert np.all(seen['admitted'][:n] >= bound)


def test_a_row_defect_inside_its_admitted_image_is_not_a_contradiction():
    # One row, one coordinate: x = 1 with a defect of 1e-12 left in the correction.
    jacobian = sp.csr_matrix([[1.0]])
    mass = sp.csr_matrix((1, 1))
    matrix = sp.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
    arguments = (
        np.array([1.0 + 2.0**-40, 0.0]),
        np.zeros((1, 0)),
        matrix,
        np.array([1.0, 0.0]),
        jacobian,
        np.array([-1.0]),
        mass,
        sp.csr_matrix([[1.0]]),
        np.zeros(1),
        1e-14,
        np.zeros(1),
        sp.csr_matrix([[1.0]]),
    )
    with pytest.raises(ValueError, match='contradicts'):
        owner._validate_joint_correction(*arguments)
    owner._validate_joint_correction(*arguments, admitted=np.array([1e-12, 0.0]))
