"""Joint stored-coordinate elimination keeps algebraically fixed quantities exact."""

import numpy as np
import pytest
import scipy.sparse as sp

from motronics.gauss.linear.initial_storage_border import stored_initial_correction
from motronics.gauss.linear.storage_border import free_pinned_columns


def test_an_algebraically_pinned_stored_rate_survives_cancellation_against_a_large_free_rate():
    # Stored row: m (r0 - r1) + 2.5e-10 x0 + x2 + R0 = 0, with m = 1 nF-like mass.
    # Differentiated algebraic row pins r0 = 0, so the free rate r1 = (x2 + R0) / m
    # is ~ -2.5e7 and r0 is recovered from it by cancellation of that size.
    # Measured on the six-flag production kernel (row 203: r96 = 0 against r98 = -5e7).
    mass = 1e-9
    stored_residual = -0.07499159179807384
    correction, unique = stored_initial_correction(
        sp.csr_matrix([[2.5e-10, 0, 1], [0, 0, 1], [-1, 0, 0]]),
        np.array([stored_residual, -0.05, 0.0]),
        sp.csr_matrix([[mass, -mass, 0], [0, 0, 0], [0, 0, 0]]),
        sp.csr_matrix([[0, 1.0, 0], [0, 0, 1.0]]),
        np.array([0]),
        np.array([0]),
        np.zeros(1),
        np.zeros(3),
        tolerance=1e-10,
        state=np.zeros(3),
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
        )


def test_a_pinned_stored_column_is_exchanged_for_its_storage_partner():
    # Shared by the rate and acceleration eliminations: both differentiate
    # the same algebraic rows, so both must keep the pinned coordinate free.
    mass = sp.csr_matrix([[1e-9, -1e-9, 0], [0, 0, 0], [0, 0, 0]])
    pinned = sp.csr_matrix([[0, 0, 1.0], [-1.0, 0, 0]])
    assert free_pinned_columns(np.array([0]), mass[[0]], pinned).tolist() == [1]
    assert free_pinned_columns(np.array([1]), mass[[0]], pinned).tolist() == [1]
