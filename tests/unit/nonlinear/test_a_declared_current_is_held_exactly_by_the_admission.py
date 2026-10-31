"""A declared winding current is a stated initial condition: the admission holds it bit-exactly.

The fixture is a field coupled to three windings through linkage storage, whose currents the
algebraic rows determine. Declared at their ACTUAL values, the currents come back bit-identical
(the joint correction anchors them) and every original row still meets its bar; declared at a
value the rows contradict, the admission refuses -- it never rounds onto the declaration.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear.storage_border import storage_border_columns, storage_border_rows
from numerics_lab.nonlinear.finite_storage import finite_storage_actual_rate_initial_pair

TURNS = np.array([1.0, -2.0, 0.5])
RESISTANCE = np.full(3, 1.1)
VOLTAGE = np.full(3, 150.0)
SIZE = 7  # field, three currents, three linkage coordinates
CURRENTS = np.arange(1, 4)


def _system():
    stiffness, mass = np.zeros((SIZE, SIZE)), np.zeros((SIZE, SIZE))
    stiffness[0, 0], stiffness[0, 1:4] = 2.0, -TURNS
    stiffness[1:4, 1:4] = np.diag(RESISTANCE)
    stiffness[4:, 0], stiffness[4:, 4:] = -TURNS, np.eye(3)
    mass[1:4, 4:] = np.eye(3)
    return sp.csr_matrix(stiffness), sp.csr_matrix(mass), np.r_[0.0, VOLTAGE, np.zeros(3)]


def _admit(initial, fixed):
    stiffness, mass, source = _system()
    bulk, border = np.empty(0, dtype=int), np.arange(SIZE)
    return finite_storage_actual_rate_initial_pair(
        initial,
        residual_at=lambda state, rate: stiffness @ state + mass @ rate - source,
        jacobian_at=lambda _state, _rate: stiffness,
        mass_at=lambda _state, _rate: mass,
        storage_at=lambda state: mass @ state,
        partial_time_at=lambda _state, _rate: np.zeros(SIZE),
        storage_rows_at=lambda _state, _rate: storage_border_rows(mass, bulk_indices=bulk, border_rows=border),
        stored_columns_at=lambda _state, _rate, rows: storage_border_columns(
            mass, bulk_indices=bulk, stored_rows=rows.stored_rows
        ),
        rate_constraint_jacobian_at=lambda _state, _rate: stiffness,
        rate_constraint_state_columns_at=lambda _state, _rate, directions: sp.csr_matrix((SIZE, directions.shape[1])),
        sparse_state_indices=np.array([0]),
        fixed_state_indices=fixed,
    )


def _actual_currents():
    field_rate = (TURNS @ (VOLTAGE / RESISTANCE)) / (TURNS @ (TURNS / RESISTANCE))
    return (VOLTAGE - TURNS * field_rate) / RESISTANCE


def test_declared_currents_come_back_bit_identical_and_every_row_holds():
    initial = np.zeros(SIZE)
    initial[CURRENTS] = _actual_currents()
    pair = _admit(initial, CURRENTS)
    assert pair.state[CURRENTS].tobytes() == initial[CURRENTS].tobytes()
    stiffness, mass, source = _system()
    np.testing.assert_allclose(stiffness @ pair.state + mass @ pair.rate - source, 0.0, rtol=0.0, atol=1e-10)


def test_a_declaration_the_rows_contradict_is_refused():
    with pytest.raises((ValueError, NotImplementedError)):
        _admit(np.zeros(SIZE), CURRENTS)


def test_the_joint_chart_admission_has_no_anchor_and_says_so():
    with pytest.raises(NotImplementedError, match='joint-chart'):
        finite_storage_actual_rate_initial_pair(
            np.zeros(SIZE), fixed_state_indices=CURRENTS, joint_coordinates_at=object()
        )


def test_a_rounding_off_declaration_is_held_where_the_free_admission_moves_it():
    initial = np.zeros(SIZE)
    initial[CURRENTS] = _actual_currents() * (1 + 1e-15)
    assert _admit(initial, None).state[CURRENTS].tobytes() != initial[CURRENTS].tobytes()
    assert _admit(initial, CURRENTS).state[CURRENTS].tobytes() == initial[CURRENTS].tobytes()
