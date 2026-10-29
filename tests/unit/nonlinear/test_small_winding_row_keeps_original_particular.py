"""A computed homogeneous range must not overwrite an original numerical particular."""

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear.storage_border import storage_border_columns, storage_border_rows
from numerics_lab.nonlinear.finite_storage import finite_storage_actual_rate_initial_pair


@pytest.mark.parametrize('stiffness', [1e8, 1e10])
@pytest.mark.parametrize('voltage', [0.075, 12.0])
def test_tiny_unforced_winding_rows_keep_the_original_q_f_particular(stiffness, voltage):
    # Field, three node voltages, three source currents, three winding currents,
    # and three physical linkages. Sources 1/2 are genuinely zero; no artificial
    # resistance or source derivative is added to their winding rows.
    n = 13
    turns, resistance = np.array([1.0, -2.0, 0.5]), np.array([3.0, 4.0, 2.0])
    winding, linkage = np.arange(7, 10), np.arange(10, 13)
    operator, mass = np.zeros((n, n)), np.zeros((n, n))
    operator[0, 0], operator[0, winding] = stiffness, -turns
    for phase in range(3):
        node, source, current, flux = 1 + phase, 4 + phase, winding[phase], linkage[phase]
        operator[node, [source, current]] = 1.0
        operator[source, node] = 1.0
        operator[current, [node, current]] = [-1.0, resistance[phase]]
        operator[flux, [0, flux]] = [-turns[phase], 1.0]
        mass[current, flux] = 1.0
    operator, mass = sp.csr_matrix(operator), sp.csr_matrix(mass)
    source, partial = np.zeros(n), np.zeros(n)
    source[4], partial[4] = voltage, -0.75
    initial = np.zeros(n)
    initial[0] = voltage / (3 * stiffness) * (1 + 1e-12)
    initial[1], initial[4], initial[7] = voltage, -voltage / 3, voltage / 3
    initial[linkage] = turns * initial[0]
    border = np.arange(n)

    def residual(state, rate):
        return operator @ state + mass @ rate - source

    pair = finite_storage_actual_rate_initial_pair(
        initial,
        residual_at=residual,
        jacobian_at=lambda _state, _rate: operator,
        mass_at=lambda _state, _rate: mass,
        storage_at=lambda state: mass @ state,
        partial_time_at=lambda _state, _rate: partial,
        storage_rows_at=lambda _state, _rate: storage_border_rows(
            mass, bulk_indices=np.empty(0, dtype=int), border_rows=border
        ),
        stored_columns_at=lambda _state, _rate, rows: storage_border_columns(
            mass, bulk_indices=np.empty(0, dtype=int), stored_rows=rows.stored_rows
        ),
        rate_constraint_jacobian_at=lambda _state, _rate: operator,
        rate_constraint_state_columns_at=lambda _state, _rate, directions: sp.csr_matrix((n, directions.shape[1])),
        sparse_state_indices=np.array([0]),
    )
    np.testing.assert_array_equal(mass @ pair.state, mass @ initial)
    scale = abs(operator) @ abs(pair.state) + abs(mass) @ abs(pair.rate) + abs(residual(pair.state, np.zeros(n)))
    assert np.all(abs(residual(pair.state, pair.rate)) <= 1e-10 * scale)
