"""Actual finite right constraints use shared Newton globalization, not bare updates."""

import numpy as np
import scipy.sparse as sp

from numerics_lab.linear.storage_border import storage_border_rows
from numerics_lab.nonlinear.finite_storage import finite_storage_right_state


def test_saturating_algebraic_root_preserves_storage_while_shared_line_search_prevents_overshoot():
    mass = sp.csr_matrix([[0.0, 0.0], [0.0, 1.0]])
    rows = storage_border_rows(mass, bulk_indices=np.array([], dtype=int), border_rows=np.array([0, 1]))
    initial = np.array([10.0, 2.0])
    state = finite_storage_right_state(
        initial,
        residual_at=lambda value: np.array([np.arctan(value[0]) - np.pi / 4, 0.0]),
        jacobian_at=lambda value: sp.csr_matrix([[1 / (1 + value[0] ** 2), 0.0], [0.0, 0.0]]),
        mass_at=lambda _value: mass,
        storage_at=lambda value: mass @ value,
        storage_rows=rows,
        max_newton=30,
    )
    np.testing.assert_allclose(state, [1.0, 2.0], atol=1e-9, rtol=0)
    np.testing.assert_allclose(mass @ state, mass @ initial, atol=1e-14)
