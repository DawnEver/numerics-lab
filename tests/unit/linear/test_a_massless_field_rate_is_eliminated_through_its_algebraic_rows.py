"""A field with no conductive bulk is algebraic: its accelerations are fixed by its differentiated rows.

THE MEASURED CASE (motronics, 2026-11-07): a magnetostatic three-phase transformer winding in an
ISOLATED star. The star point voltage is algebraic and fixed by ``d lambda / dt``, so its RATE needs
the field's second jet. Every field acceleration is a free (non-stored) acceleration, and the rate
completion put all of them -- 21160 -- into its dense border, past the bounded 512-column contract:
"stored acceleration elimination requires proven independent coordinates and a bounded actual border".

They are not free: the twice-differentiated algebraic field rows ``K a'' = B i''`` determine them
sparsely. Only the star point's rate is a genuine border unknown.

The model below is that structure at a size past the border bound. Unknowns
``x = (a [N], lambda [3], i [3], v_N)``; rows

    K a - B i = 0                         (massless field, N rows)
    lambda_k - b_k^T a = 0                (linkage, 3 rows)
    lambda_k' + R i_k + v_N - v_k(t) = 0  (winding, 3 stored rows)
    i_1 + i_2 + i_3 = 0                   (isolated star KCL)

At ``t = 0`` with zero currents, ``L i' = v - v_N``, ``sum i' = 0`` and once more differentiated
``L i'' = v' - v_N' - R i'``, ``sum i'' = 0``; ``L = B^T K^{-1} B`` is symmetric, so
``v_N' = (1^T L^{-1} (v' - R i')) / (1^T L^{-1} 1)``.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear.initial_sparse import sparse_complete_initial_rate, sparse_initial_pair
from numerics_lab.linear.storage_border import storage_border_columns, storage_border_rows

FIELD = 700  # past the 512-column border contract
RESISTANCE = 0.3
OMEGA = 2 * np.pi * 50
AMPLITUDE = 100.0
PHASES = np.array([0.4, 0.4 - 2 * np.pi / 3, 0.4 + 2 * np.pi / 3])
# An unbalanced (zero-sequence) line set: an isolated star point then moves.
OFFSETS = np.array([5.0, -1.0, 2.0])


def _transformer():
    n = FIELD + 7
    stiffness = sp.diags([-np.ones(FIELD - 1), 2.5 * np.ones(FIELD), -np.ones(FIELD - 1)], [-1, 0, 1], format='csr')
    coupling = np.zeros((FIELD, 3))
    for phase in range(3):
        start = phase * FIELD // 3
        coupling[start : start + FIELD // 4, phase] = 1.0 + 0.1 * phase
    lam, cur, star = FIELD + np.arange(3), FIELD + 3 + np.arange(3), FIELD + 6
    jacobian = sp.lil_matrix((n, n))
    jacobian[:FIELD, :FIELD] = stiffness
    jacobian[:FIELD, cur[0] : cur[-1] + 1] = -coupling
    for phase in range(3):
        jacobian[lam[phase], lam[phase]] = 1.0
        jacobian[lam[phase], :FIELD] = -coupling[:, phase][None, :]
        jacobian[FIELD + 3 + phase, cur[phase]] = RESISTANCE
        jacobian[FIELD + 3 + phase, star] = 1.0
        jacobian[star, cur[phase]] = 1.0
    # Row FIELD+3+k is winding k: mass on lambda_k.
    mass = sp.csr_matrix((np.ones(3), (FIELD + 3 + np.arange(3), lam)), shape=(n, n))
    line = AMPLITUDE * np.sin(PHASES) + OFFSETS
    line_rate = AMPLITUDE * OMEGA * np.cos(PHASES)
    residual = np.zeros(n)
    residual[FIELD + 3 + np.arange(3)] = -line
    partial = np.zeros(n)
    partial[FIELD + 3 + np.arange(3)] = -line_rate
    inductance = coupling.T @ np.linalg.solve(stiffness.toarray(), coupling)
    return sp.csr_matrix(jacobian), mass, residual, partial, inductance, line, line_rate, star, cur


def _expected_star_rate(inductance, line, line_rate):
    ones = np.ones(3)
    inverse = np.linalg.inv(inductance)
    star = ones @ inverse @ line / (ones @ inverse @ ones)
    current_rate = inverse @ (line - star)
    return ones @ inverse @ (line_rate - RESISTANCE * current_rate) / (ones @ inverse @ ones), star


def test_the_star_rate_is_completed_without_a_field_sized_border() -> None:
    jacobian, mass, residual, partial, inductance, line, line_rate, star, _ = _transformer()
    n = jacobian.shape[0]
    bulk = np.array([], dtype=int)
    storage = storage_border_rows(mass, bulk_indices=bulk, border_rows=np.arange(n))
    pair = sparse_initial_pair(
        jacobian,
        residual,
        mass,
        np.zeros(n),
        algebraic_rows=storage.algebraic_rows,
        stored_rows=storage.stored_rows,
        partial_time=partial,
    )
    assert not pair.rate_is_unique[star]  # the planted gap: the first pair cannot know v_N'
    completed = sparse_complete_initial_rate(
        jacobian,
        residual + jacobian @ pair.state,
        mass,
        pair,
        algebraic_rows=storage.algebraic_rows,
        partial_time=partial,
        acceleration_remainder=np.zeros(n),
        stored_rows=storage.stored_rows,
        stored_columns=storage_border_columns(mass, bulk_indices=bulk, stored_rows=storage.stored_rows),
    )
    expected_rate, expected_star = _expected_star_rate(inductance, line, line_rate)
    assert completed.state[star] == pytest.approx(expected_star, rel=1e-9, abs=1e-9)  # volts
    assert completed.rate_is_unique[star]
    assert completed.rate[star] == pytest.approx(expected_rate, rel=1e-8, abs=1e-6)  # volts per second
