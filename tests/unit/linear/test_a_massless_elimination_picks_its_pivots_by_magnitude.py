"""The algebraic-row elimination of massless accelerations picks its pivots by MAGNITUDE, not by storage order.

THE MEASURED CASE (motronics, 2026-11-08): the three-phase transformer chopper, an isolated star with an
OPEN line node. 7531 massless accelerations were eliminated through a STRUCTURAL maximum matching of the
twice-differentiated algebraic rows. Which maximum matching came back depended on the CSR index order of
the caller's matrix: with the order the march handed in, the open node's acceleration column was paired
with a negligible coupling entry instead of its own single-entry row ``a_open'' = 0``; ``splu`` factored the
pairing, the elimination base reached 4e21, and the final guard (correctly) refused the answer -- "declared
curvature contradicts actual stored-acceleration and differentiated equations". The same matrix with
sorted indices passed, and the method sorted the CALLER's matrix in place, so a retry passed: the result
was a function of storage order.

The model is the massless-transformer of the neighbouring test plus that open node: a coordinate ``p`` with
its own algebraic row ``p = 0`` that also enters every field row with a tiny coefficient. Any structural
maximum matching may pair ``p`` with a field row; only a magnitude-aware one is sound.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear import initial_sparse
from numerics_lab.linear.initial_sparse import sparse_complete_initial_rate, sparse_initial_pair
from numerics_lab.linear.storage_border import storage_border_columns, storage_border_rows

FIELD = 700  # past the 512-column border contract
RESISTANCE = 0.3
TINY = 1e-12  # the open node's negligible coupling into each field row
LINE = np.array([10.0, -4.0, 3.0])
LINE_RATE = np.array([2.0e3, -1.5e3, 0.7e3])


def _transformer_with_an_open_node():
    n = FIELD + 8
    stiffness = sp.diags([-np.ones(FIELD - 1), 2.5 * np.ones(FIELD), -np.ones(FIELD - 1)], [-1, 0, 1], format='csr')
    coupling = np.zeros((FIELD, 3))
    for phase in range(3):
        start = phase * FIELD // 3
        coupling[start : start + FIELD // 4, phase] = 1.0 + 0.1 * phase
    lam, cur, star, open_node = FIELD + np.arange(3), FIELD + 3 + np.arange(3), FIELD + 6, FIELD + 7
    jacobian = sp.lil_matrix((n, n))
    jacobian[:FIELD, :FIELD] = stiffness
    jacobian[:FIELD, cur[0] : cur[-1] + 1] = -coupling
    jacobian[:FIELD, open_node] = TINY
    for phase in range(3):
        jacobian[lam[phase], lam[phase]] = 1.0
        jacobian[lam[phase], :FIELD] = -coupling[:, phase][None, :]
        jacobian[FIELD + 3 + phase, cur[phase]] = RESISTANCE
        jacobian[FIELD + 3 + phase, star] = 1.0
        jacobian[star, cur[phase]] = 1.0
    jacobian[open_node, open_node] = 1.0
    mass = sp.csr_matrix((np.ones(3), (FIELD + 3 + np.arange(3), lam)), shape=(n, n))
    residual = np.zeros(n)
    residual[FIELD + 3 + np.arange(3)] = -LINE
    partial = np.zeros(n)
    partial[FIELD + 3 + np.arange(3)] = -LINE_RATE
    return sp.csr_matrix(jacobian), mass, residual, partial, star, open_node


def _reversed_within_rows(matrix):
    """The same matrix with every row's column indices stored in DESCENDING order (valid, unsorted CSR)."""
    matrix = sp.csr_matrix(matrix)
    indices, data = matrix.indices.copy(), matrix.data.copy()
    for row in range(matrix.shape[0]):
        lo, hi = matrix.indptr[row], matrix.indptr[row + 1]
        indices[lo:hi], data[lo:hi] = indices[lo:hi][::-1], data[lo:hi][::-1]
    out = sp.csr_matrix((data, indices, matrix.indptr.copy()), shape=matrix.shape)
    out.has_sorted_indices = False
    return out


def _complete(jacobian, mass, residual, partial):
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
    return sparse_complete_initial_rate(
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


def _border_sees(monkeypatch, order):
    """Hand the border method its actual equations in the given CSR storage order; record what it was handed."""
    real = initial_sparse.complete_rate_border
    handed = []

    def stored(matrix, *args, **kwargs):
        matrix = sp.csr_matrix(matrix) if order == 'sorted' else _reversed_within_rows(matrix)
        handed.append((matrix, matrix.indices.copy(), matrix.data.copy()))
        return real(matrix, *args, **kwargs)

    monkeypatch.setattr(initial_sparse, 'complete_rate_border', stored)
    return handed


@pytest.mark.parametrize('order', ['sorted', 'reversed'])
def test_the_open_node_rate_is_completed_whatever_the_storage_order(monkeypatch, order) -> None:
    jacobian, mass, residual, partial, star, open_node = _transformer_with_an_open_node()
    reference = _complete(jacobian, mass, residual, partial)
    handed = _border_sees(monkeypatch, order)
    completed = _complete(jacobian, mass, residual, partial)
    assert handed, 'the border method was never reached: the planted order tested nothing'
    assert completed.rate_is_unique[star]
    assert completed.rate[open_node] == pytest.approx(0.0, abs=1e-9)  # volts per second: its own row pins it
    np.testing.assert_allclose(completed.rate, reference.rate, rtol=1e-9, atol=1e-9)


def test_the_caller_matrix_is_never_reordered(monkeypatch) -> None:
    jacobian, mass, residual, partial, _star, _open = _transformer_with_an_open_node()
    handed = _border_sees(monkeypatch, 'reversed')
    _complete(jacobian, mass, residual, partial)
    matrix, indices, data = handed[0]
    np.testing.assert_array_equal(matrix.indices, indices)
    np.testing.assert_array_equal(matrix.data, data)
