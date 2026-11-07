"""A real square factor preconditions every original physical row at the same bar."""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from numerics_lab.linear import initial_rate


@pytest.mark.parametrize(('contradiction', 'allowance'), [(0.0, 0.0), (0.1, 0.0), (0.1, 0.2)])
@pytest.mark.parametrize('unavailable', [False, True])
def test_real_all_row_candidate_keeps_contradictions_allowance_and_unavailable_coordinates(
    contradiction, allowance, unavailable
):
    size = 600
    head = sp.csr_matrix([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
    matrix = sp.block_diag((head, sp.eye(size - 2)), format='csr')
    expected = np.r_[0.75, 0.25, np.linspace(-1.0, 1.0, size - 2)]
    rhs = matrix @ expected
    rhs[2] += contradiction
    if unavailable:
        matrix = sp.hstack((matrix, sp.csr_matrix((size + 1, 1))), format='csr')
    selected = np.arange(size)
    reduced, forcing = initial_rate.normalize_sparse_rows(matrix, rhs)
    reduced = reduced[:, selected].tocsr()
    rows = np.r_[0, 1, np.arange(3, size + 1)]
    factor = splu(reduced[rows].tocsc())
    bounds = np.full(len(rhs), allowance)
    before = (matrix.data.copy(), rhs.copy(), bounds.copy())
    values = initial_rate._preconditioned_all_row_rate_values(
        factor, matrix, rhs, reduced, forcing, selected, tolerance=1e-10, allowance=bounds
    )
    if contradiction and not allowance:
        assert values is None
    else:
        assert values is not None
        answer = np.zeros(matrix.shape[1])
        answer[selected] = values
        bar = 1e-10 * (abs(matrix) @ abs(answer) + abs(rhs)) + bounds
        assert np.all(abs(matrix @ answer - rhs) <= bar)
        if not contradiction:
            np.testing.assert_allclose(values, expected, rtol=0, atol=1e-12)
    for actual, original in zip((matrix.data, rhs, bounds), before, strict=True):
        np.testing.assert_array_equal(actual, original)


def test_representable_tiny_physical_coefficient_is_not_discarded():
    diagonal = np.ones(600)
    diagonal[-1] = 1e-300
    matrix = sp.diags(diagonal, format='csr')
    rhs = diagonal.copy()
    reduced, forcing = initial_rate.normalize_sparse_rows(matrix, rhs)
    factor = splu(reduced.tocsc())
    values = initial_rate._preconditioned_all_row_rate_values(
        factor, matrix, rhs, reduced, forcing, np.arange(600), tolerance=1e-10, allowance=np.zeros(600)
    )
    assert values is not None
    np.testing.assert_allclose(values, np.ones(600), rtol=0, atol=1e-12)
    assert matrix.data[-1] == 1e-300


def test_healthy_square_never_enters_all_row_refinement(monkeypatch):
    matrix = sp.diags(np.arange(1.0, 601.0), format='csr')
    expected = np.linspace(-1.0, 1.0, 600)
    rhs = matrix @ expected
    original = initial_rate._preconditioned_all_row_rate_values
    calls = []

    def observe(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(initial_rate, '_preconditioned_all_row_rate_values', observe)
    answer, unique = initial_rate.sparse_initial_rate(matrix, rhs)
    assert not calls
    assert unique.all()
    np.testing.assert_allclose(answer, expected, rtol=0, atol=1e-15)
