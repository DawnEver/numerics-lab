"""A joint minor's solve divides each defect by the row scale it factored with, computed once.

Profiled on the real whole-motor joint operands (capture47): re-normalizing the whole minor
on every refinement correction was the largest single cost of a projection column. The scale
is a property of the operator, so each solve only divides its forcing by it -- the same
arithmetic, never a copy of the operator.
"""

import numpy as np
import scipy.sparse as sp

from numerics_lab.linear import initial_storage_border


def _matrix():
    return sp.csr_matrix(
        np.array([
            [4.0, 0.5, -0.25, 8.0, 0.0, 0.0],
            [1.0, 3.0, 1.0, 0.0, 0.0, 0.0],
            [-1.0, 1.0, 4.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.25, -1.0, 0.5, 0.75, 6.0, 2.0],
            [0.5, 0.25, -1.0, -0.5, 0.5, 2.0],
        ])
    )


def test_a_solve_never_renormalizes_its_minor(monkeypatch):
    matrix = _matrix()
    solve = initial_storage_border._triangular_joint_solver(
        matrix, sp.diags([2.0, 0.0, 0.0], format='csr'), np.array([0]), np.array([0]), np.arange(3), np.array([0, 1])
    )
    calls = []
    original = initial_storage_border.normalize_sparse_rows
    monkeypatch.setattr(initial_storage_border, 'normalize_sparse_rows', lambda *a: calls.append(1) or original(*a))
    forcing = np.arange(1.0, 7.0)
    actual = solve(forcing)
    assert calls == []
    np.testing.assert_allclose(actual, np.linalg.solve(matrix.toarray(), forcing), rtol=0, atol=2e-15)
