"""The kernels moved from motronics-studio answer through ``numerics_lab.kernels`` as their Rust tests say.

Bit identity with SciPy is the Rust fixture's (``crates/numerics/tests/expm_is_scipy.rs``); here the seam is
checked: shapes in and out, refusals as ``ValueError``, and each result against its definition.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg

from numerics_lab import kernels


def test_expm_is_scipys_and_a_moved_entry_is_seen() -> None:
    a = np.array([[-1.0, 0.5, 0.0], [0.2, -2.0, 0.3], [0.0, 0.1, -3.0]])
    got = np.asarray(kernels.expm(a)).reshape(3, 3)
    want = scipy.linalg.expm(a)
    assert got == pytest.approx(want, rel=1e-13, abs=1e-15)  # dimensionless
    moved = scipy.linalg.expm(a + np.diag([1e-6, 0.0, 0.0]))
    assert got != pytest.approx(moved, rel=1e-13, abs=1e-15)
    with pytest.raises(ValueError, match='square'):
        kernels.expm(np.zeros((2, 3)))


def test_dense_solve_and_cond_are_numpys() -> None:
    a = np.array([[2.0, 1.0], [1.0, 3.0]])
    assert np.asarray(kernels.dense_solve(a, np.array([3.0, 5.0]))) == pytest.approx([0.8, 1.4], abs=1e-15)
    assert kernels.dense_solve(np.array([[1.0, 2.0], [2.0, 4.0]]), np.ones(2)) is None
    assert kernels.dense_cond(a) == np.linalg.cond(a)
    with pytest.raises(ValueError, match='transcribed'):
        kernels.dense_cond(np.eye(3))


def test_a_tridiagonal_pattern_factors_without_fill() -> None:
    n = 20
    cols = [sorted({max(j - 1, 0), j, min(j + 1, n - 1)}) for j in range(n)]
    indptr = np.cumsum([0] + [len(c) for c in cols]).astype(np.int64)
    indices = np.concatenate(cols).astype(np.int64)
    assert kernels.cholesky_factor_len(n, indptr, indices) == 2 * n - 1
    with pytest.raises(ValueError, match='negative'):
        kernels.cholesky_factor_len(n, indptr, -indices - 1)


def test_the_condensed_step_solves_the_block() -> None:
    rng = np.random.default_rng(3)
    k, nb = 2, 3
    j = rng.standard_normal((k + nb, k + nb)) + 4.0 * np.eye(k + nb)
    cells, schur = kernels.condense(j.ravel().copy(), np.array([[k, nb]], dtype=np.int64))
    s = j[:k, :k] - j[:k, k:] @ np.linalg.solve(j[k:, k:], j[k:, :k])
    assert np.asarray(schur).reshape(k, k) == pytest.approx(s, rel=1e-12, abs=1e-14)
    rhs_b, dv = rng.standard_normal(nb), rng.standard_normal(k)
    y, _corr = cells.eliminate(rhs_b)
    x_b = np.asarray(cells.recover(np.asarray(y), dv))
    assert j[k:, k:] @ x_b + j[k:, :k] @ dv == pytest.approx(rhs_b, abs=1e-12)
    u = rng.standard_normal(k + nb)
    assert np.asarray(kernels.block_matvec(j.ravel().copy(), np.array([k + nb], dtype=np.int64), u)) == pytest.approx(
        j @ u, abs=1e-12
    )
