"""The kernels moved from motronics-studio answer through ``numerics_lab.kernels`` as their Rust tests say.

Bit identity with SciPy is the Rust fixture's (``crates/numerics/tests/expm_is_scipy.rs``); here the seam is
checked: shapes in and out, refusals as ``ValueError``, and each result against its definition.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.interpolate
import scipy.linalg
import scipy.optimize

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


def test_an_armed_guard_scan_arrives_at_its_threshold() -> None:
    grid = np.linspace(0.0, 2.0, 17)
    time, crossings, sides, histories = kernels.guard_scan(
        np.array([[0.0, 1.0], [0.0, 0.0]]), np.array([1.0, -1.0]), np.array([[1.0, 0.0]]), 0.0, grid,
        [(1.0, 1, False, 0)], [1e-6], [None], 1e-9, 1,
    )  # fmt: skip
    assert (crossings, sides) == ([0], [-1])
    assert time == pytest.approx(1.0 + 1e-6, abs=1e-9)  # s
    assert histories[0][1:3] == (0, True)


def test_a_falling_row_first_crosses_at_its_zero_and_an_early_end_finds_none() -> None:
    system = np.array([[0.0, -1.0], [0.0, 0.0]])
    rows = np.array([[1.0, 0.0]])
    _grid, found = kernels.first_crossing(system, ([0.0, 0.0], [0.0, 0.0]), rows, [0.0], [1e-9], [1.0, 1.0], 2.0)
    assert found is not None
    assert found[1] == [0]
    assert found[0] == pytest.approx(1.0, abs=1e-12)  # s
    assert kernels.first_crossing(system, ([0.0, 0.0], [0.0, 0.0]), rows, [0.0], [1e-9], [1.0, 1.0], 0.9)[1] is None
    assert kernels.span_key(1e-3 * (1.0 + 1e-14)) == 1e-3


def test_the_pchip_and_its_antiderivative_are_scipys() -> None:
    x, y = np.array([0.0, 0.2, 0.5, 0.9, 1.4]), np.array([0.0, 40.0, 120.0, 300.0, 1200.0])
    want = scipy.interpolate.PchipInterpolator(x, y)
    q = np.array([-0.1, 0.0, 0.1, 0.45, 0.9, 1.3, 1.6])
    assert np.asarray(kernels.pchip_slopes(x, y)) == pytest.approx(want(x, 1), rel=1e-12, abs=1e-9)  # y per x
    p = kernels.PiecewisePoly.pchip(x, y)
    for nu in range(3):
        assert np.asarray(p(q, nu)) == pytest.approx(want(q, nu), rel=1e-12, abs=1e-9)  # y per x**nu
    assert np.asarray(p.antiderivative()(q)) == pytest.approx(want.antiderivative()(q), rel=1e-12, abs=1e-12)
    assert np.asarray(p.breakpoints) == pytest.approx(x, abs=0.0)
    assert np.asarray(p.coefficients) == pytest.approx(want.c.T, rel=1e-12, abs=1e-9)
    moved = scipy.interpolate.PchipInterpolator(x, y + np.array([0.0, 0.0, 1e-6, 0.0, 0.0]))
    assert np.asarray(p(q)) != pytest.approx(moved(q), rel=1e-12, abs=1e-12)
    assert np.asarray(kernels.PiecewisePoly.linear(x, y)(np.array([0.1]))) == pytest.approx([20.0], abs=1e-12)
    with pytest.raises(ValueError, match='ascending'):
        kernels.PiecewisePoly.pchip(x[::-1].copy(), y)


def test_nnls_is_scipys_and_refuses_as_scipy_does() -> None:
    a = np.array([[1.0, 2.0, 0.5], [0.0, 1.0, -1.0], [3.0, -1.0, 2.0], [1.0, 1.0, 1.0]])
    b = np.array([1.0, -2.0, 3.0, 0.5])
    x, rnorm = kernels.nnls(a, b)
    want_x, want_r = scipy.optimize.nnls(a, b)
    assert np.asarray(x) == pytest.approx(want_x, rel=1e-10, abs=1e-12)  # dimensionless weights
    assert rnorm == pytest.approx(want_r, rel=1e-10, abs=1e-12)  # b's unit
    assert np.asarray(x) != pytest.approx(want_x + 1e-6, rel=1e-10, abs=1e-12)
    with pytest.raises(ValueError, match='entries'):
        kernels.nnls(a, b[:3])
    with pytest.raises(RuntimeError, match='maximum'):
        kernels.nnls(np.eye(2), np.ones(2), maxiter=1)
