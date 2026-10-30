"""Jittered Cholesky reports the jitter it used; the KKT solve is the constrained minimiser, dense or sparse."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.linalg import LinAlgError

from numerics_lab.linear import LinearSolveError
from numerics_lab.linear.cholesky import jittered_cholesky
from numerics_lab.linear.kkt import solve_kkt

_RNG = np.random.default_rng(3)


def test_a_positive_definite_matrix_takes_no_jitter() -> None:
    m = _RNG.standard_normal((6, 6))
    a = m @ m.T + np.eye(6)
    factor = jittered_cholesky(a)
    assert factor.jitter == 0.0
    np.testing.assert_allclose(factor.lower @ factor.lower.T, a, rtol=1e-12)


def test_a_singular_gram_matrix_is_factored_with_the_reported_jitter() -> None:
    v = _RNG.standard_normal((6, 2))
    gram = v @ v.T  # rank 2: not factorable as it stands
    with pytest.raises(LinAlgError):
        np.linalg.cholesky(gram - 1e-6 * np.eye(6))
    factor = jittered_cholesky(gram)
    assert factor.jitter > 0.0
    np.testing.assert_allclose(factor.lower @ factor.lower.T, gram + factor.jitter * np.eye(6), atol=1e-12)
    # The solve is of the jittered matrix, to its backward error (it is badly conditioned by design).
    jittered = gram + factor.jitter * np.eye(6)
    rhs = _RNG.standard_normal(6)
    x = factor.solve(rhs)
    assert np.linalg.norm(jittered @ x - rhs) <= 1e-8 * np.linalg.norm(jittered, 2) * np.linalg.norm(x)


def test_a_clearly_indefinite_matrix_is_refused() -> None:
    with pytest.raises(LinearSolveError, match='jitter'):
        jittered_cholesky(np.diag([1.0, -1.0]))


def _problem():
    rng = np.random.default_rng(5)
    n, m = 7, 3
    q = rng.standard_normal((n, n))
    h = q @ q.T + np.eye(n)
    a = rng.standard_normal((m, n))
    return h, a, rng.standard_normal(n), rng.standard_normal(m)


def test_the_kkt_solution_is_the_constrained_minimiser() -> None:
    h, a, c, d = _problem()
    solution = solve_kkt(h, a, c, d)
    np.testing.assert_allclose(a @ solution.x, d, atol=1e-12)
    np.testing.assert_allclose(h @ solution.x + a.T @ solution.multipliers, c, atol=1e-11)
    # Planted control: a feasible perturbation along the null space of A raises the objective.
    null = np.linalg.svd(a)[2][-1]
    objective = lambda x: 0.5 * x @ h @ x - c @ x  # noqa: E731
    assert objective(solution.x + 1e-3 * null) > objective(solution.x)


def test_the_sparse_kkt_agrees_with_the_dense() -> None:
    h, a, c, d = _problem()
    dense = solve_kkt(h, a, c, d)
    sparse = solve_kkt(sp.csr_matrix(h), sp.csr_matrix(a), c, d)
    np.testing.assert_allclose(sparse.x, dense.x, rtol=1e-10)
    np.testing.assert_allclose(sparse.multipliers, dense.multipliers, rtol=1e-9)


def test_rank_deficient_constraints_are_refused_dense_and_sparse() -> None:
    h, a, c, d = _problem()
    a[2] = a[0]
    with pytest.raises(LinearSolveError, match='singular'):
        solve_kkt(h, a, c, d)
    with pytest.raises(LinearSolveError):
        solve_kkt(sp.csr_matrix(h), sp.csr_matrix(a), c, d)


def test_no_draw_of_a_singular_sparse_kkt_is_answered() -> None:
    """The bare ``direct`` backend answered about one draw in eight with garbage; the pivot-barred solve none."""
    refused = 0
    for seed in range(300):
        rng = np.random.default_rng(seed)
        q = rng.standard_normal((7, 7))
        a = rng.standard_normal((3, 7))
        a[2] = a[0]
        c, d = rng.standard_normal(7), rng.standard_normal(3)
        with pytest.raises(LinearSolveError):
            solve_kkt(sp.csr_matrix(q @ q.T + np.eye(7)), sp.csr_matrix(a), c, d)
        refused += 1
    assert refused == 300


def test_a_numerically_singular_sparse_kkt_is_refused_by_its_pivot_bar() -> None:
    """The macOS residue, planted: a repeated constraint row off by an ulp-scale 1e-15 is singular to working
    precision (cond * eps >= 1), and the residual guard alone accepted 47 of 1200 such draws on x86 with
    |x| up to 3.6e17. The factorisation's own pivot bar refuses every one, whatever the platform's arithmetic.
    """
    for seed in range(1200):
        rng = np.random.default_rng(seed)
        q = rng.standard_normal((7, 7))
        a = rng.standard_normal((3, 7))
        a[2] = a[0] + 1e-15 * rng.standard_normal(7)
        c, d = rng.standard_normal(7), rng.standard_normal(3)
        with pytest.raises(LinearSolveError, match='pivot'):
            solve_kkt(sp.csr_matrix(q @ q.T + np.eye(7)), sp.csr_matrix(a), c, d)
