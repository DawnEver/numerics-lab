"""The inner linear-solve seam: every backend reaches the SAME solution as ``direct`` (roadmap sec 2).

``direct`` is today's ``spsolve`` VERBATIM (bit-for-bit default); each Krylov peer (cg/minres/gmres/
bicgstab/qmr) must reach that same solution within tolerance -- the linear analogue of the nonlinear
"converge-to-reference" contract. Preconditioners (Jacobi/ILU) must not change the converged answer.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import splu, spsolve

from numerics_lab.linear import (
    LINEAR_SOLVER_REGISTRY,
    PRECONDITIONER_REGISTRY,
    LinearSolveError,
    get_linear_solver,
    native_direct,
    solve_linear_system,
)


def _spd_system(n=60, seed=0):
    """A well-conditioned SPD sparse matrix (a 1-D Laplacian + diagonal) and a rhs.

    SPD so cg/minres apply; sparse and diagonally dominant so every Krylov backend converges.
    """
    rng = np.random.default_rng(seed)
    main = 2.0 + rng.uniform(0.5, 1.5, n)
    off = -1.0 * np.ones(n - 1)
    A = sp.diags([off, main, off], offsets=[-1, 0, 1], format='csr')
    b = rng.standard_normal(n)
    return A.tocsr(), b


def _nonsymmetric_system(n=60, seed=1):
    """A well-conditioned NON-symmetric sparse matrix (asymmetric off-diagonals) and a rhs."""
    rng = np.random.default_rng(seed)
    main = 3.0 + rng.uniform(0.5, 1.5, n)
    lower = -1.0 * np.ones(n - 1)
    upper = -0.5 * np.ones(n - 1)
    A = sp.diags([lower, main, upper], offsets=[-1, 0, 1], format='csr')
    b = rng.standard_normal(n)
    return A.tocsr(), b


def test_direct_is_bit_for_bit_spsolve(monkeypatch):
    """With the native factor switched off, ``direct`` and the default route ARE raw ``spsolve``."""
    monkeypatch.setenv('NUMERICS_LAB_DIRECT_NATIVE', '0')
    A, b = _spd_system()
    ref = spsolve(A, b)
    np.testing.assert_array_equal(get_linear_solver('direct')(A, b), ref)
    np.testing.assert_array_equal(solve_linear_system(A, b), ref)  # None -> direct
    np.testing.assert_array_equal(solve_linear_system(A, b, linear_solver='direct'), ref)


@pytest.mark.parametrize('name', ['cg', 'minres', 'gmres', 'bicgstab', 'qmr'])
def test_krylov_reaches_direct_solution_spd(name):
    """Every Krylov backend converges to the direct (spsolve) solution on the SPD system."""
    A, b = _spd_system()
    ref = spsolve(A, b)
    x = solve_linear_system(A, b, linear_solver=name)
    np.testing.assert_allclose(x, ref, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize('name', ['gmres', 'bicgstab', 'qmr'])
def test_general_krylov_reaches_direct_solution_nonsymmetric(name):
    """The general (non-symmetric) Krylov backends converge on a non-symmetric system."""
    A, b = _nonsymmetric_system()
    ref = spsolve(A, b)
    x = solve_linear_system(A, b, linear_solver=name)
    np.testing.assert_allclose(x, ref, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize('precond', ['jacobi', 'ilu', 'ssor'])
@pytest.mark.parametrize('name', ['cg', 'gmres'])
def test_preconditioner_does_not_change_solution(name, precond):
    """A Jacobi/ILU/SSOR preconditioner accelerates but must reach the same converged solution."""
    A, b = _spd_system()
    ref = spsolve(A, b)
    x = solve_linear_system(A, b, linear_solver=name, preconditioner=precond)
    np.testing.assert_allclose(x, ref, rtol=1e-7, atol=1e-9)


def test_warm_start_reaches_same_solution():
    """A warm start (x0 near the answer) still converges to the direct solution."""
    A, b = _spd_system()
    ref = spsolve(A, b)
    x = solve_linear_system(A, b, linear_solver='cg', x0=ref + 1e-3)
    np.testing.assert_allclose(x, ref, rtol=1e-6, atol=1e-8)


def test_registry_covers_the_roadmap_backends():
    expected = {'direct', 'cg', 'minres', 'gmres', 'bicgstab', 'qmr'}
    assert expected <= set(LINEAR_SOLVER_REGISTRY)
    assert set(PRECONDITIONER_REGISTRY) == {'jacobi', 'ilu', 'ssor', 'ic0', 'amg'}


def test_the_symmetric_direct_backend_is_the_direct_answer_on_an_spd_system():
    """``direct-symmetric`` is a different ORDERING of the same factorization, not a new method."""
    n = 30
    lap = sp.diags([-np.ones(n - 1), 4.0 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1])
    A = (sp.kron(sp.eye(n), lap) + sp.kron(sp.diags([-np.ones(n - 1), -np.ones(n - 1)], [-1, 1]), sp.eye(n))).tocsc()
    b = np.random.default_rng(3).normal(size=n * n)
    x = solve_linear_system(A, b, linear_solver='direct-symmetric')
    np.testing.assert_allclose(x, spsolve(A, b), rtol=1e-12, atol=1e-12 * np.abs(x).max())


def test_the_symmetric_direct_backend_refuses_a_singular_system():
    A = sp.csc_matrix(np.array([[1.0, -1.0], [-1.0, 1.0]]))
    with pytest.raises(LinearSolveError):
        solve_linear_system(A, np.array([1.0, 0.0]), linear_solver='direct-symmetric')


def test_unknown_backend_raises():
    A, b = _spd_system(n=8)
    with pytest.raises(KeyError, match='unknown linear solver'):
        solve_linear_system(A, b, linear_solver='nope')
    with pytest.raises(KeyError, match='unknown preconditioner'):
        solve_linear_system(A, b, linear_solver='cg', preconditioner='nope')


def test_singular_system_raises_not_nan():
    """A singular system (zero row) must raise LinearSolveError, not return silent NaNs."""
    A = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 0.0]]))
    b = np.array([1.0, 1.0])
    with pytest.raises(LinearSolveError):
        solve_linear_system(A, b, linear_solver='direct')


def test_rank_deficient_direct_promotes_matrixrankwarning():
    """A rank-deficient (duplicate-row) system must raise, not slip through on a MatrixRankWarning.

    SuperLU emits a ``MatrixRankWarning`` on the zero pivot and may return NaNs OR a finite-but-
    meaningless vector. Either way the direct backend must promote it to ``LinearSolveError`` rather
    than reporting ``status=ok`` garbage -- this is the guard that closes the silent-singular gap.
    """
    A = sp.csr_matrix(np.array([[2.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 1.0, 3.0]]))
    b = np.array([1.0, 2.0, 1.0])
    with pytest.raises(LinearSolveError):
        solve_linear_system(A, b, linear_solver='direct')
    with pytest.raises(LinearSolveError):
        solve_linear_system(A, b)  # None -> direct


def test_high_residual_direct_solve_is_rejected(monkeypatch):
    """A finite-but-wrong direct solution (large ||A x - b||) is rejected by the residual guard.

    Simulate the pathological SuperLU case that returns finite garbage WITHOUT a warning: patch the
    ``direct`` backend to return a fixed finite vector that does not satisfy ``A x = b``. The
    finiteness check passes, so only the relative-residual guard can catch it.
    """
    A, b = _spd_system(n=8)

    def _finite_garbage(matrix, rhs, *, x0=None, M=None):  # noqa: ARG001 -- fixed wrong-but-finite x
        return np.ones_like(rhs)

    monkeypatch.setitem(LINEAR_SOLVER_REGISTRY, 'direct', _finite_garbage)
    with pytest.raises(LinearSolveError, match='residual'):
        solve_linear_system(A, b, linear_solver='direct')


def test_pardiso_is_not_a_backend() -> None:
    """Plan ruling R1: the engine depends on numpy and scipy only, so the MKL PARDISO entry is gone."""
    assert 'pardiso' not in LINEAR_SOLVER_REGISTRY
    A, b = _spd_system(n=8)
    with pytest.raises(KeyError, match="unknown linear solver 'pardiso'"):
        solve_linear_system(A, b, linear_solver='pardiso')


def test_the_symmetric_ordering_fills_less_than_colamd_on_a_2d_stiffness():
    """The CAUSE of the speedup, pinned deterministically -- fill, not seconds.

    MEASURED 2026-09-25: COLAMD fills this 200 x 200 five-point Laplacian 1.78x as much as the
    symmetric minimum-degree ordering (case_d's sdm loop matrix: 1.75x, 16.96 M / 9.69 M; its
    factor-and-solve 1.29 s -> 0.64 s). Wall time is NOT asserted: under a concurrent gate the same
    pair read 0.94x-1.11x, which is the box, not the ordering. Both backends agree on the answer.
    """
    n = 200
    lap = sp.diags([-np.ones(n - 1), 4.0 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1])
    A = (sp.kron(sp.eye(n), lap) + sp.kron(sp.diags([-np.ones(n - 1), -np.ones(n - 1)], [-1, 1]), sp.eye(n))).tocsc()
    colamd = splu(A)
    symmetric = splu(A, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0, options={'SymmetricMode': True})
    fill_ratio = (colamd.L.nnz + colamd.U.nnz) / (symmetric.L.nnz + symmetric.U.nnz)
    assert fill_ratio > 1.5, fill_ratio


def test_the_symmetric_direct_backend_takes_the_native_cholesky_first(monkeypatch):
    """``direct-symmetric`` answers through the pattern-cached native factor when it can (a verified
    symmetric operator: faer Cholesky), and through symmetric-mode SuperLU when it is switched off --
    the same answer to round-off either way.
    """
    if not native_direct.native_available():
        pytest.xfail('the optional native extension is not importable here')
    n = 30
    lap = sp.diags([-np.ones(n - 1), 4.0 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1])
    A = (sp.kron(sp.eye(n), lap) + sp.kron(sp.diags([-np.ones(n - 1), -np.ones(n - 1)], [-1, 1]), sp.eye(n))).tocsc()
    b = np.random.default_rng(5).normal(size=n * n)
    before = native_direct.DIRECT_SOLVES['faer-cholesky']
    x = solve_linear_system(A, b, linear_solver='direct-symmetric')
    assert native_direct.DIRECT_SOLVES['faer-cholesky'] == before + 1
    monkeypatch.setenv('NUMERICS_LAB_DIRECT_NATIVE', '0')
    y = solve_linear_system(A, b, linear_solver='direct-symmetric')
    assert native_direct.DIRECT_SOLVES['faer-cholesky'] == before + 1  # the planted control: SuperLU answered
    np.testing.assert_allclose(x, y, rtol=0, atol=1e-12 * np.abs(y).max())
