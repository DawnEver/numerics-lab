"""Nodal auxiliary (SSOR) preconditioner for the magnetostatic Krylov path (handoff §4).

Hiptmair-Xu / AMS is NOT applicable here -- it preconditions the curl-curl (edge-element,
H(curl)) operator, which this codebase does not have (the FEA solve is the SPD 2D scalar
nodal Poisson ``-div(nu grad A) = J``). See
``.claude/memory/2026/07/10/finding-hiptmair-xu-assessment.md`` and the module docstring of
``gauss/linear/ssor.py`` (SSOR relocated there as a physics-blind algebraic primitive).

These tests characterize the honest in-scope increment instead: an additive SSOR
preconditioner that (a) reaches the SAME solution as a direct solve (the registry's
converge-to-reference contract) and (b) MEASURABLY reduces the CG iteration count on a
representative SPD Poisson-type operator -- the value SSOR delivers on the nodal path that
Jacobi (a no-op on the constant-diagonal Laplacian) does not.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import cg, spsolve

from numerics_lab.linear import (
    PRECONDITIONER_REGISTRY,
    make_preconditioner,
    solve_linear_system,
    ssor_preconditioner,
)


def _poisson2d(n: int):
    """The 2D 5-point Laplacian on an n x n grid: SPD, Poisson-type -- the nodal operator's
    structure (a peer of the assembled ``-div(nu grad A)`` stiffness), with a random rhs.
    """
    size = n * n
    main = 4.0 * np.ones(size)
    off = -1.0 * np.ones(size - 1)
    off[np.arange(1, size) % n == 0] = 0.0  # break the wrap-around coupling at row ends
    updown = -1.0 * np.ones(size - n)
    a = sp.diags([updown, off, main, off, updown], offsets=[-n, -1, 0, 1, n], format='csr')
    rng = np.random.default_rng(0)
    b = rng.standard_normal(size)
    return a.tocsr(), b


def _cg_iters(a, b, m=None) -> int:
    """CG iteration count to a tight relative tolerance with preconditioner ``m`` (None = plain)."""
    count = {'k': 0}
    cg(
        a,
        b,
        rtol=1e-8,
        atol=0.0,
        M=m,
        maxiter=5 * a.shape[0],
        callback=lambda _xk: count.__setitem__('k', count['k'] + 1),
    )
    return count['k']


def test_ssor_reaches_direct_solution():
    """SSOR-preconditioned CG (via the registered 'ssor' option) reaches the direct answer."""
    a, b = _poisson2d(16)
    ref = spsolve(a, b)
    x = solve_linear_system(a, b, linear_solver='cg', preconditioner='ssor')
    np.testing.assert_allclose(x, ref, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize('n', [16, 24, 32])
def test_ssor_reduces_cg_iterations_vs_unpreconditioned(n):
    """SSOR MEASURABLY cuts the CG iteration count on the Poisson-type operator.

    Documented band: SSOR must reach the tolerance in at most 60% of the unpreconditioned
    iterations (measured ~45%: 52->23, 77->31, 102->41 at N=256/576/1024). This is the value
    the nodal auxiliary preconditioner delivers -- a real, mesh-growing acceleration.
    """
    a, b = _poisson2d(n)
    plain = _cg_iters(a, b)
    ssor = _cg_iters(a, b, ssor_preconditioner(a))
    assert ssor < 0.6 * plain, f'SSOR iters={ssor} not < 60% of plain={plain} (N={n * n})'


def test_ssor_beats_jacobi_on_constant_diagonal_laplacian():
    """Jacobi is a no-op on the constant-diagonal Laplacian; SSOR strictly beats it.

    This is why a NEW preconditioner is warranted rather than reusing Jacobi: Jacobi's
    diagonal scaling does nothing when the diagonal is constant, so it cannot accelerate the
    nodal Poisson solve. SSOR (full-triangle) does.
    """
    a, b = _poisson2d(24)
    jacobi_iters = _cg_iters(a, b, make_preconditioner('jacobi', a))
    ssor_iters = _cg_iters(a, b, ssor_preconditioner(a))
    plain_iters = _cg_iters(a, b)
    assert jacobi_iters == plain_iters  # Jacobi is a no-op on a constant diagonal
    assert ssor_iters < jacobi_iters


def test_ssor_is_registered_additively():
    """'ssor' is a registered preconditioner option alongside the existing jacobi/ilu set."""
    assert {'jacobi', 'ilu'} <= set(PRECONDITIONER_REGISTRY)  # existing entries untouched
    assert 'ssor' in PRECONDITIONER_REGISTRY
    # the registered builder produces the same operator as the direct implementation
    a, b = _poisson2d(16)
    via_registry = make_preconditioner('ssor', a)
    direct = ssor_preconditioner(a)
    np.testing.assert_allclose(via_registry.matvec(b), direct.matvec(b), rtol=1e-12, atol=0.0)


def test_ssor_rejects_out_of_range_omega():
    a, _ = _poisson2d(8)
    for bad in (0.0, 2.0, -0.5, 2.5):
        with pytest.raises(ValueError, match='omega'):
            ssor_preconditioner(a, omega=bad)


def test_ssor_symmetric_operator_preserves_cg_validity():
    """The SSOR operator is symmetric (as A is SPD), so CG stays a valid Krylov method.

    Applying M to a dense basis reconstructs a symmetric matrix -- unlike a non-symmetric
    ILU factor, which would break CG's symmetry assumption on the SPD nodal operator.
    """
    a, _ = _poisson2d(8)
    m = ssor_preconditioner(a)
    n = a.shape[0]
    dense = np.column_stack([m.matvec(e) for e in np.eye(n)])
    np.testing.assert_allclose(dense, dense.T, rtol=1e-9, atol=1e-11)
