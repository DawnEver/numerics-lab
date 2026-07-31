"""Pluggable INNER linear-solve seam -- one registry of ``solve(matrix, rhs) -> x`` backends.

Roadmap ``src/motronics/hamilton/.claude/memory/2026/07/09/roadmap-solver-methods.md`` sec 2: today every solve is a direct
sparse factorization (MEC ``_ReluctanceSystem.solve_system`` and FEA ``fea/_solve.py::_solve`` both
call :func:`scipy.sparse.linalg.spsolve`). This module extracts that one call behind a
:data:`LINEAR_SOLVER_REGISTRY` so a solve can pick a KRYLOV iterative solver (needed by JFNK /
newton-krylov and by large 3D / eddy-current systems) without the equation code changing.

First principle (``AGENTS.md``): IMPLEMENT every method and characterize it. The ``direct`` backend is
today's ``spsolve`` VERBATIM (bit-for-bit default, so every golden is unchanged); the iterative
backends are additive peers, each of which must reach the SAME solution as ``direct`` within its
tolerance (the linear analogue of the nonlinear "converge-to-reference" contract).

The seam is deliberately tiny: a :class:`LinearSolver` is a callable
``(matrix, rhs, *, x0=None) -> x``. A peer :data:`PRECONDITIONER_REGISTRY` builds a
:class:`scipy.sparse.linalg.LinearOperator` approximating ``matrix^{-1}`` (Jacobi diagonal / ILU),
handed to the Krylov backends as ``M``. This module touches no physics and no mesh -- it only solves
``matrix @ x = rhs``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Protocol

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import (
    LinearOperator,
    MatrixRankWarning,
    bicgstab,
    cg,
    gmres,
    minres,
    qmr,
    spilu,
    spsolve,
)

from motronics.gauss.linear.dirichlet import apply_dirichlet_triplets
from motronics.gauss.linear.pin_rows import pin_rows
from motronics.gauss.linear.ssor import ssor_preconditioner

__all__ = [
    'LINEAR_SOLVER_REGISTRY',
    'PRECONDITIONER_REGISTRY',
    'LinearSolveError',
    'LinearSolver',
    'apply_dirichlet_triplets',
    'get_linear_solver',
    'make_preconditioner',
    'pin_rows',
    'requires_symmetric',
    'solve_linear_system',
    'ssor_preconditioner',
]

# Krylov convergence target: tight enough that the iterative solution matches the direct factorization
# to the accuracy the nonlinear outer loop needs (the outer energy tolerance is ~1e-4). atol=0 makes
# the stop purely relative so a small-norm rhs still converges to full relative accuracy.
_KRYLOV_RTOL = 1e-10
_KRYLOV_ATOL = 0.0
_KRYLOV_MAXITER_FACTOR = 4  # maxiter = factor * n, a generous ceiling before declaring non-convergence

# Direct-solve singular guard: even when SuperLU returns a FINITE vector for an (exactly or
# numerically) singular system -- while emitting a MatrixRankWarning -- the relative residual
# ||A x - b|| / ||b|| is large. We reject above this tolerance so a rank-deficient reluctance
# network fails honestly instead of shipping garbage as ``status=ok``. The bound is loose (1e-6)
# so a well-conditioned direct factorization (residual ~ machine eps) always passes.
_DIRECT_RESIDUAL_RTOL = 1e-6


class LinearSolveError(RuntimeError):
    """A linear backend failed to produce a finite solution (singular system / Krylov breakdown).

    Raised instead of returning silent NaNs, mirroring the MEC singular-system guard: a
    reference-free network or a Krylov breakdown is a hard failure the caller must see.
    """


class LinearSolver(Protocol):
    """A linear-solve backend: ``matrix @ x = rhs`` for ``x``, optionally seeded by ``x0``.

    Direct backends ignore ``x0``; iterative (Krylov) backends use it as the initial guess (a warm
    start from the previous nonlinear iterate). ``M`` is an optional preconditioner
    :class:`~scipy.sparse.linalg.LinearOperator` (ignored by direct backends).
    """

    def __call__(self, matrix, rhs: np.ndarray, *, x0: np.ndarray | None = None, M=None) -> np.ndarray: ...


def _check_finite(x: np.ndarray, backend: str) -> np.ndarray:
    if not np.all(np.isfinite(x)):
        msg = (
            f'linear solver {backend!r} produced a non-finite solution (singular system or breakdown); '
            'is a Dirichlet boundary set / is the matrix well-conditioned?'
        )
        raise LinearSolveError(msg)
    return x


def _direct(matrix, rhs, *, x0=None, M=None):  # noqa: ARG001 -- direct ignores x0/M by contract
    """Sparse direct factorization -- today's ``spsolve`` (the bit-for-bit default), singular-guarded.

    Kept byte-identical to the inline ``spsolve`` the MEC/FEA solvers used for a well-conditioned
    system, so routing the default path through this registry changes nothing on the happy path.

    The one addition is a SINGULAR guard: SuperLU emits a :class:`MatrixRankWarning` when the factor
    hits a zero/near-zero pivot, then returns EITHER NaNs OR a FINITE-but-meaningless vector. The
    finite case slipped past the caller's finiteness check and shipped as ``status=ok`` garbage. We
    promote that warning to a hard :class:`LinearSolveError` here (belt), and the caller additionally
    rejects a large relative residual (suspenders), so a rank-deficient reluctance network -- e.g. a
    feature-aligned polar-grid loop basis that is not full-rank at some mesh resolution -- fails
    honestly instead of returning a wrong number.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('error', MatrixRankWarning)
        try:
            return spsolve(matrix, rhs)
        except MatrixRankWarning as exc:
            msg = (
                'direct solver spsolve reports the matrix is (near-)singular '
                f'({exc}); the linear system is rank-deficient -- is a gauge/reference set '
                'and is the (loop) basis full-rank at this mesh resolution?'
            )
            raise LinearSolveError(msg) from exc


def _krylov(fn: Callable, *, symmetric: bool):
    """Wrap a scipy Krylov solver as a :class:`LinearSolver`.

    ``symmetric`` documents the intended operator class (cg/minres assume SPD/symmetric); the wrapper
    itself is agnostic. Converts the rhs to dense (scipy Krylov wants a 1-D array), forwards ``x0``
    as the warm start and ``M`` as the preconditioner, and raises :class:`LinearSolveError` on a
    non-zero ``info`` (non-convergence within maxiter or a breakdown) rather than returning a wrong x.

    The scipy Krylov signatures are NOT uniform: ``minres`` has no ``atol`` (it takes ``rtol``/
    ``shift``), and ``qmr`` splits the preconditioner into left/right ``M1``/``M2``. Per-solver kwargs
    handle those quirks so every backend shares one call site.
    """

    def _solve(matrix, rhs, *, x0=None, M=None):
        b = np.asarray(rhs, dtype=float).ravel()
        n = b.shape[0]
        kw = {
            'x0': None if x0 is None else np.asarray(x0, dtype=float).ravel(),
            'rtol': _KRYLOV_RTOL,
            'maxiter': _KRYLOV_MAXITER_FACTOR * max(n, 1),
        }
        if fn is qmr:
            kw['M1'] = M  # qmr takes a left preconditioner M1 (M2 defaults to identity)
        else:
            kw['M'] = M
        if fn is not minres:
            kw['atol'] = _KRYLOV_ATOL  # minres has no atol argument
        x, info = fn(matrix, b, **kw)
        if info != 0:
            kind = 'did not converge' if info > 0 else 'illegal input / breakdown'
            msg = f'Krylov solver {fn.__name__!r} {kind} (info={info}) on an n={n} system'
            raise LinearSolveError(msg)
        return x

    _solve.__name__ = fn.__name__
    # Tag the backend with the operator class it assumes (cg/minres want SPD/symmetric); the
    # benchmark harness reads this to skip a symmetric-only solver on a non-symmetric Jacobian.
    _solve.requires_symmetric = symmetric
    return _solve


def requires_symmetric(name: str) -> bool:
    """Whether the named linear backend assumes a symmetric/SPD operator (cg / minres)."""
    return bool(getattr(LINEAR_SOLVER_REGISTRY[name], 'requires_symmetric', False))


# name -> LinearSolver. 'direct' is the bit-for-bit default (spsolve); the Krylov peers each reach the
# same solution within _KRYLOV_RTOL. cg/minres want a symmetric/SPD operator (the magnetostatic
# secant stiffness is SPD-ish); gmres/bicgstab/qmr handle general (non-symmetric, e.g. eddy-current).
LINEAR_SOLVER_REGISTRY: dict[str, LinearSolver] = {
    'direct': _direct,
    'cg': _krylov(cg, symmetric=True),
    'minres': _krylov(minres, symmetric=True),
    'gmres': _krylov(gmres, symmetric=False),
    'bicgstab': _krylov(bicgstab, symmetric=False),
    'qmr': _krylov(qmr, symmetric=False),
}


def _jacobi_preconditioner(matrix) -> LinearOperator:
    """Diagonal (Jacobi) preconditioner ``M ~ diag(matrix)^{-1}``.

    The cheapest useful preconditioner: scales each row by its diagonal. A zero/NaN diagonal entry
    (a pinned identity row is 1, so this is rare) is left as 1 so ``M`` stays finite.
    """
    diag = np.asarray(matrix.diagonal(), dtype=float)
    inv = np.where(np.abs(diag) > 0.0, 1.0 / diag, 1.0)
    n = diag.shape[0]
    return LinearOperator((n, n), matvec=lambda v: inv * v)


def _ilu_preconditioner(matrix) -> LinearOperator:
    """Incomplete-LU preconditioner ``M ~ matrix^{-1}`` via :func:`scipy.sparse.linalg.spilu`.

    A much stronger preconditioner than Jacobi for the sparse magnetostatic operator. Requires CSC;
    the factorization solve is wrapped as a :class:`LinearOperator` the Krylov backends apply as ``M``.
    """
    ilu = spilu(sp.csc_matrix(matrix))
    n = matrix.shape[0]
    return LinearOperator((n, n), matvec=ilu.solve)


# name -> builder(matrix) -> LinearOperator. AMG (pyamg) is an optional dep -- the hook is here
# (add 'amg' when pyamg is present) but no hard dependency is introduced. 'ssor' is the nodal
# auxiliary accelerator (symmetric, strengthens CG on the SPD Poisson-type nodal operator); its
# implementation lives in the physics-blind ``gauss.linear.ssor`` module (pure algebra on the
# matrix triangles), imported at module level -- no cycle back into any physics package.
PRECONDITIONER_REGISTRY: dict[str, Callable[[object], LinearOperator]] = {
    'jacobi': _jacobi_preconditioner,
    'ilu': _ilu_preconditioner,
    'ssor': ssor_preconditioner,
}


def get_linear_solver(name: str | None) -> LinearSolver:
    """Resolve a linear-solver name; ``None`` -> the default ``direct`` (today's spsolve, bit-for-bit)."""
    if name is None:
        return LINEAR_SOLVER_REGISTRY['direct']
    try:
        return LINEAR_SOLVER_REGISTRY[name]
    except KeyError as e:
        msg = f'unknown linear solver {name!r}; known: {sorted(LINEAR_SOLVER_REGISTRY)}'
        raise KeyError(msg) from e


def make_preconditioner(name: str | None, matrix) -> LinearOperator | None:
    """Build a preconditioner :class:`LinearOperator` for ``matrix``; ``None`` -> no preconditioner."""
    if name is None:
        return None
    try:
        builder = PRECONDITIONER_REGISTRY[name]
    except KeyError as e:
        msg = f'unknown preconditioner {name!r}; known: {sorted(PRECONDITIONER_REGISTRY)}'
        raise KeyError(msg) from e
    return builder(matrix)


def solve_linear_system(
    matrix,
    rhs: np.ndarray,
    *,
    linear_solver: str | None = None,
    preconditioner: str | None = None,
    x0: np.ndarray | None = None,
) -> np.ndarray:
    """Solve ``matrix @ x = rhs`` with the named backend, guarding a non-finite result.

    ``linear_solver=None`` is the default ``direct`` (bit-for-bit ``spsolve``). A Krylov backend may
    take an optional ``preconditioner`` (Jacobi / ILU) and a warm-start ``x0``. The finite-solution
    guard is uniform across backends (the MEC singular-system guard, generalized), so a singular
    system or a Krylov breakdown raises :class:`LinearSolveError` instead of propagating NaNs.
    """
    name = linear_solver or 'direct'
    backend = get_linear_solver(linear_solver)
    M = make_preconditioner(preconditioner, matrix) if linear_solver not in (None, 'direct') else None
    x = _check_finite(np.asarray(backend(matrix, rhs, x0=x0, M=M)), name)
    # Direct-solve residual guard (suspenders to _direct's MatrixRankWarning belt): a singular
    # SuperLU factorization can return a FINITE vector that does not satisfy A x = b. Reject a large
    # relative residual so such a solve raises rather than reporting garbage as ``status=ok``. Only
    # the direct backend needs this -- the Krylov backends already raise on a non-zero convergence
    # ``info`` and asserting their (looser) residual here would clash with their own stop criterion.
    if name == 'direct':
        b = np.asarray(rhs, dtype=float).ravel()
        b_norm = float(np.linalg.norm(b))
        if b_norm > 0.0:
            residual = float(np.linalg.norm(np.asarray(matrix @ x, dtype=float).ravel() - b)) / b_norm
            if residual > _DIRECT_RESIDUAL_RTOL:
                msg = (
                    f'direct solve residual ||A x - b|| / ||b|| = {residual:.3e} exceeds '
                    f'{_DIRECT_RESIDUAL_RTOL:.0e}; the system is singular / rank-deficient '
                    '(the factorization returned a finite but meaningless solution)'
                )
                raise LinearSolveError(msg)
    return x
