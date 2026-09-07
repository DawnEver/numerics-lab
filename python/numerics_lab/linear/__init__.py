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
    splu,
    spsolve,
)

from motronics.gauss.linear.dirichlet import apply_dirichlet_triplets
from motronics.gauss.linear.gauge import gauge_pin_orphan_rows, orphan_rows
from motronics.gauss.linear.pin_rows import pin_rows
from motronics.gauss.linear.ssor import ssor_preconditioner

try:  # heavy optional dep (Intel MKL, the multi-threaded sparse LU -- MATLAB-parity on
    # the solver). The registry entry exists only when pypardiso is installed; a
    # 'pardiso' request without it is refused by name with the install hint below.
    from pypardiso import spsolve as _pardiso_spsolve
except ImportError:
    _pardiso_spsolve = None

__all__ = [
    'LINEAR_SOLVER_REGISTRY',
    'PRECONDITIONER_REGISTRY',
    'LinearSolveError',
    'LinearSolver',
    'apply_dirichlet_triplets',
    'direct_refinement_correction',
    'direct_relative_residual',
    'gauge_pin_orphan_rows',
    'get_linear_solver',
    'make_preconditioner',
    'orphan_rows',
    'pin_rows',
    'reject_unsolved_direct_system',
    'requires_symmetric',
    'solve_equilibrated',
    'solve_linear_system',
    'ssor_preconditioner',
]

# Krylov convergence target: tight enough that the iterative solution matches the direct factorization
# to the accuracy the nonlinear outer loop needs (the outer energy tolerance is ~1e-4). atol=0 makes
# the stop purely relative so a small-norm rhs still converges to full relative accuracy.
_KRYLOV_RTOL = 1e-10
_KRYLOV_ATOL = 0.0
_KRYLOV_MAXITER_FACTOR = 4  # maxiter = factor * n, a generous ceiling before declaring non-convergence

# Direct-solve singular guard, in TWO stages. The residual is the TRIGGER, never the verdict.
#
# ``||A x - b|| / ||b||`` is NOT a failure statistic: it tracks the CONDITION NUMBER of a
# perfectly good solve. Measured on the CST patch test (n=60), sweeping only Young's modulus --
# which changes the scaling of A and nothing else -- against the exact linear field:
#
#     E        cond(A)     ||Ax-b||/||b||   max err vs exact
#     1.0      1.36e+01       3.87e-16          4.3e-19
#     1e6      1.10e+07       4.17e-10          4.4e-17
#     1e9      1.10e+10       2.64e-07          5.5e-17
#     210e9    2.31e+12       1.13e-04          2.0e-17   <- steel, real SI properties
#     1e15     1.62e+16       2.41e-01          -
#
# The ANSWER is exact at every scale; only the residual moves, and it moves as ``cond * eps``.
# A fixed bar on it is therefore a disguised CONDITION test, and at 1e-6 it refused every
# native structural stationary solve at real SI material properties while calling them
# "singular". So the residual only decides whether the question is worth asking (stage 1);
# what ANSWERS it is one step of iterative refinement (stage 2, below).
_DIRECT_RESIDUAL_RTOL = 1e-6

# Stage 2 -- the verdict, and it tests what the CALLER depends on: the solution itself.
# One step of iterative refinement computes ``d = A^-1 (b - A x)``, an estimate of the error
# ALREADY IN x, and ``||d||/||x||`` is the relative amount a second solve would move the answer.
#
# This separates the two cases the residual conflates, by thirteen orders of magnitude. Measured
# with the same sweep plus the committed order-157 ``lipo_inductor`` finite-garbage plant:
#
#     cond(A)  1.4e+01 .. 7.1e+17 (well-posed, E sweep):  ||d||/||x|| in [4.3e-16, 8.7e-14]
#     cond(A)  1.5e+17 (the singular plant):              ||d||/||x|| = 4.7  (CSR) / 5.0 (CSC)
#
# Flat across the well-posed sweep -- unlike the residual, refinement does not track cond -- and
# 1e13 x larger on the system that must be refused. 1e-6 sits ~7 decades above the largest
# measured well-posed correction and ~7 below the plant, i.e. mid-gap in log space; it is also
# the coarsest accuracy any caller here tolerates (the outer energy tolerance is ~1e-4).
#
# TWO CANDIDATE CRITERIA WERE MEASURED AND REFUTED, so nobody re-derives them:
#   * the normwise relative BACKWARD error ``||Ax-b|| / (||A|| ||x|| + ||b||)`` -- the textbook
#     answer to a scaling problem -- reads 0.4 * eps on the singular plant, the SAME value as on
#     the well-posed sweep. SuperLU is backward stable even when it returns garbage: the garbage
#     exactly solves a nearby system. A backward-error bar would ADMIT the plant.
#   * scaling the bar by a condition estimate (``residual <= k cond eps``) puts the plant at
#     k = 0.49 against k = 0.07..0.22 for the well-posed sweep -- a factor of two, not a
#     separation -- and the E=1e16 row (cond 7.1e17, refinement 7.0e-14) is a WELL-POSED system
#     above 1/eps, so cond is not the discriminator in either direction.
_DIRECT_REFINEMENT_RTOL = 1e-6


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


def _pardiso(matrix, rhs, *, x0=None, M=None):  # noqa: ARG001 -- the direct-shaped signature
    """Intel MKL PARDISO sparse LU -- the multi-threaded direct solver (MATLAB-parity).

    The direct counterpart of scipy's single-threaded SuperLU: PARDISO threads the
    factorization, the same gap laplace has vs MATLAB today (whose sparse LU threads).
    The singular guard rides with the caller's ``_check_finite`` + the relative-residual
    belt (extended to this name in :func:`solve_linear_system`), exactly like ``direct``.
    """
    return _pardiso_spsolve(matrix, rhs)


# name -> LinearSolver. 'direct' is the bit-for-bit default (spsolve); the Krylov peers each reach the
# same solution within _KRYLOV_RTOL. cg/minres want a symmetric/SPD operator (the magnetostatic
# secant stiffness is SPD-ish); gmres/bicgstab/qmr handle general (non-symmetric, e.g. eddy-current).
# 'pardiso' joins when pypardiso is installed (the optional MKL entry above).
LINEAR_SOLVER_REGISTRY: dict[str, LinearSolver] = {
    'direct': _direct,
    'cg': _krylov(cg, symmetric=True),
    'minres': _krylov(minres, symmetric=True),
    'gmres': _krylov(gmres, symmetric=False),
    'bicgstab': _krylov(bicgstab, symmetric=False),
    'qmr': _krylov(qmr, symmetric=False),
}
if _pardiso_spsolve is not None:
    LINEAR_SOLVER_REGISTRY['pardiso'] = _pardiso


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
        hint = " ('pardiso' needs the optional pypardiso package: pip install pypardiso)" if name == 'pardiso' else ''
        msg = f'unknown linear solver {name!r}; known: {sorted(LINEAR_SOLVER_REGISTRY)}{hint}'
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


def direct_relative_residual(matrix, rhs, x) -> float | None:
    """``||A x - b|| / ||b||`` for a direct solve, or ``None`` when the statistic cannot speak.

    ``None`` means ``b == 0``: EVERY null-space vector satisfies ``A x = 0`` exactly, so a
    relative residual is 0/0 there and reporting 0.0 would be a green the arithmetic never
    earned. The caller must treat ``None`` as UNJUDGED, not as a pass.

    NOT cast to ``float``: the time-harmonic operator is COMPLEX, and casting its residual to a
    real dtype is what kept the harmonic call sites from being able to consult this guard at all.
    """
    b = np.asarray(rhs.todense() if sp.issparse(rhs) else rhs)
    b_norm = float(np.linalg.norm(b.ravel()))
    if b_norm == 0.0:
        return None
    xv = np.asarray(x.todense() if sp.issparse(x) else x).reshape(b.shape)
    return float(np.linalg.norm(np.asarray(matrix @ xv).ravel() - b.ravel())) / b_norm


def direct_refinement_correction(matrix, rhs, x) -> float | None:
    """``||d|| / ||x||`` for one step of iterative refinement, or ``None`` when it cannot speak.

    ``d = A^-1 (b - A x)`` estimates the error already present in ``x``, so this is the relative
    amount a second solve would MOVE THE ANSWER -- the quantity the caller actually depends on,
    and (unlike the relative residual) one that does not track the condition number. See
    :data:`_DIRECT_REFINEMENT_RTOL` for the measurements.

    ``None`` is UNJUDGED, never a pass, and it has exactly two causes: the factorization is
    EXACTLY singular, so no correction can be computed at all; or ``matrix`` is not square, so
    there is no factorization to take. ``inf`` is returned when ``x`` is the zero vector and the
    correction is not -- a correction infinitely larger than the answer, which is what that is.

    The factorization is taken fresh here rather than reused from the solve, because the caller
    is ``spsolve``, which discards its factor. That costs a second factorization, and is why this
    is reached ONLY after the residual trigger has already fired -- see
    :func:`reject_unsolved_direct_system`.
    """
    shape = np.shape(matrix)
    if len(shape) != 2 or shape[0] != shape[1]:
        return None
    b = np.asarray(rhs.todense() if sp.issparse(rhs) else rhs)
    xv = np.asarray(x.todense() if sp.issparse(x) else x).reshape(b.shape)
    residual_vec = b.ravel() - np.asarray(matrix @ xv).ravel()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', MatrixRankWarning)
            correction = splu(sp.csc_matrix(matrix)).solve(residual_vec)
    except (RuntimeError, MatrixRankWarning, ValueError):
        return None
    if not np.all(np.isfinite(correction)):
        return None
    d_norm = float(np.linalg.norm(correction))
    x_norm = float(np.linalg.norm(xv.ravel()))
    if x_norm == 0.0:
        return 0.0 if d_norm == 0.0 else float('inf')
    return d_norm / x_norm


def reject_unsolved_direct_system(matrix, rhs, x, backend: str) -> None:
    """Raise :class:`LinearSolveError` when a DIRECT solve did not actually solve its system.

    The half of the singular guard that catches FINITE garbage: SuperLU emits a
    :class:`MatrixRankWarning` on a zero pivot and then returns either NaNs (the belt and
    :func:`_check_finite` catch those) or a finite vector that does not satisfy ``A x = b`` at all.
    Measured on the production CLI at ``lipo_inductor --mesh-size 6mm``: order 157, condition
    estimate 1.5e17, relative residual **1.6e+01**.

    TWO STAGES, because the residual alone cannot tell a failed solve from a merely ill-CONDITIONED
    one -- a residual of 1e-4 at cond 2.3e12 and a residual of 1e-4 at cond 1e2 are opposite
    findings, and a fixed bar reads them the same. So:

    * the residual is the cheap TRIGGER (~0.2-0.6% of the solve it guards, measured on 2D
      Laplacians of order 1.6e3 / 1e4 / 4e4). Below :data:`_DIRECT_RESIDUAL_RTOL` nothing further
      is computed and the cost is exactly what it was;
    * above it, one step of iterative refinement is the VERDICT. It costs a second factorization
      (~100% of the solve, same measurement), paid only on systems that were refused OUTRIGHT
      before this existed -- so no solve that passes today pays anything for it.

    The message names which of the two was established, both statistics, the ORDER, and what the
    caller can do; a refusal that says "singular" about a solution accurate to 2e-17 sends its
    reader to the wrong place.
    """
    residual = direct_relative_residual(matrix, rhs, x)
    if residual is None or residual <= _DIRECT_RESIDUAL_RTOL:
        return
    correction = direct_refinement_correction(matrix, rhs, x)
    if correction is not None and correction <= _DIRECT_REFINEMENT_RTOL:
        return  # ill-CONDITIONED, not unsolved: the residual is cond*eps, the answer is sound.
    n = int(np.shape(matrix)[0])
    if correction is None:
        finding = 'the factorization is EXACTLY singular, so no refinement step exists to judge the answer by'
        measured = f'residual = {residual:.3e}, refinement = unavailable'
    else:
        finding = (
            'one step of iterative refinement moves the solution by '
            f'{correction:.3e} of its own norm, so the returned vector is finite but meaningless'
        )
        measured = f'residual = {residual:.3e}, refinement = {correction:.3e}'
    msg = (
        f'direct solve on an n={n} system by {backend!r} did not solve it: {finding} '
        f'(measured: {measured}; the residual trigger is {_DIRECT_RESIDUAL_RTOL:.0e} and the '
        f'refinement bar is {_DIRECT_REFINEMENT_RTOL:.0e}). The system is singular / '
        'rank-deficient. Check that the model carries a REFERENCE the solve can pin to (a gauge, '
        'a Dirichlet condition, a ground node) and that no unknown is left with a dependent row -- '
        'a DOF orphaned by a degenerate or merged element, a region the mesh let collapse (a '
        'coarser mesh is the usual cause in a FIELD solve), or two ideal sources constraining the '
        'same node pair (the usual cause in a LUMPED one). NOTE a large residual ALONE is '
        'not this finding -- it scales with the condition number, and an ill-conditioned but '
        'well-posed system (e.g. stiffness in SI units) passes here on its refinement step.'
    )
    raise LinearSolveError(msg)


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
    # factorization can return a FINITE vector that does not satisfy A x = b. Reject a large
    # relative residual so such a solve raises rather than reporting garbage as ``status=ok``.
    # Only the direct backends need this (SuperLU and PARDISO alike) -- the Krylov backends
    # already raise on a non-zero convergence ``info`` and asserting their (looser) residual
    # here would clash with their own stop criterion.
    if name in ('direct', 'pardiso'):
        reject_unsolved_direct_system(matrix, rhs, x, name)
    return x


def solve_equilibrated(matrix, rhs):
    """Solve ``matrix @ x = rhs`` after taking the operator's own row scaling out of it.

    SYMMETRIC JACOBI EQUILIBRATION, and it is an ALGEBRAIC IDENTITY: with
    ``D = diag(1/sqrt(|diag A|))``, solving ``(D A D) y = D b`` and returning ``D y`` is the same
    ``x`` in exact arithmetic. It is not a different method, not an approximation and not a
    preconditioner -- the same factorization runs on a system of the same sparsity. Only the
    floating-point path differs, because SuperLU's pivot thresholding compares entries whose
    magnitudes a badly scaled operator has already made incomparable.

    THE CLAIM IS NARROW ON PURPOSE. This is NOT "more accurate in general", and that was measured
    rather than assumed: on a synthetic SPD system with rows scaled across nine decades -- the
    shape a stiffness matrix was ASSUMED to have -- the equilibrated solve came back slightly
    WORSE than the unscaled one (2026-09-06, componentwise 4.6e-7 vs 1.0e-7). The synthetic
    fixture did not reproduce the defect, which is itself the finding: the loss depends on the
    operator, not on a row-scale histogram.

    WHAT IS MEASURED, on the real operator: `cases/study/field/benchmark/rotor_stress`'s
    overspeed/rated peak-hoop ratio, a quantity in which every discretization error cancels by
    construction, so the exact 2.25 is arithmetic and nothing else. Unscaled it held at
    ``n_radial`` 8/16/32 and drifted at 64 -- 2.2338 recorded 2026-08-14, 2.2514 re-measured
    2026-09-06, a spread across runs that is what a conditioning floor looks like. Equilibrated,
    at 8/16/32/64/96:

        2.250000000  2.250000001  2.250000008  2.250000000  2.250000032

    That drift had been recorded as a property of the MESH: `DISTANCE_TO_DONE`'s `rotor_stress`
    row refused a refinement because it "would break the case linearity pin". It was the solve.

    A ROW WITH NO DIAGONAL ENTRY has no scale to take out and passes through at unit scale rather
    than dividing by zero -- an off-diagonal-only row is a legal constraint shape, and turning it
    into ``inf`` here would convert a solvable system into a ``nan`` one further along. The
    magnitude is taken before the ``sqrt`` for the matching reason: an indefinite (saddle-point /
    KKT) operator has negative diagonal entries by construction, and ``sqrt`` of one is ``nan``.

    THE SCALED SOLVE GOES THROUGH THE GUARDED ENTRY POINT, not through `spsolve` directly, and
    that is the 2026-09-07 merge resolution rather than a preference. Both lanes changed this call:
    one wrapped the assembly path in equilibration (the measurement above), the other routed every
    `spsolve` in the tree through :func:`solve_linear_system` for its non-finite check and its
    direct-solve residual rejection -- a singular factorization can return a FINITE vector that
    does not satisfy ``A x = b``, and unguarded that is reported as a successful solve. Taking
    either side alone loses a real property. Composed, the guard runs on the SCALED system, which
    is the stronger place for it: the residual it rejects on is the one the factorization actually
    saw, so a badly scaled operator can no longer pass the check by having its residual measured
    in units the scaling had already inflated.
    """
    a = sp.csc_matrix(matrix)
    diag = np.abs(a.diagonal())
    diag[diag == 0.0] = 1.0
    s = 1.0 / np.sqrt(diag)
    scaling = sp.diags(s)
    y = solve_linear_system((scaling @ a @ scaling).tocsc(), s * np.asarray(rhs))
    return np.asarray(s * y)
