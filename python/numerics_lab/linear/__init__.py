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

import math
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
from motronics.gauss.linear.gauge import (
    constraint_rows,
    floats_on_a_constant,
    gauge_pin_floating_constant,
    gauge_pin_orphan_rows,
    orphan_rows,
)
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
    'constraint_rows',
    'describe_refinement_trajectory',
    'direct_refinement_correction',
    'direct_relative_residual',
    'floats_on_a_constant',
    'gauge_pin_floating_constant',
    'gauge_pin_orphan_rows',
    'get_linear_solver',
    'make_preconditioner',
    'orphan_rows',
    'pin_rows',
    'refinement_move_trajectory',
    'refinement_residual',
    'reject_nonfinite_solution',
    'reject_unsolved_direct_system',
    'repair_direct_solution',
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

# Stage 2b -- THE THIRD CATEGORY, and the guard above could not see it because it takes exactly
# ONE step. A system can be ill-conditioned enough that one step moves the answer past the bar and
# still be perfectly SOLVABLE: the step sequence collapses. Measured 2026-09-07 on the n=864
# tied-contact KKT of `physics/structural/contact.py`, the system that found this:
#
#     step 0: ||d||/||x|| = 2.5893e-05     <- past the 1e-6 bar, so the guard refused outright
#     step 1: ||d||/||x|| = 6.3701e-12
#     step 2: ||d||/||x|| = 5.4032e-15
#
# and the answer MOVED by 9.493e-05 of itself, so the direct vector really was wrong -- the guard
# was right to refuse it and wrong about why. Its message said "finite but meaningless ... the
# system is singular / rank-deficient", which sends a reader looking for a missing gauge on a
# system that has none missing. Two cheap steps produce the true solution.
#
# WHY THE RESIDUAL CANNOT ARBITRATE THIS: ||Ax-b||_inf sits at ~3e-09 before AND after the repair,
# because ||A||_inf is ~9.6e+11 and round-off times that is the floor. The refinement move is the
# only statistic that separates the three cases -- the argument the block above already makes for
# stage 2; this only applies it more than once.
#
# THIS LOOSENS NOTHING. The acceptance bar is still `_DIRECT_REFINEMENT_RTOL`, unchanged, and it
# is now demanded of the vector the caller actually receives. A system that passed before passes
# unchanged and pays nothing (the residual trigger gates all of it). A system that was REFUSED
# before either becomes correct -- proven by reaching the same bar -- or is refused still, with
# the step count in the message as evidence that repair was tried.
_DIRECT_REFINEMENT_MAX_STEPS = 4

# THE CEILING ON THE REPAIR, and an escape hatch needs one rather than just a reason
# (`.claude/rules/taste.md`). Refinement will drive ANY starting vector to the solution of a
# solvable system, including one the backend never derived from it -- so without a ceiling the
# repair above silently corrects a BROKEN BACKEND instead of reporting it, which is the same
# family as a warning on an unchanged success return, minus the warning.
#
# THE RATIO IS WHAT PINS IT, never the value at one point. Measured 2026-09-07:
#
#     the real conditioning artefact  (n=864 tied-contact KKT)       first move = 2.5893e-05
#     a backend returning garbage     (`np.ones_like(rhs)` on SPD)   first move = 9.7219e-01
#
# a separation of 3.76e+04 -- 4.6 decades -- and `test_high_residual_direct_solve_is_rejected`
# PLANTS the second one, which is what caught the uncapped first draft. 1e-3 sits ~1.6 decades
# above the artefact and ~3 below the plant, near the geometric mean (5.0e-3). Above this the
# starting vector is not an ANSWER that conditioning spoiled; it is unrelated to the system, and
# the refusal is the correct output.
_DIRECT_REPAIRABLE_MAX = 1e-3


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


def reject_nonfinite_solution(x: np.ndarray, backend: str) -> np.ndarray:
    """Raise :class:`LinearSolveError` when a solve returned inf/NaN; return ``x`` unchanged otherwise.

    PUBLIC since 2026-09-06, and the rename is the point. It was `_check_finite`, private to this
    module and reached from exactly one line -- so the cheapest half of the singular guard was
    unavailable to `gauss.time_integrators.factor_reuse`, whose cache HIT path returned
    back-substitution output entirely unjudged. Its expensive sibling
    :func:`reject_unsolved_direct_system` needs the MATRIX and a hit deliberately does not build
    one; this needs only the answer.

    THE PRICE, as a RATIO rather than a value: ``O(n)`` against a back-substitution the caller has
    already paid at ``O(n^2)`` dense / ``O(nnz)`` sparse. That ratio is why this guard is
    affordable exactly where the residual one is not.
    """
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
    The singular guard rides with the caller's :func:`reject_nonfinite_solution` + the relative-residual
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


#: Dekker's splitting constant, ``2**27 + 1``. Multiplying a float64 by it and taking
#: ``c - (c - a)`` yields the leading 26 bits of ``a`` EXACTLY, so ``a = hi + lo`` with both halves
#: representable and their product exact -- the standard error-free transformation, and the whole
#: of what :func:`refinement_residual` needs beyond ``math.fsum``.
_DEKKER_SPLIT = 134217729.0


def refinement_residual(matrix, rhs, x) -> np.ndarray | None:
    """``b - A x`` computed to DOUBLE the working precision, or ``None`` when it cannot be.

    WHY THIS EXISTS, measured 2026-09-16 on `spm_ferrari_luce_rear --scope full --chart
    cartesian_3d --select fea,tet` at 3.0 mm: an n=207104 H(curl) solve was refused as "singular /
    rank-deficient" on a refinement move of **1.635e-06** -- 1.6x the bar, against the committed
    singular plant's 4.7 and the well-posed sweep's 1e-13 (see :data:`_DIRECT_REFINEMENT_RTOL`).
    A system that is genuinely rank-deficient does not land at 1.6x the bar. What it had was a
    refinement sequence that STAGNATED instead of collapsing, and the stagnation was in the
    INSTRUMENT rather than in the answer.

    ``d = A^-1 (b - A x)`` is only as good as ``b - A x``, and a working-precision residual of an
    ``x`` that already solves the system to round-off is CANCELLATION: the computed ``b - A x``
    carries an error of order ``eps * ||A|| ||x||``, so ``||d|| / ||x||`` cannot fall below
    ``cond(A) * eps`` however many steps are taken. That floor is a property of the ESTIMATOR, not
    of ``x``. Measured on a synthetic SPD system at ``cond = 1.0e10``:

        working-precision residual:  1.924e-07, 2.248e-07, 2.505e-07, 2.315e-07, 2.234e-07
        this function's residual:    1.216e-07, 5.383e-15, 4.398e-17, 4.398e-17, 4.398e-17

    -- the same first step (both estimate the same real error in ``x``), and then one collapses
    and the other does not. THE BAR IS UNTOUCHED, and that is the point: the guard still demands
    1e-06 of the vector it returns. What changed is that the number it reads is now the move the
    answer would actually make rather than the noise floor of its own subtraction.

    THE CONSTRUCTION, and it is exact rather than merely careful. Each product ``a_ij x_j`` is
    split by :data:`_DEKKER_SPLIT` into a rounded value plus its EXACT error term, and the row is
    then summed together with ``-b_i`` by :func:`math.fsum`, which is correctly rounded over any
    number of terms. So the returned residual is the correctly rounded value of an exactly
    computed quantity: double-double in, one float64 out.

    ``None`` is UNJUDGED and means the caller must fall back to the plain residual: a complex or
    non-float64 operator (the error-free transformation is stated for binary64 reals), or a
    non-square/unshaped matrix. Returning a silently plain residual under this name would be the
    declaration-that-lies shape, so the fallback is the CALLER's, visibly.

    THE PRICE, as a ratio: measured 1.35 s for the fsum pass at n=207104 / nnz=3.1e6 against
    0.014 s for the working-precision matvec -- ~100x the matvec and ~1% of the sparse
    factorization the same step already pays. It is reached only after the residual trigger has
    fired, i.e. only on solves that were being REFUSED.
    """
    shape = np.shape(matrix)
    if len(shape) != 2 or shape[0] != shape[1]:
        return None
    b = np.asarray(rhs.todense() if sp.issparse(rhs) else rhs).ravel()
    xv = np.asarray(x.todense() if sp.issparse(x) else x).ravel()
    if b.dtype != np.float64 or xv.dtype != np.float64:
        return None
    # A NON-FINITE ITERATE CANNOT BE JUDGED HERE, and saying so is what keeps the SINGULAR
    # refusal intact. An exactly singular solve hands back inf/NaN, and the error-free
    # transformation below is stated for FINITE binary64: `inf - inf` in the Dekker split is NaN
    # and `math.fsum` RAISES `-inf + inf in fsum` on the mixed signs that follow -- so without
    # this the compensated path turns a clean `LinearSolveError` into a ValueError from inside
    # the estimator, and a caller that was refusing a singular system starts crashing on it.
    # `None` is the channel that already exists for "cannot speak", and the caller refuses on it.
    if not (np.all(np.isfinite(b)) and np.all(np.isfinite(xv))):
        return None
    csr = sp.csr_matrix(matrix)
    if csr.dtype != np.float64:
        return None

    data, indices, indptr = csr.data, csr.indices, csr.indptr
    factor = xv[indices]
    product = data * factor
    # The error-free transformation, vectorized over every nonzero at once: `product` is the
    # rounded product and `error` is EXACTLY what the rounding discarded, so their sum is the
    # product with no error at all.
    split = _DEKKER_SPLIT * data
    data_hi = split - (split - data)
    data_lo = data - data_hi
    split = _DEKKER_SPLIT * factor
    factor_hi = split - (split - factor)
    factor_lo = factor - factor_hi
    error = ((data_hi * factor_hi - product) + data_hi * factor_lo + data_lo * factor_hi) + data_lo * factor_lo

    out = np.empty(csr.shape[0], dtype=np.float64)
    for row in range(csr.shape[0]):
        lo, hi = indptr[row], indptr[row + 1]
        # ONE fsum over products, their errors AND the right-hand side, rather than three sums
        # added together: adding two correctly rounded partial sums re-introduces exactly the
        # cancellation this function exists to remove.
        out[row] = -math.fsum([*product[lo:hi], *error[lo:hi], -b[row]])
    return out


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
    # THE RESIDUAL IN DOUBLE THE WORKING PRECISION, with the plain one as the declared fallback:
    # `refinement_residual` returns None exactly when the error-free transformation does not hold
    # (complex or non-float64), and the estimate below then means what it always meant. See that
    # function for the measurement that made the difference matter.
    residual_vec = refinement_residual(matrix, b.ravel(), xv.ravel())
    if residual_vec is None:
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


def repair_direct_solution(matrix, rhs, x) -> tuple[np.ndarray, tuple[float, ...]]:
    """``x`` refined until the refinement move reaches the bar, plus the move sequence.

    THE REPAIR THE GUARD'S OWN METRIC IMPLIES. :func:`reject_unsolved_direct_system` measures how
    far one step of iterative refinement would move the answer and refuses when that exceeds
    :data:`_DIRECT_REFINEMENT_RTOL`. For a genuinely singular system the moves do not shrink; for
    an ill-conditioned but solvable one they collapse in one or two steps (see
    :data:`_DIRECT_REFINEMENT_MAX_STEPS` for the measurement that found this). Taking the steps
    turns the second case into the answer instead of a refusal.

    IT ONLY EVER RESCUES A REFUSAL, and that is the property that makes it safe in a solver the
    whole tree shares. THREE early returns leave ``x`` BIT-IDENTICAL:

    * below :data:`_DIRECT_RESIDUAL_RTOL` nothing is even factorized -- only the residual the
      guard was going to take anyway;
    * above it, whenever the FIRST move is already inside :data:`_DIRECT_REFINEMENT_RTOL`, which
      is the ill-conditioned-but-sound case the guard accepts;
    * and whenever that move exceeds :data:`_DIRECT_REPAIRABLE_MAX`, i.e. the vector is not an
      answer this system's conditioning spoiled but one unrelated to it -- a broken backend, which
      must be reported rather than corrected.

    So the set of solves whose answer this can change is a SUBSET of the set that used to raise,
    and it is bounded at both ends by measurements rather than by a preference.

    ONE FACTORIZATION FOR ALL THE STEPS, unlike :func:`direct_refinement_correction`, which takes
    a fresh one because it is handed a vector rather than a factor. That is what makes the repair
    cheaper than the diagnosis it replaces.

    Returns ``(x, moves)``. ``moves`` is empty when the trigger never fired, and otherwise holds
    one relative move per step attempted -- the caller passes its LAST element to the guard so the
    verdict is taken on the returned vector without a further factorization. A non-square matrix,
    an exactly singular factorization, or a non-finite step returns ``x`` as it came: unjudged is
    not a pass, and the guard is what refuses.
    """
    residual = direct_relative_residual(matrix, rhs, x)
    if residual is None or residual <= _DIRECT_RESIDUAL_RTOL:
        return np.asarray(x), ()
    shape = np.shape(matrix)
    if len(shape) != 2 or shape[0] != shape[1]:
        return np.asarray(x), ()
    b = np.asarray(rhs.todense() if sp.issparse(rhs) else rhs).ravel()
    original = np.asarray(x.todense() if sp.issparse(x) else x).ravel()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', MatrixRankWarning)
            factor = splu(sp.csc_matrix(matrix))
    except (RuntimeError, MatrixRankWarning, ValueError):
        return np.asarray(x), ()

    def _step_and_move(vec):
        # The SAME estimator `direct_refinement_correction` uses, and it has to be: the repair
        # steps and the verdict taken on their result must read one instrument, or a vector could
        # be repaired against one residual and judged against another.
        residual_vec = refinement_residual(matrix, b, vec)
        if residual_vec is None:
            residual_vec = b - np.asarray(matrix @ vec).ravel()
        step = factor.solve(residual_vec)
        if not np.all(np.isfinite(step)):
            return None, None
        norm = float(np.linalg.norm(vec))
        return step, (float(np.linalg.norm(step)) / norm if norm else np.inf)

    step, move = _step_and_move(original)
    if step is None:
        return np.asarray(x), ()
    # ILL-CONDITIONED BUT SOUND STAYS BIT-IDENTICAL, and this early return is the whole reason the
    # change is safe to make in a shared solver. The guard ACCEPTS this category already (a large
    # residual with a tiny refinement move is cond*eps, not a failed solve), so refining it would
    # move an answer that was never in question -- which is exactly what the first draft of this
    # function did, and `tests/unit/gauss/test_nonlinear_port_golden.py`'s frozen trajectories
    # moved with it. The move is still REPORTED, so the verdict below costs no factorization.
    # ONE DECISION WITH A WINDOW, not two: repair only a move that is BOTH past the acceptance bar
    # and small enough to be this system's conditioning. Below `_DIRECT_REFINEMENT_RTOL` the guard
    # already accepts the answer and refining it would move a number never in question. Above
    # `_DIRECT_REPAIRABLE_MAX` the vector is not an answer conditioning spoiled but one unrelated
    # to the system -- a broken backend, which must be reported rather than corrected. Outside the
    # window `x` goes back bit-identical and the guard's verdict is exactly what it always was.
    if move <= _DIRECT_REFINEMENT_RTOL or not move <= _DIRECT_REPAIRABLE_MAX:
        return np.asarray(x), (move,)

    moves = [move]
    current = original + step
    for _extra in range(_DIRECT_REFINEMENT_MAX_STEPS - 1):
        if moves[-1] <= _DIRECT_REFINEMENT_RTOL:
            break
        step, move = _step_and_move(current)
        if step is None:
            break
        moves.append(move)
        current = current + step
    return current, tuple(moves)


#: How far a trajectory must fall for :func:`refinement_move_trajectory` to be called COLLAPSING.
#: Six decades, and it is a RATIO between two points of the same run rather than a value at one --
#: so it carries no unit, no scale and no dependence on the order of the system. The two plants in
#: `test_the_refusal_says_whether_refinement_collapses_or_stalls.py` sit on either side of it by
#: many decades; it is not a knob tuned to squeeze one of them across.
_REFINEMENT_COLLAPSE_RATIO = 1e-6


def refinement_move_trajectory(matrix, rhs, x, *, max_steps: int = _DIRECT_REFINEMENT_MAX_STEPS) -> tuple[float, ...]:
    """The sequence of relative refinement moves from ``x`` -- EVIDENCE, never a repaired vector.

    :func:`repair_direct_solution` stops at :data:`_DIRECT_REPAIRABLE_MAX` and hands the guard a
    single number, so a refusal above that ceiling cannot say which of two things it saw: a
    solvable system whose starting vector is merely further out than the ceiling (the moves
    COLLAPSE, because refinement converges to the one answer), or a genuinely rank-deficient one
    (the moves STALL, because there is no unique answer to converge to). Both used to be reported
    as "singular / rank-deficient", which sends the reader of the second message looking for a
    missing gauge on a system that has none missing.

    IT CANNOT CHANGE A VERDICT, by construction: it returns floats and no vector, and its only
    caller is :func:`reject_unsolved_direct_system` on the path that is already raising. So no bar
    moves and nothing that is refused today becomes accepted -- the refusal simply names what it
    measured. It is priced at one factorization plus ``max_steps`` triangular solves, paid ONLY by
    a solve that is about to fail, which is the one place that cost is already lost.

    Returns the moves in order, or ``()`` when no step could be taken (a non-square matrix, an
    exactly singular factorization, or a non-finite step): unmeasured is reported as unmeasured.
    """
    shape = np.shape(matrix)
    if len(shape) != 2 or shape[0] != shape[1]:
        return ()
    b = np.asarray(rhs.todense() if sp.issparse(rhs) else rhs).ravel()
    current = np.asarray(x.todense() if sp.issparse(x) else x).ravel().astype(float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', MatrixRankWarning)
            factor = splu(sp.csc_matrix(matrix))
    except (RuntimeError, MatrixRankWarning, ValueError):
        return ()
    moves: list[float] = []
    for _step in range(int(max_steps)):
        residual_vec = refinement_residual(matrix, b, current)
        if residual_vec is None:
            residual_vec = b - np.asarray(matrix @ current).ravel()
        step = factor.solve(residual_vec)
        if not np.all(np.isfinite(step)):
            break
        norm = float(np.linalg.norm(current))
        moves.append(float(np.linalg.norm(step)) / norm if norm else np.inf)
        current = current + step
    return tuple(moves)


def describe_refinement_trajectory(moves: tuple[float, ...]) -> str:
    """One clause naming what *moves* shows -- COLLAPSES, STALLS, or was not measurable."""
    if not moves:
        return 'diagnostic refinement trajectory: unavailable (no step could be taken)'
    shown = ', '.join(f'{m:.3e}' for m in moves)
    if len(moves) < 2:
        verdict = 'only ONE step was available, so it says nothing about convergence'
    elif moves[-1] <= moves[0] * _REFINEMENT_COLLAPSE_RATIO:
        verdict = (
            'the trajectory COLLAPSES, so this system is SOLVABLE and the starting vector was '
            f'merely further out than the repair ceiling {_DIRECT_REPAIRABLE_MAX:.0e} -- look at '
            'CONDITIONING and at the backend, not for a missing gauge'
        )
    else:
        verdict = (
            'the trajectory STALLS, so refinement has no unique answer to converge to -- this is '
            'a genuine rank deficiency and the gauge/reference advice above applies'
        )
    return f'diagnostic refinement trajectory [{shown}]: {verdict}'


def reject_unsolved_direct_system(matrix, rhs, x, backend: str, *, refinement: float | None = None) -> None:
    """Raise :class:`LinearSolveError` when a DIRECT solve did not actually solve its system.

    The half of the singular guard that catches FINITE garbage: SuperLU emits a
    :class:`MatrixRankWarning` on a zero pivot and then returns either NaNs (the belt and
    :func:`reject_nonfinite_solution` catch those) or a finite vector that does not satisfy ``A x = b``.
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
    # `refinement` IS A MEASUREMENT ALREADY TAKEN, passed in by `repair_direct_solution` so that
    # judging a repaired vector does not pay for a third factorization. It is not a way to supply
    # a friendlier number: the value handed in is the LAST move of the repair, i.e. the one
    # measured on the vector being judged here, and omitting it falls back to measuring it.
    correction = refinement if refinement is not None else direct_refinement_correction(matrix, rhs, x)
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
    # AND WHICH OF THE TWO IT ACTUALLY IS. The sentence above is the right advice for a rank
    # deficiency and the wrong advice for a solvable system whose starting vector landed past
    # `_DIRECT_REPAIRABLE_MAX` -- a ceiling measured on an n=864 system, which a field solve is
    # under no obligation to stay below. The trajectory separates them and costs one factorization
    # on a solve that is already lost. It changes no verdict: this line is inside the raise.
    msg = f'{msg} {describe_refinement_trajectory(refinement_move_trajectory(matrix, rhs, x))}'
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
    x = reject_nonfinite_solution(np.asarray(backend(matrix, rhs, x0=x0, M=M)), name)
    # Direct-solve residual guard (suspenders to _direct's MatrixRankWarning belt): a singular
    # factorization can return a FINITE vector that does not satisfy A x = b. Reject a large
    # relative residual so such a solve raises rather than reporting garbage as ``status=ok``.
    # Only the direct backends need this (SuperLU and PARDISO alike) -- the Krylov backends
    # already raise on a non-zero convergence ``info`` and asserting their (looser) residual
    # here would clash with their own stop criterion.
    if name in ('direct', 'pardiso'):
        # REPAIR, THEN JUDGE, both against the same unchanged bar. `repair_direct_solution` returns
        # `x` untouched unless the residual trigger fires, so this is free for every solve that
        # passes today; when it does fire, the steps it takes are the ones the verdict below was
        # going to measure anyway.
        x, moves = repair_direct_solution(matrix, rhs, x)
        reject_unsolved_direct_system(matrix, rhs, x, name, refinement=moves[-1] if moves else None)
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

    AND ON A REAL OPERATOR IT LOSES BADLY, which is the sharper half and was found by
    over-applying this function. 2026-09-07, on the assembled tied-contact KKT of
    `physics/structural/contact.py`: ||Ax-b||_inf went 5.24e-10 -> 1.47e-09 (relative 2.62e-05 ->
    7.35e-05) and the solution vector moved 9.09e-07, which is not round-off. Two displacement
    comparisons in that physics' unit file reddened and were right to. The operator is an
    INDEFINITE saddle point with a zero block -- the very shape the `|diag|`-before-`sqrt` and
    the no-diagonal-row passthrough below make LEGAL, and legal is not the same as better. That
    site now takes :func:`solve_linear_system` instead.

    Its sibling `physics/structural/harmonic.py` measured the OPPOSITE on the same day
    (||Ax-b||_inf 3.48e-09 -> 3.12e-09, answer unchanged to 4e-11 relative) and keeps the
    scaling. So the reading to take from all three measurements is: THE GUARD IS THE
    REQUIREMENT, THE SCALING IS A PER-SITE CLAIM. `solve_linear_system` is what a caller owes;
    reaching for this one is a statement about a particular operator and has to be measured on
    THAT operator, never inherited from a neighbour.

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
