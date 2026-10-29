"""Pluggable INNER linear-solve seam -- one registry of ``solve(matrix, rhs) -> x`` backends.

The solver-methods roadmap (sec 2): today every solve is a direct
sparse factorization (MEC ``_ReluctanceSystem.solve_system`` and FEA ``fea/_solve.py::_solve`` both
call :func:`scipy.sparse.linalg.spsolve`). This module extracts that one call behind a
:data:`LINEAR_SOLVER_REGISTRY` so a solve can pick a KRYLOV iterative solver (needed by JFNK /
newton-krylov and by large 3D / eddy-current systems) without the equation code changing.

First principle (``AGENTS.md``): IMPLEMENT every method and characterize it. The ``direct`` backend is
faer's pattern-cached factorisation when the native wheel provides it and ``spsolve`` VERBATIM when it
does not (see :mod:`~numerics_lab.linear.native_direct` -- the two agree to round-off, not
bit-for-bit, and the SciPy fallback is logged, never silent); the iterative
backends are additive peers, each of which must reach the SAME solution as ``direct`` within its
tolerance (the linear analogue of the nonlinear "converge-to-reference" contract).

The seam is deliberately tiny: a :class:`LinearSolver` is a callable
``(matrix, rhs, *, x0=None) -> x``. A peer :data:`PRECONDITIONER_REGISTRY` builds a
:class:`scipy.sparse.linalg.LinearOperator` approximating ``matrix^{-1}`` (Jacobi diagonal / ILU),
handed to the Krylov backends as ``M``. This module touches no physics and no mesh -- it only solves
``matrix @ x = rhs``.

The seam has TWO SIBLINGS, split out of this file on 2026-09-17 at the 800-line line and
re-exported UNCHANGED, so the surface below is the one this module has always published:
:mod:`~numerics_lab.linear.errors` owns :class:`LinearSolveError`, and
:mod:`~numerics_lab.linear.refinement` owns the direct-solve guard -- the residual TRIGGER,
the refinement VERDICT, and the conditioning ceiling that decides whether a move is this
system's own or one a backend that never answered it produced.
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

from numerics_lab.linear.amg import amg_preconditioner
from numerics_lab.linear.constrained.fixed_values import apply_dirichlet_csr, apply_dirichlet_triplets
from numerics_lab.linear.constrained.null_space_pinning import (
    constraint_rows,
    floats_on_a_constant,
    gauge_pin_floating_constant,
    gauge_pin_orphan_rows,
    orphan_rows,
)
from numerics_lab.linear.constrained.pin_rows import pin_rows

#: ``errors`` (2026-09-17) -- THE SEAM'S OWN ERROR TYPE, and a TRUE LEAF. Both the backends above
#: and the direct-solve guard in ``refinement`` raise it, so it can sit in neither without a cycle;
#: it imports nothing at all, which is what makes it legal here.
from numerics_lab.linear.errors import LinearSolveError
from numerics_lab.linear.ic0 import IC0Operator, ic0_preconditioner, iccg_system
from numerics_lab.linear.native_direct import native_solve

#: ``refinement`` (2026-09-17) -- THE DIRECT-SOLVE GUARD: the residual TRIGGER, the refinement
#: VERDICT, the repair that turns an ill-conditioned-but-SOLVABLE system into its answer, and the
#: conditioning ceiling that separates that from a backend which never answered this system at all.
#:
#: THE THREE PRIVATE CONSTANTS a test reaches for are re-exported with a redundant alias, which is
#: the spelling ``ruff`` reads as a deliberate re-export rather than as an unused import (an
#: ``F401`` suppression would be the alternative, and suppressions may only shrink here).
from numerics_lab.linear.refinement import _DIRECT_REFINEMENT_RTOL as _DIRECT_REFINEMENT_RTOL
from numerics_lab.linear.refinement import _DIRECT_RESIDUAL_RTOL as _DIRECT_RESIDUAL_RTOL
from numerics_lab.linear.refinement import _REPAIRABLE_CONDITION_FACTOR as _REPAIRABLE_CONDITION_FACTOR

#: The PUBLIC half: the same names, signatures and semantics, so every ``from
#: numerics_lab.linear import ...`` in the tree is unaffected.
from numerics_lab.linear.refinement import (
    condition_estimate,
    describe_refinement_trajectory,
    direct_refinement_correction,
    direct_relative_residual,
    move_is_this_systems_conditioning,
    refinement_move_trajectory,
    refinement_residual,
    reject_unsolved_direct_system,
    repair_direct_solution,
    settle_direct_solution,
)
from numerics_lab.linear.ssor import ssor_preconditioner

__all__ = [
    'LINEAR_SOLVER_REGISTRY',
    'PRECONDITIONER_REGISTRY',
    'IC0Operator',
    'LinearSolveError',
    'LinearSolver',
    'apply_dirichlet_csr',
    'apply_dirichlet_triplets',
    'condition_estimate',
    'constraint_rows',
    'describe_refinement_trajectory',
    'direct_refinement_correction',
    'direct_relative_residual',
    'floats_on_a_constant',
    'gauge_pin_floating_constant',
    'gauge_pin_orphan_rows',
    'get_linear_solver',
    'ic0_preconditioner',
    'iccg_system',
    'make_preconditioner',
    'move_is_this_systems_conditioning',
    'orphan_rows',
    'pin_rows',
    'refinement_move_trajectory',
    'refinement_residual',
    'reject_nonfinite_solution',
    'reject_unsolved_direct_system',
    'repair_direct_solution',
    'requires_symmetric',
    'settle_direct_solution',
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


def _direct(matrix, rhs, **_iterative_only):
    """Sparse direct factorization -- faer through :mod:`~numerics_lab.linear.native_direct`, else ``spsolve``.

    THE NATIVE PATH FIRST (2026-09-26): a real sparse operator goes to the pattern-cached faer
    factorisation (supernodal LU, or Cholesky when the values are verified exactly symmetric),
    which measured faster than SuperLU at every production size. It is a DIFFERENT factorisation,
    so it agrees with ``spsolve`` to round-off and not bit-for-bit; both sit behind the same
    residual guard in :func:`solve_linear_system`. When the extension is absent, switched off
    (``NUMERICS_LAB_DIRECT_NATIVE=0``) or the operator is outside its reach (complex, dense), the
    SciPy ``spsolve`` below runs -- logged once and counted in ``native_direct.DIRECT_SOLVES``,
    never silently.

    The SciPy half keeps a SINGULAR guard: SuperLU emits a :class:`MatrixRankWarning` when the factor
    hits a zero/near-zero pivot, then returns EITHER NaNs OR a FINITE-but-meaningless vector. The
    finite case slipped past the caller's finiteness check and shipped as ``status=ok`` garbage. We
    promote that warning to a hard :class:`LinearSolveError` here (belt), and the caller additionally
    rejects a large relative residual (suspenders), so a rank-deficient reluctance network -- e.g. a
    feature-aligned polar-grid loop basis that is not full-rank at some mesh resolution -- fails
    honestly instead of returning a wrong number.
    """
    native = native_solve(matrix, rhs)
    if native is not None:
        return native
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


def _direct_symmetric(matrix, rhs, **_iterative_only):
    """SuperLU in SYMMETRIC mode: a minimum-degree ordering of ``A' + A`` and diagonal pivots.

    For a caller whose operator is SPD BY CONSTRUCTION -- a Galerkin stiffness with no convection,
    no eddy term and no constraint block. ``direct``'s COLAMD orders the COLUMNS alone, which is
    blind to the symmetric pattern: MEASURED 2026-09-25 on case_d's sdm loop matrix
    (n = 89,871, nnz = 826,275) it fills 16.96 M against 9.69 M here, and a factor-and-solve
    takes 1.29 s against 0.64 s, with the answers 1.7e-13 apart (relative). Diagonal pivoting is
    safe only because the operator is SPD, which is why this is a separate NAME a caller chooses
    and not a change to ``direct``. Singular-guarded exactly like ``direct``.

    THE NATIVE PATH FIRST, as ``direct`` takes it: the pattern-cached faer factor, a Cholesky when the
    values are verified exactly symmetric. MEASURED 2026-10-02 on a case_d sdm Newton tangent
    (n = 31,212, nnz = 286,760): 0.07 s against symmetric-mode SuperLU's 0.31 s, the answers 7.8e-14
    apart (relative); SuperLU's factor was 115 s of a two-position sdm sweep's profile. The SuperLU
    half below stays the fallback, counted and warned by :mod:`~numerics_lab.linear.native_direct`.
    """
    native = native_solve(matrix, rhs)
    if native is not None:
        return native
    with warnings.catch_warnings():
        warnings.simplefilter('error', MatrixRankWarning)
        try:
            return splu(
                sp.csc_matrix(matrix),
                permc_spec='MMD_AT_PLUS_A',
                diag_pivot_thresh=0.0,
                options={'SymmetricMode': True},
            ).solve(np.asarray(rhs, dtype=float))
        except (MatrixRankWarning, RuntimeError) as exc:  # SuperLU reports an exact zero pivot as RuntimeError
            msg = f'direct-symmetric solver reports the matrix is (near-)singular ({exc}); is a gauge/reference set?'
            raise LinearSolveError(msg) from exc


def _krylov(fn: Callable):
    """Wrap a scipy Krylov solver as a :class:`LinearSolver`.

    Operator-class requirements live beside the named registry below; the wrapper itself is
    agnostic. Converts the rhs to dense (scipy Krylov wants a 1-D array), forwards ``x0``
    as the warm start and ``M`` as the preconditioner, and raises :class:`LinearSolveError` on a
    non-zero ``info`` (non-convergence within maxiter or a breakdown) rather than returning a wrong x.

    The scipy Krylov signatures are NOT uniform: ``minres`` has no ``atol`` (it takes ``rtol``/
    ``shift``), and ``qmr`` splits the preconditioner into left/right ``M1``/``M2``. Per-solver kwargs
    handle those quirks so every backend shares one call site.
    """

    def _solve(matrix, rhs, *, x0=None, M=None, rtol=_KRYLOV_RTOL, atol=_KRYLOV_ATOL, max_iterations=None):
        dtype = np.result_type(
            matrix.dtype, np.asarray(rhs).dtype, np.float64 if x0 is None else np.asarray(x0).dtype, np.float64
        )
        b = np.asarray(rhs, dtype=dtype).ravel()
        n = b.shape[0]
        kw = {
            'x0': None if x0 is None else np.asarray(x0, dtype=dtype).ravel(),
            'rtol': rtol,
            'maxiter': _KRYLOV_MAXITER_FACTOR * max(n, 1) if max_iterations is None else max_iterations,
        }
        if fn is qmr:
            kw['M1'] = M  # qmr takes a left preconditioner M1 (M2 defaults to identity)
        else:
            kw['M'] = M
        if fn is not minres:
            kw['atol'] = atol  # minres has no atol argument
        if fn is gmres and max_iterations is not None:
            # scipy's gmres maxiter counts RESTART CYCLES unless a 'legacy' callback is present, in
            # which case it caps INNER iterations -- the unit an explicit budget is stated in.
            kw.update(callback=lambda _residual: None, callback_type='legacy')
        x, info = fn(matrix, b, **kw)
        if info != 0:
            kind = 'did not converge' if info > 0 else 'illegal input / breakdown'
            msg = f'Krylov solver {fn.__name__!r} {kind} (info={info}) on an n={n} system'
            raise LinearSolveError(msg)
        return x

    _solve.__name__ = fn.__name__
    return _solve


def requires_symmetric(name: str) -> bool:
    """Whether the named linear backend assumes a symmetric/SPD operator (cg / minres)."""
    LINEAR_SOLVER_REGISTRY[name]
    return name in _SYMMETRIC_SOLVERS


# name -> LinearSolver. 'direct' is the bit-for-bit default (spsolve); the Krylov peers each reach the
# same solution within _KRYLOV_RTOL. cg/minres want a symmetric/SPD operator (the magnetostatic
# secant stiffness is SPD-ish); gmres/bicgstab/qmr handle general (non-symmetric, e.g. eddy-current).
LINEAR_SOLVER_REGISTRY: dict[str, LinearSolver] = {
    'auto': lambda matrix, rhs, **kwargs: solve_linear_system(
        matrix,
        rhs,
        linear_solver='auto',
        x0=kwargs.get('x0'),
        rtol=kwargs.get('rtol', _KRYLOV_RTOL),
        atol=kwargs.get('atol', _KRYLOV_ATOL),
    ),
    'direct': _direct,
    'direct-symmetric': _direct_symmetric,
    'cg': _krylov(cg),
    'iccg': _krylov(cg),
    'minres': _krylov(minres),
    'gmres': _krylov(gmres),
    'bicgstab': _krylov(bicgstab),
    'qmr': _krylov(qmr),
}
_SYMMETRIC_SOLVERS = frozenset({'cg', 'iccg', 'minres'})


def _jacobi_preconditioner(matrix) -> LinearOperator:
    """Diagonal (Jacobi) preconditioner ``M ~ diag(matrix)^{-1}``.

    The cheapest useful preconditioner: scales each row by its diagonal. A zero/NaN diagonal entry
    (a pinned identity row is 1, so this is rare) is left as 1 so ``M`` stays finite.
    """
    diag = np.asarray(matrix.diagonal())
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


# name -> builder(matrix) -> LinearOperator. AMG is a real SPD nodal Galerkin V-cycle
# (not the 3D A-phi electromagnetic decoupling solver). 'ssor' is the nodal
# auxiliary accelerator (symmetric, strengthens CG on the SPD Poisson-type nodal operator); its
# implementation lives in the physics-blind ``gauss.linear.ssor`` module (pure algebra on the
# matrix triangles), imported at module level -- no cycle back into any physics package.
PRECONDITIONER_REGISTRY: dict[str, Callable[[object], LinearOperator]] = {
    'jacobi': _jacobi_preconditioner,
    'ilu': _ilu_preconditioner,
    'ssor': ssor_preconditioner,
    'ic0': ic0_preconditioner,
    'amg': amg_preconditioner,
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


def make_preconditioner(name: str | LinearOperator | None, matrix) -> LinearOperator | None:
    """Build a preconditioner :class:`LinearOperator` for ``matrix``; ``None`` -> no preconditioner."""
    if name is None:
        return None
    if isinstance(name, LinearOperator):
        if name.shape != matrix.shape:
            msg = 'explicit preconditioner shape must match the linear operator'
            raise ValueError(msg)
        if name.dtype is None or np.dtype(name.dtype).kind not in 'fciu':
            msg = 'explicit preconditioner requires a numeric dtype'
            raise ValueError(msg)
        return name
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
    preconditioner: str | LinearOperator | None = None,
    x0: np.ndarray | None = None,
    rtol: float = _KRYLOV_RTOL,
    atol: float = _KRYLOV_ATOL,
    max_iterations: int | None = None,
) -> np.ndarray:
    """Solve ``matrix @ x = rhs`` with the named backend, guarding a non-finite result.

    ``linear_solver=None`` is the default ``direct`` (bit-for-bit ``spsolve``). A Krylov backend may
    take an optional ``preconditioner`` (Jacobi / ILU) and a warm-start ``x0``. The finite-solution
    guard is uniform across backends (the MEC singular-system guard, generalized), so a singular
    system or a Krylov breakdown raises :class:`LinearSolveError` instead of propagating NaNs.
    """
    if not np.isfinite(rtol) or rtol <= 0 or not np.isfinite(atol) or atol < 0:
        msg = 'linear tolerance requires finite rtol > 0 and atol >= 0'
        raise ValueError(msg)
    name = linear_solver or 'direct'
    if max_iterations is not None:
        if (
            isinstance(max_iterations, (bool, np.bool_))
            or not isinstance(max_iterations, (int, np.integer))
            or max_iterations <= 0
        ):
            msg = 'linear iteration budget requires a positive nonboolean integer'
            raise ValueError(msg)
        if name not in ('cg', 'bicgstab', 'gmres', 'minres', 'qmr', 'iccg'):
            msg = 'linear iteration budget requires an explicit iterative backend'
            raise ValueError(msg)
    # A built IC0 operator (one per J11, shared by its columns) -- built on the LIFTED operator.
    prebuilt = preconditioner if isinstance(preconditioner, IC0Operator) else None
    if name in ('auto', 'iccg'):
        matrix, rhs, x0 = iccg_system(matrix, rhs, x0)
    M = None
    if name == 'auto':
        # Select by admissibility, not an unmeasured size/speed crossover.
        # IC0 breakdown or iterative failure falls back to the guarded direct solve.
        name = 'direct'
        try:
            M = prebuilt if prebuilt is not None else make_preconditioner('ic0', matrix)
            name = 'iccg'
        except ValueError:
            pass
    if name == 'iccg' and prebuilt is None and preconditioner not in (None, 'ic0'):
        msg = 'ICCG requires the IC0 preconditioner'
        raise ValueError(msg)
    backend = get_linear_solver(name)
    if M is None and name not in ('direct', 'direct-symmetric'):
        M = (
            prebuilt
            if prebuilt is not None
            else make_preconditioner('ic0' if name == 'iccg' else preconditioner, matrix)
        )
    try:
        tolerances = {} if name in ('direct', 'direct-symmetric') else {'rtol': rtol, 'atol': atol}
        if max_iterations is not None:
            tolerances['max_iterations'] = int(max_iterations)
        x = reject_nonfinite_solution(np.asarray(backend(matrix, rhs, x0=x0, M=M, **tolerances)), name)
    except LinearSolveError:
        if linear_solver != 'auto' or name == 'direct':
            raise
        return solve_linear_system(matrix, rhs, linear_solver='direct')
    # Direct-solve residual guard (suspenders to _direct's MatrixRankWarning belt): a singular
    # factorization can return a FINITE vector that does not satisfy A x = b. Reject a large
    # relative residual so such a solve raises rather than reporting garbage as ``status=ok``.
    # Only the direct backends need this (SuperLU and PARDISO alike) -- the Krylov backends
    # already raise on a non-zero convergence ``info`` and asserting their (looser) residual
    # here would clash with their own stop criterion.
    if name in ('direct', 'direct-symmetric'):
        # REPAIR, THEN JUDGE, both against the same unchanged bar. `repair_direct_solution` returns
        # `x` untouched unless the residual trigger fires, so this is free for every solve that
        # passes today; when it does fire, the steps it takes are the ones the verdict below was
        # going to measure anyway.
        x = settle_direct_solution(matrix, rhs, x, name)
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

    WHAT IS MEASURED, on the real operator: `case_k`'s
    overspeed/rated peak-hoop ratio, a quantity in which every discretization error cancels by
    construction, so the exact 2.25 is arithmetic and nothing else. Unscaled it held at
    ``n_radial`` 8/16/32 and drifted at 64 -- 2.2338 recorded 2026-08-14, 2.2514 re-measured
    2026-09-06, a spread across runs that is what a conditioning floor looks like. Equilibrated,
    at 8/16/32/64/96:

        2.250000000  2.250000001  2.250000008  2.250000000  2.250000032

    That drift had been recorded as a property of the MESH: `DISTANCE_TO_DONE`'s `case_k`
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
