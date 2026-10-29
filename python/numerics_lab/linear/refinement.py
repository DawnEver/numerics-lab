"""The DIRECT-SOLVE GUARD -- when a direct solve's answer can be trusted, and how it is refused.

Split out of :mod:`numerics_lab.linear` on 2026-09-17 at the 800-line line. The seam itself did
not move: :func:`~numerics_lab.linear.solve_linear_system` is still the entry point, the
registry still chooses the backend, and every name below is re-exported from the package head
UNCHANGED, so no call site was touched.

WHAT THIS HALF ANSWERS, and it is not "did the solve fail". ``||A x - b|| / ||b||`` is NOT a
failure statistic -- it tracks the CONDITION NUMBER of a perfectly good solve -- so a fixed bar on
it is a disguised condition test: the same answer to 2e-17 at cond 1.4e+01 and 2.3e+12, with the
residual moving as ``cond * eps``. The residual is therefore only the cheap TRIGGER
(:data:`_DIRECT_RESIDUAL_RTOL`), and what ANSWERS it is one step of iterative refinement, whose
relative move ``||A^-1 (b - A x)|| / ||x||`` estimates the error ALREADY IN the answer
(:data:`_DIRECT_REFINEMENT_RTOL`). Three categories come out of that pair, and each has its own
constant and its own paragraph beside it:

* a system that PASSES -- the move is already inside the acceptance bar;
* a system that is ill-CONDITIONED but SOLVABLE -- the move collapses in a step or two, and
  :func:`repair_direct_solution` takes those steps (bounded by
  :data:`_DIRECT_REFINEMENT_MAX_STEPS`) instead of refusing the answer;
* a vector that is not an answer THIS system spoiled -- a broken backend. It collapses exactly
  like the second category, which is why the magnitude gate
  (:func:`move_is_this_systems_conditioning`, :data:`_REPAIRABLE_CONDITION_FACTOR`) exists, and
  why neither statistic subsumes the other.

Every constant here is pinned to a MEASUREMENT quoted beside it, and two candidate criteria that
were measured and then REFUTED are recorded with their numbers so nobody re-derives them.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, MatrixRankWarning, onenormest, splu

from numerics_lab.linear.errors import LinearSolveError

__all__ = [
    'condition_estimate',
    'describe_refinement_trajectory',
    'direct_refinement_correction',
    'direct_relative_residual',
    'move_is_this_systems_conditioning',
    'refine_factored_solution',
    'refinement_move_trajectory',
    'refinement_residual',
    'reject_unsolved_direct_system',
    'repair_direct_solution',
    'settle_direct_solution',
]


def refine_factored_solution(matrix, rhs, initial, solve_correction):
    """Refine original rows with an owned factor; retain the best backward error."""
    matrix = sp.csr_matrix(matrix)
    rhs, best = np.asarray(rhs, dtype=float), np.array(initial, dtype=float, copy=True)
    if (
        matrix.shape != (best.size, best.size)
        or best.ndim != 1
        or rhs.shape != best.shape
        or not np.isfinite(matrix.data).all()
    ):
        msg = 'factored refinement requires finite square operators and aligned vectors'
        raise LinearSolveError(msg)

    def measured(value):
        defect = rhs - matrix @ value
        magnitude = abs(matrix) @ abs(value) + abs(rhs)
        if not all(np.isfinite(item).all() for item in (value, defect, magnitude)):
            msg = 'factored refinement requires finite original row operands'
            raise LinearSolveError(msg)
        ratio = np.divide(abs(defect), magnitude, out=np.zeros_like(defect), where=magnitude != 0)
        ratio[(magnitude == 0) & (defect != 0)] = np.inf
        return defect, float(np.max(ratio, initial=0))

    defect, score = measured(best)
    for _ in range(3):
        # A defect below the smallest NORMAL float has no representable correction: a row-scaled
        # solve underflows it to zero and refuses. The refined value is final there.
        if not np.any(abs(defect) >= np.finfo(float).tiny):
            break
        correction = np.asarray(solve_correction(defect), dtype=float)
        if correction.shape != best.shape:
            msg = 'factored refinement correction must retain actual coordinate shape'
            raise LinearSolveError(msg)
        candidate = best + correction
        next_defect, next_score = measured(candidate)
        if next_score >= score:
            break
        best, defect, score = candidate, next_defect, next_score
    move = None
    relative = direct_relative_residual(matrix, rhs, best)
    if relative is not None and relative > _DIRECT_RESIDUAL_RTOL:
        # The owned factor may use actual mixed-unit minors or row normalization.
        # Refactoring the unscaled operator would judge a different numerical solve.
        compensated = refinement_residual(matrix, rhs, best)
        forcing = defect if compensated is None else compensated
        correction = np.asarray(solve_correction(forcing), dtype=float)
        if correction.shape != best.shape or not np.isfinite(correction).all():
            msg = 'factored refinement correction must retain finite actual coordinate shape'
            raise LinearSolveError(msg)
        # A callback cannot certify itself by returning a small or zero vector.
        # First retain the original equation guard for A*correction = forcing.
        reject_unsolved_direct_system(matrix, forcing, correction, 'factored_refinement_correction')
        correction_norm, solution_norm = float(np.linalg.norm(correction)), float(np.linalg.norm(best))
        move = (
            correction_norm / solution_norm if solution_norm != 0 else (0.0 if correction_norm == 0 else float('inf'))
        )
    reject_unsolved_direct_system(matrix, rhs, best, 'factored_refinement', refinement=move)
    return best


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
# with the same sweep plus the committed order-157 ``case_i`` finite-garbage plant:
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
# TWO STATISTICS, AND NEITHER SUBSUMES THE OTHER. This is the distinction the constant that stood
# here used to conflate, which is why its refusal said "singular / rank-deficient" about two
# different animals:
#
#     the TRAJECTORY  (does it collapse?)  answers   does the SYSTEM have a unique answer?
#     the MAGNITUDE   (how big is move 0?) answers   did the BACKEND answer THIS system?
#
# A broken backend on a well-posed system COLLAPSES -- measured, `np.ones_like(rhs)` on an n=8 SPD
# plant gives [9.722e-01, 2.062e-15, ...], fifteen decades -- so collapse alone would repair
# garbage into the right answer. A rank-deficient system STALLS whatever its magnitude. Both gates
# are required and each catches what the other cannot.
#
# WHY THE MAGNITUDE GATE IS NO LONGER AN ABSOLUTE NUMBER. It stood at `1e-3`, calibrated
# 2026-09-07 on exactly two points:
#
#     the real conditioning artefact  (n=864 tied-contact KKT)       first move = 2.5893e-05
#     a backend returning garbage     (`np.ones_like(rhs)` on SPD)   first move = 9.7219e-01
#
# Neither point carries n, and the first refinement move goes as cond*eps -- so a PHYSICALLY SOUND
# solve crosses the ceiling simply by getting bigger, and the larger the problem the more certain
# the false refusal. Measured 2026-09-17 on `case_c --scope full --set
# solve.chart=cartesian_3d --select fea,tet --set mesh.mesh_size="5 mm"`: n = 123,233, a move of 1.434e-03 that is
# 1.43x past the old ceiling, and a trajectory [1.434e-03, 2.062e-06, 2.961e-09, 4.252e-12] that
# collapses nine decades to three orders INSIDE `_DIRECT_REFINEMENT_RTOL`. The solve was always
# solvable; the instrument said otherwise, and it said so more confidently the bigger n grew.
# A 140x growth in n was enough to break it.
#
# So the ceiling is now pinned to the system's OWN conditioning -- `move <= C * cond_1(A) * eps`,
# a dimensionless ratio rather than a value at one operating point, which is what `taste.md` asks
# for. See :data:`_REPAIRABLE_CONDITION_FACTOR` for how C itself is pinned.
#: The safety factor C in ``move[0] <= C * cond_1(A) * eps``: how many times its own conditioning
#: floor a starting vector may sit above and still be treated as an ANSWER this system spoiled
#: rather than a vector unrelated to it.
#:
#: PINNED AS A RATIO BETWEEN TWO MEASURED OPERATING POINTS, never chosen. Both are live plants:
#:
#:     np.ones_like(rhs) on a well-posed n=8 SPD    R = move0 / (cond_1 * eps) = 5.91e+14
#:     the n=123,233 `case_c` ring   R = 4.09e-07 to 9.38e-07 (three Newton solves)
#:
#: C is their geometric mean, which is how the constant it replaces was placed between ITS two
#: points. Margin: 1.07e+08 on the repair side (C / 9.38e-07) and 5.91e+12 on the refusal side (5.91e+14 / C).
#:
#: THE ESTIMATE LEANS SAFE, and a reader checking that should not have to derive it.
#: :func:`condition_estimate` uses `onenormest`, which returns a LOWER bound on the 1-norm, so
#: `cond_1` comes out UNDER-estimated (measured 0.85x and 0.95x of the dense truth on n=8 and
#: n=200 plants). An underestimated `cond` LOWERS the ceiling and makes this gate STRICTER: the
#: error direction can only cause a false refusal, never a false repair.
#:
#: The n=864 tied-contact KKT artefact (first move 2.5893e-05) cannot be placed on this axis --
#: its `cond` was never recorded and the system is not kept -- so it is two points, not three.
_REPAIRABLE_CONDITION_FACTOR = 100.0

# WHY 100 AND NOT THE GEOMETRIC MEAN, which is how the constant this replaces was placed. That
# rule was right THERE because both its points were raw magnitudes with no model behind them, so
# their midpoint was the only pin available. Here there IS a model -- one step of refinement from
# a backward-stable solve moves the answer by about cond*eps -- so C is an allowance on a theory,
# not a midpoint between two observations. The geometric mean of the two points below is ~2e+07,
# which at cond = 1e12 puts the ceiling at 4.4e+03: every possible move passes, and the gate that
# exists to catch a broken backend would catch nothing. Two decades is the round allowance for the
# modest unknown constant in `move ~ cond*eps` and for `onenormest`'s underestimate.
#
# THE RING DOES NOT NEED THE ALLOWANCE, which is the point: measured 2026-09-17 its R is 4.09e-07
# to 9.38e-07, so it passes at C = 1 -- the bare model with no fudge -- by six decades. C is not
# fitted to it. What the allowance protects is some OTHER sound system whose move sits a little
# above its conditioning floor, and the n=864 artefact that cannot be placed here is exactly that
# unknown.
#
# AND ON A BADLY ENOUGH CONDITIONED SYSTEM THIS GATE IS INERT, which it must SAY rather than
# pretend otherwise. The ring measured cond_1 = 1.16e+17 to 2.41e+19, i.e. cond*eps = 25.8 to
# 5.3e+03: past cond*eps = 1 no digits are guaranteed and NO magnitude is anomalous, so magnitude
# cannot discriminate and the TRAJECTORY is the only working gate there. That is the correct
# behaviour and not a gap -- a gate claiming to separate what it cannot is the declaration that
# lies. The two gates are independent precisely so that one can go quiet without taking the other
# with it.


def direct_relative_residual(matrix, rhs, x) -> float | None:
    """``||A x - b|| / ||b||`` for a direct solve, or ``None`` when the statistic cannot speak.

    ``None`` means ``b == 0``: EVERY null-space vector satisfies ``A x = 0`` exactly, so a
    relative residual is 0/0 there and reporting 0.0 would be a green the arithmetic never
    earned. The caller must treat ``None`` as UNJUDGED, not as a pass.

    NOT cast to ``float``: the time-harmonic operator is COMPLEX, and casting its residual to a
    real dtype is what kept the harmonic call sites from being able to consult this guard at all.
    """
    # THE DENSE REAL VECTOR CASE, spelled without the storage dispatch: `np.linalg.norm` of a real
    # 1-D array IS `sqrt(v.dot(v))`, so this is the same statistic to the bit, at a fraction of
    # the call overhead the general path pays on the tiny systems a circuit march solves.
    if (
        type(matrix) is np.ndarray
        and type(rhs) is np.ndarray
        and type(x) is np.ndarray
        and rhs.ndim == 1
        and rhs.dtype.kind == x.dtype.kind == matrix.dtype.kind == 'f'
    ):
        b_square = float(rhs.dot(rhs))
        if b_square == 0.0:
            return None
        defect = matrix @ x.reshape(rhs.shape) - rhs
        return math.sqrt(float(defect.dot(defect))) / math.sqrt(b_square)
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

    WHY THIS EXISTS, measured 2026-09-16 on `case_c --scope full --set
    solve.chart=cartesian_3d --select fea,tet` at 3.0 mm: an n=207104 H(curl) solve was refused as "singular /
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
    * and whenever that move exceeds :data:`_REPAIRABLE_CONDITION_FACTOR` times this system's
      own ``cond * eps``, i.e. the vector is not an answer this system's conditioning spoiled but
      one unrelated to it -- a broken backend, which must be reported rather than corrected.

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
    # function did, and `tests/unit/test_nonlinear_port_golden.py`'s frozen trajectories
    # moved with it. The move is still REPORTED, so the verdict below costs no factorization.
    # ONE DECISION WITH A WINDOW, not two: repair only a move that is BOTH past the acceptance bar
    # and explicable as THIS system's own conditioning. Below `_DIRECT_REFINEMENT_RTOL` the guard
    # already accepts the answer and refining it would move a number never in question. Above
    # `_REPAIRABLE_CONDITION_FACTOR * cond * eps` the vector is not an answer conditioning spoiled
    # but one unrelated to the system -- a broken backend, which must be reported rather than
    # corrected. Outside the window `x` goes back bit-identical and the guard's verdict is exactly
    # what it always was.
    #
    # UNDECIDABLE IS NOT A PASS. When no condition estimate can be formed the magnitude gate has
    # no scale to judge against, and the answer is to DECLINE the repair -- the caller keeps its
    # vector and the guard below judges it on the acceptance bar, which is the behaviour that
    # existed before any repair did.
    if move <= _DIRECT_REFINEMENT_RTOL or move_is_this_systems_conditioning(matrix, move, factor=factor) is not True:
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


def _factorize(matrix):
    """``splu`` of *matrix*, or ``None`` when it is not square or is exactly singular.

    ONE factorization is shared by :func:`condition_estimate` and
    :func:`refinement_move_trajectory` because a refusal wants both numbers and they are both
    about the same operator; taking two would double the most expensive thing on the path.
    """
    shape = np.shape(matrix)
    if len(shape) != 2 or shape[0] != shape[1]:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', MatrixRankWarning)
            return splu(sp.csc_matrix(matrix))
    except (RuntimeError, MatrixRankWarning, ValueError):
        return None


def condition_estimate(matrix, *, factor=None) -> float | None:
    """A 1-norm condition estimate ``||A||_1 * ||A^-1||_1``, reusing a factorization if given.

    THE SCALE THE REFINEMENT MOVE IS JUDGED AGAINST. One step of iterative refinement from a
    backward-stable direct solve moves the answer by about ``cond * eps``, so "is this move big?"
    is not a question a constant can answer -- it is big RELATIVE TO THIS SYSTEM or it is not, and
    a solve gets a bigger ``cond`` by being a bigger solve. :data:`_REPAIRABLE_CONDITION_FACTOR`
    is the ratio that turns this into the ceiling.

    ``A^-1``'s norm is estimated through the factorization rather than formed: `onenormest` needs
    only matvecs, and both ``solve`` and its transpose come off the same `splu`. Priced at a
    handful of triangular solves on top of a factorization the caller already has.

    THE ERROR LEANS ONE WAY AND IT IS THE SAFE ONE. `onenormest` returns a LOWER bound, so this
    UNDER-estimates (measured 0.85x and 0.95x against the dense truth on n=8 and n=200 SPD
    plants). Used as a ceiling that makes the gate stricter, never looser.

    ``None`` when the operator cannot be factorized -- an exactly singular system has no finite
    condition number, and that case is the trajectory's to report, not this one's.
    """
    lu = factor if factor is not None else _factorize(matrix)
    if lu is None:
        return None
    n = int(np.shape(matrix)[0])
    operator = LinearOperator((n, n), matvec=lu.solve, rmatvec=lambda v: lu.solve(v, 'T'))
    try:
        return float(onenormest(sp.csc_matrix(matrix)) * onenormest(operator))
    except (ValueError, RuntimeError):
        return None


def move_is_this_systems_conditioning(matrix, move: float, *, factor=None) -> bool | None:
    """Is *move* explicable as this system's own ``cond * eps``, or is it an unrelated vector?

    THE MAGNITUDE GATE, and the half of the decision the trajectory cannot make: refinement
    converges on any solvable system whatever it is started from, so a backend returning garbage
    collapses exactly like a backend returning a conditioning-spoiled answer. What separates them
    is whether the starting vector was ever NEAR the answer, and ``cond * eps`` is the scale that
    says so in this system's own units.

    ``None`` when no condition estimate could be formed -- undecidable is reported as undecidable
    and the caller must not read it as either verdict.
    """
    cond = condition_estimate(matrix, factor=factor)
    if cond is None or not np.isfinite(cond):
        return None
    return bool(move <= _REPAIRABLE_CONDITION_FACTOR * cond * float(np.finfo(float).eps))


#: How far a trajectory must fall for :func:`refinement_move_trajectory` to be called COLLAPSING.
#: Six decades, and it is a RATIO between two points of the same run rather than a value at one --
#: so it carries no unit, no scale and no dependence on the order of the system. The two plants in
#: `test_the_refusal_says_whether_refinement_collapses_or_stalls.py` sit on either side of it by
#: many decades; it is not a knob tuned to squeeze one of them across.
_REFINEMENT_COLLAPSE_RATIO = 1e-6


def refinement_move_trajectory(
    matrix, rhs, x, *, max_steps: int = _DIRECT_REFINEMENT_MAX_STEPS, factor=None
) -> tuple[float, ...]:
    """The sequence of relative refinement moves from ``x`` -- EVIDENCE, never a repaired vector.

    :func:`repair_direct_solution` stops at the CONDITIONING ceiling and hands the guard a
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
    b = np.asarray(rhs.todense() if sp.issparse(rhs) else rhs).ravel()
    current = np.asarray(x.todense() if sp.issparse(x) else x).ravel().astype(float)
    factor = factor if factor is not None else _factorize(matrix)
    if factor is None:
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


def describe_refinement_trajectory(moves: tuple[float, ...], cond: float | None = None) -> str:
    """One clause naming WHICH of the two gates refused, and what the evidence was.

    The two are independent and the message must say which fired, because they send the reader to
    opposite places: a STALL is a missing gauge or a collapsed region, a magnitude past the
    conditioning ceiling is a backend that did not answer this system. Reporting the wrong one is
    the declaration-that-lies shape, and it is what the single constant here used to do.
    """
    if not moves:
        return 'diagnostic refinement trajectory: unavailable (no step could be taken)'
    shown = ', '.join(f'{m:.3e}' for m in moves)
    ceiling = (
        f'{_REPAIRABLE_CONDITION_FACTOR * cond * float(np.finfo(float).eps):.3e}'
        if cond is not None and np.isfinite(cond)
        else 'unavailable'
    )
    scale = f'cond_1 ~ {cond:.3e}, so the repairable ceiling is {ceiling}' if cond is not None else 'cond_1 unavailable'
    if len(moves) < 2:
        verdict = 'only ONE step was available, so it says nothing about convergence'
    elif moves[-1] <= moves[0] * _REFINEMENT_COLLAPSE_RATIO:
        verdict = (
            'the trajectory COLLAPSES, so the SYSTEM is solvable and this is NOT a rank '
            f'deficiency -- the refusal is the MAGNITUDE gate ({scale}): the starting vector is '
            'too far out to be an answer this system spoiled, so suspect the BACKEND, not a '
            'missing gauge'
        )
    else:
        verdict = (
            'the trajectory STALLS, so refinement has no unique answer to converge to -- this is '
            'a genuine rank deficiency and the gauge/reference advice above applies'
        )
    return f'diagnostic refinement trajectory [{shown}]: {verdict}'


def settle_direct_solution(matrix, rhs, x, backend: str) -> np.ndarray:
    """REPAIR, THEN JUDGE a direct answer -- the one spelling of the pair every direct site runs.

    A healthy answer takes ONE residual: below :data:`_DIRECT_RESIDUAL_RTOL` neither
    :func:`repair_direct_solution` nor :func:`reject_unsolved_direct_system` does anything else,
    so taking it twice bought nothing (measured 2026-10-12: 186,014 residuals for 93,007 solves on
    a 0.02 s PMSM position march). Past the trigger the pair runs exactly as before.
    """
    residual = direct_relative_residual(matrix, rhs, x)
    if residual is None or residual <= _DIRECT_RESIDUAL_RTOL:
        return np.asarray(x)
    x, moves = repair_direct_solution(matrix, rhs, x)
    reject_unsolved_direct_system(matrix, rhs, x, backend, refinement=moves[-1] if moves else None)
    return x


def reject_unsolved_direct_system(matrix, rhs, x, backend: str, *, refinement: float | None = None) -> None:
    """Raise :class:`LinearSolveError` when a DIRECT solve did not actually solve its system.

    The half of the singular guard that catches FINITE garbage: SuperLU emits a
    :class:`MatrixRankWarning` on a zero pivot and then returns either NaNs (the belt and
    :func:`~numerics_lab.linear.reject_nonfinite_solution` catch those) or a finite vector that
    does not satisfy ``A x = b``.
    Measured on the production CLI at ``case_i --set mesh.mesh_size="6mm"``: order 157, condition
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
    # the conditioning ceiling -- sized on an n=864 system, which a field solve is
    # under no obligation to stay below. The trajectory separates them and costs one factorization
    # on a solve that is already lost. It changes no verdict: this line is inside the raise.
    factor = _factorize(matrix)
    msg = (
        f'{msg} '
        f'{describe_refinement_trajectory(refinement_move_trajectory(matrix, rhs, x, factor=factor), condition_estimate(matrix, factor=factor))}'
    )
    raise LinearSolveError(msg)
