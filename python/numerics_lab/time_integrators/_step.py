"""One implicit-step Newton solve, factor holding, continuation and line searches."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from motronics.contracts.spec.analysis_formulation import (
    AffineConstraintProjectionProblem,
    AlternativeJacobianProblem,
    DerivativeWeights,
    FactorReusableProblem,
    HistoryProblem,
    ResidualFloorProblem,
    SemiDiscreteProblem,
    StepEvaluationProblem,
)
from motronics.gauss.linear import reject_nonfinite_solution
from motronics.gauss.linear.resolution import ADMISSION_BAR, row_bar, system_resolution
from motronics.gauss.time_integrators._forms import (
    IntegratorState,
    StepForm,
    change_below_residual_resolution,
    residual_floor,
)
from motronics.gauss.time_integrators.factor_reuse import FactorCache, apply_factor, factor_matrix
from motronics.gauss.time_integrators.local_convergence import LocalIterationCheck
from motronics.gauss.time_integrators.prepared import (
    PreparedCandidates,
    PreparedStageContext,
    PreparedStepAttempt,
    PreparedStepContext,
    PreparedStepProblem,
    finished_candidate,
    prepared_stage,
)

__all__ = [
    'HELD_STALL_RATIO',
    'HeldFactor',
    'ProblemCapabilities',
    'StepOutcome',
    'TrialNotEvaluable',
    'directional_line_search',
    'take_step',
]


class TrialNotEvaluable(RuntimeError):
    """Raised by a problem's ``residual`` when a line-search TRIAL point has no value to give.

    The condensed field-circuit residual is an inner field root; far from the outer root that inner
    Newton can have none. The backtracking search treats the trial as rejected and halves, exactly as
    for a residual above its band -- the march fails only if the step itself cannot converge.
    """


@dataclass(frozen=True)
class ProblemCapabilities:
    """Structural optional protocols, resolved once per run, never physical values.

    A direct step resolves its own scope. A march supplies one resolution to every
    stage, retry and probe; the problem's protocol structure stays fixed in that run.
    """

    factor_reusable: bool
    alternative_jacobian: bool
    residual_floor: bool
    history: bool
    prepared: bool = False
    step_evaluation: bool = False
    affine_projection: bool = False
    #: The problem solves this step itself (a native kernel); ``None`` from it declines.
    native_step: bool = False

    @classmethod
    def resolve(cls, problem: SemiDiscreteProblem) -> ProblemCapabilities:
        return cls(
            isinstance(problem, FactorReusableProblem),
            isinstance(problem, AlternativeJacobianProblem),
            isinstance(problem, ResidualFloorProblem),
            isinstance(problem, HistoryProblem),
            isinstance(problem, PreparedStepProblem),
            isinstance(problem, StepEvaluationProblem),
            isinstance(problem, AffineConstraintProjectionProblem),
            getattr(problem, 'native_step', None) is not None,
        )


def _at_roundoff_floor(problem, tangent, u: NDArray[np.floating], form, capabilities, r) -> bool:
    """Whether the residual ``r`` is already at the arithmetic floor; ``False`` if unknowable.

    The problem's own `residual_roundoff_floor` is preferred: it is a statement about that
    system's cancellation, which no generic derivation can see. :func:`residual_floor` is the
    fallback and needs only the tangent. A non-finite reading from either is discarded rather
    than compared -- a floor that is not a number cannot bound anything.

    A declared floor may be a scalar (compared with ``||r||_2``) or a PER-ROW vector, compared
    row by row: rows of different units cannot share one bound, and pooling them let a volt-scale
    branch row hide an ampere-scale KCL defect (`laplace.circuit.core.mna_solves.transient_residual_floor`).
    """
    if capabilities.residual_floor:
        # `None` IS AN ANSWER HERE -- "this problem cannot bound its own cancellation" -- and it
        # must fall through to the generic fallback rather than be accepted. It was not expressible
        # before: `float(...)` would have raised on it, so `mna_solves.transient_residual_floor`
        # said "no floor" by returning `0.0`, which is finite, was accepted, and suppressed the
        # fallback entirely while comparing every real residual against zero.
        declared = problem.residual_roundoff_floor(u, form, residual=r)
        if declared is not None:
            floor = np.asarray(declared, dtype=float)
            if np.isfinite(floor).all():
                if floor.ndim == 0:
                    return float(np.linalg.norm(r)) <= float(floor)
                return bool(np.all(np.abs(np.asarray(r, dtype=float).reshape(floor.shape)) <= floor))
    if tangent is None:
        return False
    floor = float(residual_floor(tangent, u))
    return bool(np.isfinite(floor)) and float(np.linalg.norm(r)) <= floor


#: The stall-polish's ratio threshold (MANUAL-627 (i)): a per-iteration residual ratio
#: above this marks the direction stalled. Calibrated on the GaN bridge's toggle steps:
#: the normal phase's ratios are ALL below 0.9, the stall's three-cycle (creep ~0.996,
#: jump 8-260, drop 0.004) trips twice consecutively (the creep + the jump) -- 0.9
#: separates them; a monotone creep (0.968 per iteration, MANUAL-627 (d)) also trips.
_STALL_RATIO = 0.9

#: A held (modified-Newton) factor is refreshed when its last step contracted the residual by less
#: than this -- the same bar the stationary ``newton-held-factor`` uses (``stall_ratio``).
HELD_STALL_RATIO = 0.5


class HeldFactor:
    """One tangent factorization HELD across Newton iterations AND steps (the modified Newton, 3A).

    Refreshed (re-assembled and re-factored) when it has served ``refactor_every`` solves, when the
    step form's DERIVATIVE WEIGHTS moved (a new ``dt`` changes ``alpha1`` and so the operator itself,
    not just its state dependence), and -- by the caller -- when a held step stalled. ``factorizations``
    and ``solves`` are the counters a benchmark reports. A non-finite back-substitution is refused.
    """

    def __init__(self, refactor_every: int) -> None:
        if int(refactor_every) < 1:
            msg = f'refactor_every must be >= 1, got {refactor_every!r}'
            raise ValueError(msg)
        self.refactor_every = int(refactor_every)
        self.matrix = None
        self._factor = None
        self._weights: tuple | None = None
        self._age = 0
        self.factorizations = 0
        self.solves = 0

    @staticmethod
    def _key(weights: DerivativeWeights) -> tuple:
        return (complex(weights.alpha0), complex(weights.alpha1), complex(weights.alpha2))

    def needs_refresh(self, weights: DerivativeWeights) -> bool:
        return self._factor is None or self._age >= self.refactor_every or self._key(weights) != self._weights

    def refresh(self, matrix, weights: DerivativeWeights) -> None:
        self.matrix = matrix
        self._factor = factor_matrix(matrix)
        self._weights = self._key(weights)
        self._age = 0
        self.factorizations += 1

    def solve(self, rhs: NDArray[np.floating]) -> NDArray[np.floating]:
        self._age += 1
        self.solves += 1
        x = np.asarray(apply_factor(self._factor, np.asarray(rhs, dtype=float)), dtype=float)
        return reject_nonfinite_solution(x, 'held-factor')


#: The gmin retry stages (the SPICE-standard continuation, the DC-op's own :data:`_GMIN_STAGES`
#: shape): when a step's plain Newton exhausts, the diagonal gmin is added to the tangent and
#: stepped down geometrically, so a near-singular node's step is bounded and the iterate follows
#: the damped path to the root (the final stage IS the plain solver). Opt-in -- a march that does
#: not pass ``gmin_stages`` takes the plain Newton, byte-identical (MEASURED 2026-08-13: the
#: SRDAB runner config's 55 exhausted steps at the compound channel+SR toggle drop to 0 with the
#: retry; the laplace power stage / buck marches are untouched because they never ask for it).
_GMIN_RETRY_STAGES = (1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8, 1.0e-10, 1.0e-12)


def _gmin_continue(
    problem: SemiDiscreteProblem,
    form: StepForm,
    guess: NDArray[np.floating],
    *,
    solve,
    max_newton: int,
    tol: float,
    tol_rel: float | None,
    residual_0: float,
    gmin_stages: tuple[float, ...],
) -> tuple[NDArray[np.floating], bool, float, bool]:
    """The gmin continuation for an exhausted step: solve ``(J + gmin I) delta = -R`` for
    each stage, the residual always the TRUE one (the same contract the DC-op's
    :func:`~motronics.laplace.circuit.core.transient.mna_solves._gmin_stepped` follows). Each stage is
    seeded by the previous stage's iterate; the bar is the caller's own (``tol`` or
    ``tol_rel * residual_0`` -- the plain Newton's first-iteration residual, the same bar the
    step already failed) PLUS, once the continuation is spent, :func:`residual_floor` -- the damped
    path must be allowed to land on the same arithmetic floor the plain Newton is.

    Returns ``(u, converged, residual, on_floor)``; ``on_floor`` says the verdict was the
    FLOOR's rather than the bar's, so :class:`TransientSolution` can record WHICH one carried
    the step.
    """
    t = form.t_eval
    u = np.asarray(guess, dtype=float).copy()
    tangent = None  # the last tangent built here, for the exhaustion floor below
    local = LocalIterationCheck(problem, tol_rel)
    for gmin in gmin_stages:
        for _it in range(max(2, max_newton)):
            u_arg, u_dot, u_ddot = form.u_arg(u), form.u_dot(u), form.u_ddot(u)
            r = np.asarray(problem.residual(u_arg, u_dot, u_ddot, t), dtype=float)
            residual = float(np.linalg.norm(r))
            initial = local.previous is None
            stable = local.observe(u, t)
            local_ok = local.accepts(stable, residual, initial=initial)
            if local_ok and (residual <= tol or (tol_rel is not None and residual <= tol_rel * residual_0)):
                return u, True, residual, False
            jac = problem.jacobian(u_arg, u_dot, u_ddot, t, form.weights)
            tangent = jac
            if sp.issparse(jac):
                damped = jac + gmin * sp.eye(jac.shape[0], format='csc')
            else:
                damped = jac + gmin * np.eye(jac.shape[0])
            delta = np.asarray(solve(damped, -r), dtype=float)
            u = u + delta.reshape(u.shape)
    u_arg, u_dot, u_ddot = form.u_arg(u), form.u_dot(u), form.u_ddot(u)
    r = np.asarray(problem.residual(u_arg, u_dot, u_ddot, t), dtype=float)
    residual = float(np.linalg.norm(r))
    local_ok = local.observe(u, t)
    if local_ok and (residual <= tol or (tol_rel is not None and residual <= tol_rel * residual_0)):
        return u, True, residual, False
    if local_ok and tangent is not None and residual <= residual_floor(tangent, u):
        return u, True, residual, True
    return u, False, residual, False


#: Secant (Illinois) evaluations :func:`directional_line_search` spends bracketing the line minimum.
_DIRECTIONAL_EVALS = 12


def directional_line_search(residual, u, step, r0, *, curvature: float = 0.5):
    """Minimize the step's ENERGY along ``step`` using only residuals: ``phi'(s) = R(u + s step) . step``.

    For a residual that IS the gradient of a convex potential (a magnetostatic step with a monotone
    law: the variational play, a monotone B-H curve, ``sigma/dt`` mass), ``phi`` is convex along the
    ray, so its minimizer is the root of the nondecreasing ``phi'``. Judging a step by ``|R|``
    instead is not a merit of that problem -- on a law with kinks the Newton direction need not
    lower ``|R|`` at all, measured on ipm_uv with the hysteretic steel: full steps 4e2 -> 5e3,
    100 iterations at ~10 % each. Differences of the potential itself would cancel to round-off
    near the root; its DERIVATIVE does not. The full step is taken when ``phi'(1) <= 0``; otherwise
    a bracketed Illinois secant stops at a LOW-side point ``-curvature |phi'(0)| <= phi'(s) <= 0``.
    A non-descent direction (``phi'(0) >= 0``) is taken whole: there is nothing to minimize along it.
    """
    g0 = float(np.dot(np.ravel(r0), np.ravel(step)))
    if not g0 < 0.0:
        return u + step
    g1 = float(np.dot(np.ravel(residual(u + step)), np.ravel(step)))
    if g1 <= 0.0:
        return u + step
    lo, glo, hi, ghi = 0.0, g0, 1.0, g1
    s, side = 1.0, 0
    for _ in range(_DIRECTIONAL_EVALS):
        s = lo - glo * (hi - lo) / (ghi - glo)
        g = float(np.dot(np.ravel(residual(u + s * step)), np.ravel(step)))
        if -curvature * abs(g0) <= g <= 0.0:
            return u + s * step
        if g < 0.0:
            lo, glo = s, g
            if side == -1:
                ghi *= 0.5
            side = -1
        else:
            hi, ghi = s, g
            if side == 1:
                glo *= 0.5
            side = 1
    # Unclosed: the bracket's LOW end, where phi' < 0 and so the energy fell; a point past the
    # minimum may sit above phi(0) however small its slope.
    return u + (lo if lo > 0.0 else s) * step


class RowAcceptance:
    """The per-row step verdict: every row of the step residual at :data:`ADMISSION_BAR` of its own scale.

    The scale is the admission's: ``|F_u| |u| + |F_u'| |u'| + |F(u, 0)|``, so a step this accepts is a
    state the finite-storage admission at the next law change can admit (ruling 2026-10-14; the NORM
    bar it replaces accepted Cummins row 16580 at 1.19x). Noise rows are judged by
    :func:`~motronics.gauss.linear.resolution.row_bar`. The two extra tangents are paid only when the
    previous iterate's bars do not already refuse the residual by 10x: that screen can only skip a
    check, never pass one.
    """

    _SCREEN = 10.0

    def __init__(self, problem, form, forcing) -> None:
        self.problem, self.form, self.forcing = problem, form, forcing
        self.bars = None

    def accepts(self, u, r) -> bool:
        residual = np.abs(np.asarray(r, dtype=float))
        if self.bars is not None and np.any(residual > self._SCREEN * self.bars):
            return False
        form, problem, t = self.form, self.problem, self.form.t_eval
        u_arg, u_dot, u_ddot = form.u_arg(u), form.u_dot(u), form.u_ddot(u)
        zero = np.zeros_like(np.asarray(u_dot, dtype=float))
        scale = np.abs(np.asarray(problem.residual(u_arg, zero, zero, t), dtype=float))
        for weights, operand in ((DerivativeWeights(1, 0, 0), u_arg), (DerivativeWeights(0, 1, 0), u_dot)):
            matrix = problem.jacobian(u_arg, u_dot, u_ddot, t, weights)
            magnitude = abs(matrix) if sp.issparse(matrix) else np.abs(np.asarray(matrix, dtype=float))
            scale = scale + np.asarray(magnitude @ np.abs(np.asarray(operand, dtype=float))).ravel()
        forcing = self.forcing()
        self.bars = row_bar(ADMISSION_BAR * scale, scale, system_resolution(scale, forcing), forcing)
        return bool(np.all(residual <= self.bars))


def _locally_stable(local: LocalIterationCheck, u, t, observed, tangent, forcing) -> bool:
    """The local check's verdict, or a change the residual's own arithmetic cannot resolve.

    A relative per-field bar judges a physically ZERO field against its own rounding noise and
    can never pass there; :func:`change_below_residual_resolution` asks the unit-safe question
    instead -- whether the change moved any residual row above that row's rounding floor. Both
    verdicts need a previous observation, so the first iteration is decided by the check alone.
    ``forcing`` returns each row's own source (the residual at the zero state), read only when the
    relative check has refused.
    """
    stable = local.observe(u, t)
    if stable or observed is None or tangent is None or not local.finite:
        return stable
    return change_below_residual_resolution(tangent, observed, u, forcing())


def _solve_step(
    problem: SemiDiscreteProblem,
    form: StepForm,
    guess: NDArray[np.floating],
    *,
    solve,
    max_newton: int,
    tol: float,
    tol_rel: float | None = None,
    factor_cache: FactorCache | None = None,
    backtrack_band: float | None = None,
    backtrack_cap: int = 8,
    line_search: str = 'backtrack',
    jacobian_tangent: Callable | None = None,
    gmin_stages: tuple[float, ...] | None = None,
    held: HeldFactor | None = None,
    capabilities: ProblemCapabilities | None = None,
    row_acceptance: bool = False,
) -> NDArray[np.floating]:
    """Advance one step by a Newton loop on ``residual(u_arg, u_dot, u_ddot, t)`` for ``u_{n+1}``.

    Each iteration evaluates the residual at the affine step form and asks the problem for the
    weighted tangent ``jacobian(..., form.weights)``, then takes ``du = -J^{-1} R``. For a LINEAR
    problem one iteration is exact (``max_newton = 1``, the default first cut); a NONLINEAR problem
    is where a ``gauss.nonlinear`` strategy plugs in -- this local loop is the minimal Newton that
    the richer registry-driven iteration would replace, driving the SAME residual/jacobian seam.

    ``factor_cache`` (W6) engages only when the problem exposes ``factor_reuse_key``: while the
    derived key holds, the tangent is neither rebuilt nor re-factored -- the cached factorization
    back-substitutes. A problem without the seam takes the per-solve path exactly as before.

    The BACKTRACKING line search (MANUAL-632/634): a Newton step that would RAISE the residual
    (the sharp GaN channel's overshoot at the commutation -- the measured two-cycle between the
    physical state and the nonphysical ~824 V state) is halved up to ``backtrack_cap`` times,
    accepting the first half-step whose residual drops below ``backtrack_band * r_current``
    (``None`` = the strict monotone: accept only a strict decrease). The calibration (measured
    on the GaN bridge with the CORRECTED root-find): the STRICT band with cap 8 drops the
    exhausted steps 82 -> 33 at the same converged trajectory -- the simple circuit's ON drop
    stays 0.4674 V in every variant, where the earlier measured 'path-change' was the wrong
    root-find's artifact. ``backtrack_cap = 0`` restores the bare Newton.

    The STALL-POLISH (MANUAL-627 (i)): when the backtracking still cannot break a direction
    (the per-iteration residual ratio > :data:`_STALL_RATIO` for two consecutive iterations --
    the fixed jacobian's step is nearly perpendicular to the residual's gradient, the measured
    three-cycle on the GaN toggle steps), the jacobian source switches to ``jacobian_tangent``
    -- the problem's local-derivative tangent, or the explicitly passed callable. A problem
    without the seam keeps the single-source Newton unchanged.

    Returns ``(u, converged, iterations, residual, on_floor)`` -- the CONVERGENCE CONTRACT:
    the caller (the marcher) labels an exhausted step rather than taking it silently (the FVS
    plan's replacement for the old silent accept; the DC-op precedent is
    :class:`OpNonConvergence`).

    ``on_floor`` is the verdict the other three values cannot express: a step that exhausted
    ``max_newton`` and was passed by :func:`residual_floor` rather than by ``tol``/``tol_rel``
    used to be indistinguishable from one that met its bar on the first iteration. Both read
    ``converged=True``, so a march that quietly STOPPED converging could be certified by the
    arithmetic with nothing for a caller to read -- the shape ``taste.md`` forbids. The flag
    rides out to :attr:`TransientSolution.floor_converged_times`.

    ``row_acceptance`` replaces every norm verdict (``tol``, ``tol_rel`` and the arithmetic floor) by
    :class:`RowAcceptance`; it is the one bar there is, so the norm bars and the gmin continuation
    beside it refuse.
    """
    if row_acceptance and (tol_rel is not None or gmin_stages is not None):
        msg = 'a per-row step acceptance is the only bar: it takes no tol_rel and no gmin continuation'
        raise ValueError(msg)
    if capabilities is None:
        capabilities = ProblemCapabilities.resolve(problem)
    if capabilities.step_evaluation:
        # An endpoint owner's DIFFERENT problem has protocols of its own; the owner itself
        # (no endpoint at this instant) keeps the march's resolution -- re-resolving it cost
        # seven runtime protocol checks per stage (measured 2026-10-12, ~11% of a dq march).
        stepped = problem.step_problem(form.t_eval)
        if stepped is not problem:
            if not isinstance(stepped, SemiDiscreteProblem):
                msg = 'step endpoint owner must return a semi-discrete problem'
                raise TypeError(msg)
            problem, capabilities = stepped, ProblemCapabilities.resolve(stepped)
    if (
        capabilities.native_step
        and tol_rel is None
        and not row_acceptance
        and line_search == 'backtrack'
        and jacobian_tangent is None
        and gmin_stages is None
        and held is None
    ):
        # THE PROBLEM'S OWN KERNEL, on the default Newton options only: it is golden-equivalent to
        # this loop on that path and nowhere else, so every other option keeps the loop below.
        outcome = problem.native_step(
            form, guess, tol=tol, max_newton=max_newton, backtrack_cap=backtrack_cap, backtrack_band=backtrack_band
        )
        if outcome is not None:
            return outcome
    reusable = factor_cache is not None and capabilities.factor_reusable
    # The stall-polish's phase-2 source (MANUAL-627 (i)): the problem's local-derivative
    # jacobian, engaged when the default jacobian's per-iteration creep marks a stalled
    # direction. The default ``None`` reads the problem's optional ``jacobian_tangent``
    # seam; a problem without it keeps the single-source Newton unchanged. The
    # ``has_tangent_jacobian`` flag (the laplace's channel-switch presence) gates the
    # ENGAGEMENT: a circuit whose tangent equals its default jacobian (the RON/ROFF
    # switches, the diodes -- their stamps already ARE the local derivative) would pay
    # the tangent phase's factor-cache bypass for nothing (measured: the buck's reuse
    # counts broke without the gate).
    alternative = problem if capabilities.alternative_jacobian else None
    tangent_source = (
        jacobian_tangent
        if jacobian_tangent is not None
        else (alternative.jacobian_tangent if alternative is not None else None)
    )
    tangent_helpful = jacobian_tangent is not None or (alternative is not None and alternative.has_tangent_jacobian)

    def project_trial(value):
        value = np.asarray(value, dtype=float)
        if not capabilities.affine_projection:
            return value
        alpha = form.weights.alpha0
        if alpha.imag != 0 or alpha.real == 0:
            msg = 'affine state constraint projection requires a nonzero real state weight'
            raise ValueError(msg)
        argument = form.u_arg(value)
        projected = np.asarray(problem.project_affine_state(argument.copy(), form.t_eval))
        if projected.shape != argument.shape or not np.isrealobj(projected) or not np.isfinite(projected).all():
            msg = 'affine constraint owner must return a same-shape finite real trial state'
            raise ValueError(msg)
        return value + (projected - argument) / alpha.real

    t = form.t_eval
    # THE GUESS IS NOT PROJECTED. A full Newton step lands every affine row exactly, so projecting
    # the caller's guess buys nothing there -- and it costs: it moves the constrained rows ALONE,
    # so the first tangent is taken where every other unknown is stale. Behind a steep clamp that
    # is a worse linearisation, the absolute ``tol`` then passed a nanosiemens node millivolts off,
    # and `test_varistor`'s divider read 2.6e-5 off an identity exact to 2.4e-16 (MEASURED
    # 2026-10-09; introduced by 2bbcffe10e). Only a DAMPED trial can leave the plane, so only the
    # backtracking trials below are projected.
    u = np.asarray(guess, dtype=float).copy()
    converged = False
    on_floor = False
    iterations = 0
    residual = float('inf')
    residual_0 = float('inf')
    prev_residual = None
    prev_ratio: float | None = None
    slow = 0
    tangent_active = False
    polish_spent = False
    capped = False
    # The last tangent BUILT (never rebuilt for this purpose): the exhaustion check below needs
    # the matrix, not its norm, because the floor is a per-row ``|J| . |u|``. On a factor-cache
    # HIT no matrix is built, so the cache's own -- valid by its contract that equal keys imply an
    # equal jacobian -- stands in, and a hit still never pays a build.
    tangent = None
    local = LocalIterationCheck(problem, ADMISSION_BAR if row_acceptance else tol_rel)
    observed = None  # the iterate the local check last observed, for the resolution test

    def row_forcing():
        # Each row's own source: the residual with every state and rate at zero.
        zero = np.zeros_like(np.asarray(guess, dtype=float))
        return np.asarray(problem.residual(zero, zero, zero, t), dtype=float)

    rows = RowAcceptance(problem, form, row_forcing) if row_acceptance else None

    def met(residual, r, u):
        if rows is not None:
            return rows.accepts(u, r)
        return residual <= tol or (tol_rel is not None and residual <= tol_rel * residual_0)

    for _it in range(1, max_newton + 1):
        iterations = _it
        u_arg, u_dot, u_ddot = form.u_arg(u), form.u_dot(u), form.u_ddot(u)
        r = np.asarray(problem.residual(u_arg, u_dot, u_ddot, t), dtype=float)
        residual = float(np.linalg.norm(r))
        stable = _locally_stable(local, u, t, observed, tangent, row_forcing)
        observed = u.copy()
        local_ok = local.accepts(stable, residual, initial=_it == 1)
        if _it == 1:
            residual_0 = residual
        # The RELATIVE contract (MANUAL-628): the absolute tol is scale-blind -- a
        # residual that stalls at 1e-1 on a ~5400 A switch-current scale is 2e-5
        # relative, and the Newton cannot reach 1e-12 there by SCALE, not by error.
        # The opt-in ``tol_rel`` accepts a step whose residual dropped below
        # ``tol_rel * r0`` (r0 = the first iteration's norm). Off by default: every
        # existing march keeps the absolute contract unchanged.
        if local_ok and met(residual, r, u):
            converged = True
            break
        # The STALL-POLISH (MANUAL-627 (i)): the fixed jacobian (the switch's chord i/v)
        # creeps -- the per-iteration residual ratio ~1 -- when its direction is nearly
        # perpendicular to the residual's gradient: the measured three-cycle on the GaN
        # channel-toggle steps (creep 0.996, jump 8-260, drop 0.004) exhausts the ceiling
        # at a ~1e-3-relative floor OFF the root. After TWO consecutive stalled
        # iterations the jacobian source switches to the problem's local-derivative
        # tangent: the chord's basin keeps the global exploration, the tangent finishes
        # the local convergence the chord cannot (measured: the toggle step -- 60-exhausted
        # at the floor vs 16 iterations to 4e-5). The factor cache is bypassed while the
        # tangent drives: its reuse key proves constancy for the default stamps only.
        if prev_residual is not None:
            ratio = residual / prev_residual
            prev_ratio = ratio
            slow = slow + 1 if (capped or ratio > _STALL_RATIO) else 0
            # A RESIDUAL AT THE FLOOR IS FINISHED: it is already rounding, so the remaining
            # iterations can only reshuffle it (and a rounding residual need not fall
            # monotonically, so no stall ratio can be relied on to see it). It is the same floor
            # verdict the exhaustion branch below gives, reached before the ceiling instead of at
            # it (measured on ipm_uv's chopper: 50 iterations, ~30 s a step, at |r| ~ 4e-9).
            if rows is None and local_ok and _at_roundoff_floor(problem, tangent, u, form, capabilities, r):
                converged = True
                on_floor = True
                break
            if slow >= 2 and tangent_helpful and not polish_spent:
                tangent_active = True
        prev_residual = residual
        capped_prev, capped = capped, False
        if tangent_active:
            jac = tangent_source(u_arg, u_dot, u_ddot, t, form.weights)
            tangent = jac
            du = np.asarray(solve(jac, -r), dtype=float)
        elif not reusable and held is not None:
            # THE MODIFIED NEWTON (ruling 3A): back-substitute the held factor; refresh it when it is
            # ``refactor_every`` solves old, the step form's weights moved (a new dt), or the last
            # held step contracted the residual by less than HELD_STALL_RATIO / needed backtracking.
            stalled = prev_ratio is not None and prev_ratio > HELD_STALL_RATIO
            if held.needs_refresh(form.weights) or stalled or capped_prev:
                held.refresh(problem.jacobian(u_arg, u_dot, u_ddot, t, form.weights), form.weights)
            tangent = held.matrix
            du = held.solve(-r)
        elif not reusable:
            jac = problem.jacobian(u_arg, u_dot, u_ddot, t, form.weights)
            tangent = jac
            du = np.asarray(solve(jac, -r), dtype=float)
        else:
            key = problem.factor_reuse_key(u_arg, u_dot, u_ddot, t, form.weights)
            du = np.asarray(
                factor_cache.solve(key, lambda: problem.jacobian(u_arg, u_dot, u_ddot, t, form.weights), -r),
                dtype=float,
            )
            tangent = factor_cache.matrix
        step = du.reshape(u.shape)
        if max_newton > 1 and line_search == 'directional':
            u = directional_line_search(
                lambda x: np.asarray(problem.residual(form.u_arg(x), form.u_dot(x), form.u_ddot(x), t), dtype=float),
                u,
                step,
                r,
            )
        elif max_newton > 1 and backtrack_cap > 0:
            raised = False
            evaluable = True
            for _ in range(backtrack_cap):
                u_try = project_trial(u + step)
                u_arg, u_dot, u_ddot = form.u_arg(u_try), form.u_dot(u_try), form.u_ddot(u_try)
                try:
                    r_try = np.asarray(problem.residual(u_arg, u_dot, u_ddot, t), dtype=float)
                except TrialNotEvaluable:
                    # The problem cannot evaluate this trial (an inner solve without a root): the
                    # trial is REJECTED exactly like one whose residual is too large -- halve.
                    evaluable = False
                    step = 0.5 * step
                    continue
                evaluable = True
                limit = residual if backtrack_band is None else backtrack_band * residual
                trial = float(np.linalg.norm(r_try))
                if trial < limit:
                    u = u_try
                    raised = trial >= residual
                    break
                step = 0.5 * step
            else:
                # Exhausted: take the last trial only if it could be evaluated at all.
                if evaluable:
                    u = project_trial(u + step)
                capped = True
            if tangent_active and (capped or raised):
                # THE POLISH IS A TRIAL, NOT A ONE-WAY SWITCH: an alternative operator whose step
                # does not LOWER the residual has done no better than the direction it replaced, so
                # the default source takes back over and the polish is spent for this step. The
                # test is a strict decrease and not the tolerant band the march accepts a default
                # step within: under that band a bad alternative is never capped, only accepted
                # while the residual climbs. Measured on `ipm_uv --point rated` (vacuum B-H tail):
                # the exact tangent's early overshoot engaged the magnetic secant, which drove the
                # t = 4.44 ms step 158 -> 445 over 100 iterations; withheld, it converged in 12.
                tangent_active, polish_spent, slow = False, True, 0
        else:
            u = project_trial(u + step)
    else:
        # The ceiling exhausted: re-evaluate the residual at the FINAL u (the last solve's
        # result). The loop's last check read the PRE-solve guess -- reporting that would
        # label a step that actually converged as exhausted, the contract's own lie.
        u_arg, u_dot, u_ddot = form.u_arg(u), form.u_dot(u), form.u_ddot(u)
        r = np.asarray(problem.residual(u_arg, u_dot, u_ddot, t), dtype=float)
        residual = float(np.linalg.norm(r))
        local_ok = _locally_stable(local, u, t, observed, tangent, row_forcing)
        if local_ok and met(residual, r, u):
            converged = True
        elif rows is None and local_ok and _at_roundoff_floor(problem, tangent, u, form, capabilities, r):
            on_floor = True
            # THE ARITHMETIC FLOOR, and the reason an exactly-solved LINEAR step no longer lies.
            # ``tol`` is ABSOLUTE on a residual whose scale is the supply, so the same one-Newton
            # exact solve read 6.17e-13 at 24 V (under the bar) and 4.95e-11 at 2400 V (over it).
            # A floor has to scale WITH the supply, exactly as the residual does.
            #
            # TWO SOURCES, BOTH KEPT, because each has a consumer and either alone drops a
            # capability. The PROBLEM'S OWN reading wins where it exists: some affine systems
            # cannot represent a zero residual after their one exact solve, because large signed
            # terms cancel only to the scale of floating-point row summation, and only the problem
            # knows that scale (`laplace.circuit.core.mna` measures and exposes it). Where no
            # problem declares one, :func:`residual_floor` derives ``eps * |J| . |u|`` from the
            # tangent, which needs nothing of the caller and covers every other problem.
            #
            # Consulted ONLY after the requested bar was missed: here at the ceiling, and inside
            # the loop from the second iteration on, where a residual at the floor has no progress
            # left to make.
            converged = True
    if not converged and gmin_stages is not None:
        # The SPICE-standard gmin continuation (opt-in): a fresh damped Newton path from the
        # clean guess, stepping a diagonal gmin down geometrically. The plain Newton's stall at
        # a near-singular node is a non-solution fixed point (MEASURED 2026-08-13: the SRDAB
        # runner config's compound channel+SR toggle flings the state ~2e7 V across the SR's
        # PWL body-diode knee); the damped path bounds the step and lands where the plain one
        # cannot. If the continuation ALSO fails, the plain Newton's exhausted iterate stands
        # (the label carries the outcome either way).
        u_g, conv_g, residual_g, floor_g = _gmin_continue(
            problem,
            form,
            guess,
            solve=solve,
            max_newton=max_newton,
            tol=tol,
            tol_rel=tol_rel,
            residual_0=residual_0,
            gmin_stages=gmin_stages,
        )
        if conv_g:
            u = u_g
            converged = True
            on_floor = floor_g
            residual = residual_g
        iterations += max(2, max_newton) * len(gmin_stages)
    return u, converged, iterations, residual, on_floor


@dataclass(frozen=True)
class StepOutcome:
    """One whole step of a scheme, every stage solved: the new history and the step's verdict.

    ``converged`` holds only if EVERY stage converged; ``residual`` is the worst stage's, and
    ``rate`` is the final stage's ``u_dot`` -- the rate the trajectory point was solved with.
    """

    state: IntegratorState
    u: NDArray[np.floating]
    converged: bool
    iterations: int
    residual: float
    on_floor: bool
    rate: NDArray[np.floating]
    residual_time: float
    candidate: object | None = None


def take_step(
    problem,
    integrator,
    state: IntegratorState,
    dt: float,
    t_new: float,
    *,
    endpoint_left: bool = False,
    capabilities: ProblemCapabilities | None = None,
    candidate_registry: PreparedCandidates | None = None,
    **solve_kw,
) -> StepOutcome:
    """Advance ``state`` by ``dt`` to ``t_new`` through every stage of ``integrator``.

    The ONE place a marcher turns a scheme into solves: ``prepare`` -> :func:`_solve_step` ->
    ``advance``, ``integrator.stages`` times. A multi-stage scheme's intermediate state never leaves
    this function, so every marcher, probe and landing treats a step as one opaque move.
    ``endpoint_left`` lands a declared discontinuity using its left limit; the clock still lands
    exactly at ``t_new`` and the next restarted step evaluates the post-edge system.
    """
    if capabilities is None:
        capabilities = ProblemCapabilities.resolve(problem)
    converged, on_floor, iterations, worst = True, False, 0, 0.0
    u = state.last_u
    form = None
    attempt = None
    parent_candidate = state.prepared_candidate
    if capabilities.prepared:
        attempt = problem.begin_step(PreparedStepContext(integrator, state.copy(), dt, t_new, parent_candidate))
        if not isinstance(attempt, PreparedStepAttempt):
            msg = 'prepared step owner must return a prepared step attempt'
            raise TypeError(msg)
    candidate = None
    try:
        for index in range(integrator.stages):
            form = integrator.prepare(state, dt, t_new)
            stage_endpoint_left = endpoint_left and form.t_eval == t_new
            if stage_endpoint_left and attempt is None:
                form = replace(form, t_eval=float(np.nextafter(t_new, -np.inf)))
            stage_problem = problem
            stage_capabilities = capabilities
            if attempt is not None:
                stage_problem = prepared_stage(
                    attempt, PreparedStageContext(form, state.copy(), index, stage_endpoint_left)
                )
                stage_capabilities = None
            u, ok, its, residual, floor = _solve_step(
                stage_problem, form, state.last_u, capabilities=stage_capabilities, **solve_kw
            )
            converged, on_floor = converged and ok, on_floor or floor
            iterations, worst = iterations + its, max(worst, residual)
            if attempt is not None:
                attempt.stage_solved(u.copy())
            state = integrator.advance(state, u, dt, t_new, form)
        if attempt is not None:
            candidate = finished_candidate(attempt, parent_candidate, candidate_registry)
    except BaseException:
        if attempt is not None:
            attempt.abort()
        raise
    state.prepared_candidate = candidate
    assert form is not None
    return StepOutcome(state, u, converged, iterations, worst, on_floor, form.u_dot(u), form.t_eval, candidate)
