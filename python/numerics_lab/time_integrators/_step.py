"""One implicit-step Newton solve, factor holding, continuation and line searches."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from numerics_lab.formulation import (
    AffineConstraintAdmissionProblem,
    AffineConstraintProjectionProblem,
    AlternativeJacobianProblem,
    DerivativeWeights,
    FactorReusableProblem,
    HistoryProblem,
    ResidualFloorProblem,
    SemiDiscreteProblem,
    StepEvaluationProblem,
)
from numerics_lab.line_search import TrialNotEvaluable, backtracking, directional_line_search
from numerics_lab.linear import reject_nonfinite_solution
from numerics_lab.linear.resolution import ADMISSION_BAR, FLOOR_SAFETY, row_bar, system_resolution
from numerics_lab.time_integrators._forms import (
    IntegratorState,
    StepForm,
    _roundoff_floor,
    change_below_residual_resolution,
    residual_floor_by_row,
)
from numerics_lab.time_integrators.factor_reuse import FactorCache, apply_factor, factor_matrix
from numerics_lab.time_integrators.local_convergence import LocalIterationCheck
from numerics_lab.time_integrators.prepared import (
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
    'RowVerdict',
    'StepOutcome',
    'TrialNotEvaluable',
    'take_step',
]


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
    affine_admission: bool = False

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
            isinstance(problem, AffineConstraintAdmissionProblem),
        )


def _affine_admitted(problem, form, u, capabilities) -> bool:
    if not capabilities.affine_admission:
        return True
    admitted = problem.affine_state_is_admitted(form.u_arg(u), form.t_eval)
    if not isinstance(admitted, (bool, np.bool_)):
        msg = 'affine state admission must return a boolean verdict'
        raise TypeError(msg)
    return bool(admitted)


def _at_roundoff_floor(problem, tangent, u: NDArray[np.floating], form, capabilities, r) -> bool:
    """Whether the residual ``r`` is already at the arithmetic floor; ``False`` if unknowable.

    The problem's own `residual_roundoff_floor` is preferred: it is a statement about that
    system's cancellation, which no generic derivation can see. The fallback
    :func:`residual_floor_by_row` needs only the tangent and preserves each row's units.
    A non-finite reading from either is discarded rather
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
    floor = residual_floor_by_row(tangent, u)
    if not np.isfinite(floor).all():
        return False
    residual = np.abs(np.asarray(r, dtype=float).reshape(floor.shape))
    if np.all(residual <= floor) or form is None:
        return bool(np.all(residual <= floor))
    # ``|J| |u|`` sees only the state-proportional terms. A row is also a sum of its SOURCE ``F(u, 0)``
    # and its RATE ``F_u' u'``, and cannot resolve below their rounding either: a winding row whose
    # linkage crosses zero under 512 V reads ``|J| |u| ~ 0`` and is a 512 V sum (case_a dq-map,
    # 2026-10-22). Read only where the tangent floor refused, so a passing verdict costs nothing more.
    # ``F(u, 0)`` is read ONCE here and handed to both judges: each used to evaluate it for itself, at the
    # same state and time (2026-10-25: two full residuals per refused check, ~a third of this function).
    source = _zero_rate_source(problem, form, u)
    seen = None if source is None else _unseen_term_floor(problem, form, u, source)
    # An unknowable term floor is NaN, so the finiteness test below answers it ``False``.
    floor = floor + (np.full(floor.shape, np.nan) if seen is None else seen.reshape(floor.shape))
    # A HOMOGENEOUS row holding only the solve's rounding sits far under its own floor; it is noise,
    # judged at the system's floor -- the same rule `RowAcceptance` applies (`_noise_judged`).
    if np.isfinite(floor).all():
        floor = _noise_judged(problem, form, u, floor, floor, float(np.max(floor, initial=0.0)), source=source)
    return bool(np.isfinite(floor).all()) and bool(np.all(residual <= floor))


def _zero_rate_source(problem, form, u):
    """The residual at the actual state and time with zero rate and acceleration, ``F(u, 0)``; ``None`` if refused."""
    u_arg = np.asarray(form.u_arg(u), dtype=float)
    zero = np.zeros_like(np.asarray(form.u_dot(u), dtype=float))
    try:
        return np.asarray(problem.residual(u_arg, zero, zero, form.t_eval), dtype=float).ravel()
    except (ValueError, ArithmeticError):
        return None


def _noise_judged(problem, form, u, bars, operands, resolution: float, *, source=None):
    """``bars``, with every HOMOGENEOUS row whose ``operands`` are under ``resolution`` judged at it.

    A row's source is read as its affine intercept at the ACTUAL state, ``F(u, 0) - F_u u``: no zero
    state is queried, so a constitutive law's domain is never left (the constitutive-domain control).
    A row whose intercept vanishes to its own rounding has no source of its own; when its whole scale is
    also under the system's resolution, its residual is the coupled solve's rounding (`row_bar`).
    """
    u_arg = np.asarray(form.u_arg(u), dtype=float)
    zero = np.zeros_like(np.asarray(form.u_dot(u), dtype=float))
    try:
        if source is None:
            source = np.asarray(problem.residual(u_arg, zero, zero, form.t_eval), dtype=float).ravel()
        jacobian = problem.jacobian(u_arg, zero, zero, form.t_eval, DerivativeWeights(1, 0, 0))
    except (ValueError, ArithmeticError):
        return np.asarray(bars, dtype=float)  # Unknowable homogeneity keeps every row's own bar.
    magnitude = abs(jacobian) if sp.issparse(jacobian) else np.abs(np.asarray(jacobian, dtype=float))
    state_terms = np.asarray(magnitude @ np.abs(u_arg)).ravel()
    intercept = source - np.asarray(jacobian @ u_arg).ravel()
    homogeneous = np.abs(intercept) <= FLOOR_SAFETY * float(np.finfo(float).eps) * state_terms
    return row_bar(bars, operands, resolution, np.where(homogeneous, 0.0, 1.0))


def _unseen_term_floor(problem, form, u, source):
    """The rounding of a row's SOURCE, RATE and ACCELERATION terms, which ``|J| |u|`` cannot see; ``None`` if unknowable."""
    # KNOWN LIMIT: the source term is read as |F(u, 0)|, the magnitude of its SUM. Two state-independent
    # sources that cancel in one row (v_a - v_b, both near 512 V) read near zero here although their
    # rounding is that of 512 V; a generic residual does not expose its terms, so only a problem's own
    # `residual_roundoff_floor` can state that row's floor.
    u_arg, u_dot, u_ddot, t = form.u_arg(u), form.u_dot(u), form.u_ddot(u), form.t_eval
    source = np.abs(source)
    try:
        rate = problem.jacobian(u_arg, u_dot, u_ddot, t, DerivativeWeights(0, 1, 0))
        acceleration = problem.jacobian(u_arg, u_dot, u_ddot, t, DerivativeWeights(0, 0, 1))
    except (ValueError, ArithmeticError):
        # A constitutive law may refuse the zero-rate point. The extra floor is then UNKNOWABLE, which this
        # function answers ``False`` -- Newton iterates on; it never crashes on a diagnostic read.
        return None
    return (
        residual_floor_by_row(rate, u_dot)
        + residual_floor_by_row(acceleration, np.asarray(u_ddot, dtype=float))
        + _roundoff_floor(source)
    )


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
    a downstream MNA gmin-stepping solve follows). Each stage is
    seeded by the previous stage's iterate; the bar is the caller's own (``tol`` or
    ``tol_rel * residual_0`` -- the plain Newton's first-iteration residual, the same bar the
    step already failed) PLUS, once the continuation is spent, the same per-row arithmetic floor -- the damped
    path must be allowed to land on the same arithmetic floor the plain Newton is.

    Returns ``(u, converged, residual, on_floor)``; ``on_floor`` says the verdict was the
    FLOOR's rather than the bar's, so :class:`TransientSolution` can record WHICH one carried
    the step.
    """
    t = form.t_eval
    u = np.asarray(guess, dtype=float).copy()
    tangent = None  # the last tangent built here, for the exhaustion floor below
    local = LocalIterationCheck(problem, tol_rel)
    capabilities = ProblemCapabilities.resolve(problem)
    for gmin in gmin_stages:
        for _it in range(max(2, max_newton)):
            u_arg, u_dot, u_ddot = form.u_arg(u), form.u_dot(u), form.u_ddot(u)
            r = np.asarray(problem.residual(u_arg, u_dot, u_ddot, t), dtype=float)
            residual = float(np.linalg.norm(r))
            initial = local.previous is None
            stable = local.observe(u, t)
            local_ok = local.accepts(stable, residual, initial=initial)
            local_ok = local_ok and _affine_admitted(problem, form, u, capabilities)
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
    local_ok = local_ok and _affine_admitted(problem, form, u, capabilities)
    if local_ok and (residual <= tol or (tol_rel is not None and residual <= tol_rel * residual_0)):
        return u, True, residual, False
    if local_ok and _at_roundoff_floor(problem, tangent, u, form, capabilities, r):
        return u, True, residual, True
    return u, False, residual, False


@dataclass(frozen=True)
class RowVerdict:
    """The row a per-row verdict judged WORST: over its bar by the largest ratio (or nearest to it).

    ``bar = max(own_bar, floor)`` with ``own_bar = ADMISSION_BAR * scale``; ``owner`` is the problem's
    ``row_owner(row)`` when it declares one (which block, which unknown), else ``None``. A refused step
    carries this so the refusal says WHICH row and against WHAT, not a residual norm.
    """

    row: int
    owner: str | None
    residual: float
    bar: float
    own_bar: float
    floor: float
    scale: float

    @property
    def ratio(self) -> float:
        return self.residual / self.bar if self.bar > 0 else float('inf')

    def __str__(self) -> str:
        who = '' if self.owner is None else f' ({self.owner})'
        return (
            f'row {self.row}{who}: |r| {self.residual:.3e} against bar {self.bar:.3e} ({self.ratio:.3g}x); '
            f'scale {self.scale:.3e}, own bar {self.own_bar:.3e}, construction floor {self.floor:.3e}'
        )


class RowAcceptance:
    """The per-row step verdict: every row of the step residual at :data:`ADMISSION_BAR` of its own scale.

    The scale is the admission's: ``|F_u| |u| + |F_u'| |u'| + |F(u, 0)|``, so a step this accepts is a
    state the finite-storage admission at the next law change can admit (ruling 2026-10-14; the NORM
    bar it replaces accepted case_a row 16580 at 1.19x). No bar sits below the step's own CONSTRUCTION
    floor ``FLOOR_SAFETY eps |F_u'| |alpha1 u|``: the rate is a difference of states, so a row whose rate
    is small cannot resolve below it. ``|u|`` there is each unknown's RESOLUTION: ``|u|`` itself, or what
    the problem's optional ``state_resolution(u)`` declares for an unknown that is a cancelling sum (a
    winding linkage ``C^T A``, case_a 2026-10-24: resolved to ``eps |C|^T |A|``, ~1e3x ``eps |lambda|``) (case_a 2026-10-15: a linkage row at a 1.85 us zero-vector step
    was held to 1.2e-13 under a ~1e-11 rounding, and Newton exhausted). No source homogeneity is
    inferred from a generic nonlinear equation: its zero state need not belong to its domain.

    THE SCREEN. Reading the scale costs two tangents, so an iterate within :data:`_NEAR` of the last
    judged one -- relative, in the STATE's own max norm AND in the RATE's, because the scale reads
    ``|F_u'| |u'|`` and ``u' = alpha1 (u - origin)`` turns a state change that is near into a rate
    change that is not (case_a 2026-10-24: a circuit row judged at rest kept rest-sized bars while
    every later iterate moved its rate from zero to the row's whole size, and was refused unread 49
    times) -- whose residual still exceeds that one's bars by
    :data:`_SCREEN` on some row is refused unread: so close, the scale has not moved by 10x, and a row
    10x over its bar is over it. An iterate farther away is always read -- a first iterate at rest has
    ``|u| = 0`` and a rest-sized scale, and reusing its bars refused every later iterate (measured on
    the case_a first step, 2026-10-15). A screen can only cost an iteration, never pass one.
    """

    _SCREEN = 10.0
    _NEAR = 1e-3

    def __init__(self, problem, form) -> None:
        self.problem, self.form = problem, form
        self.bars = self.judged = self.judged_rate = self.worst = None

    def accepts(self, u, r) -> bool:
        residual = np.abs(np.asarray(r, dtype=float))
        u = np.asarray(u, dtype=float)
        form, problem, t = self.form, self.problem, self.form.t_eval
        u_arg, u_dot, u_ddot = form.u_arg(u), form.u_dot(u), form.u_ddot(u)
        rate = np.asarray(u_dot, dtype=float)
        if (
            self.bars is not None
            and _near(u, self.judged, self._NEAR)
            and _near(rate, self.judged_rate, self._NEAR)
            and np.any(residual > self._SCREEN * self.bars)
        ):
            return False
        zero = np.zeros_like(np.asarray(u_dot, dtype=float))
        scale = np.abs(np.asarray(problem.residual(u_arg, zero, zero, t), dtype=float))
        magnitudes = []
        for weights, operand in ((DerivativeWeights(1, 0, 0), u_arg), (DerivativeWeights(0, 1, 0), u_dot)):
            matrix = problem.jacobian(u_arg, u_dot, u_ddot, t, weights)
            magnitudes.append(abs(matrix) if sp.issparse(matrix) else np.abs(np.asarray(matrix, dtype=float)))
            scale = scale + np.asarray(magnitudes[-1] @ np.abs(np.asarray(operand, dtype=float))).ravel()
        # THE RATE IS BUILT FROM THE STATE, ``u' = alpha1 (u - origin) + h``: it carries the rounding of
        # ``alpha1 u`` however small the rate is, so no row resolves below that construction floor.
        # An unknown that is itself a CANCELLING SUM (a linkage ``C^T A``) is resolved only to that sum's
        # rounding, never to ``eps |u|``: the problem that knows the sum declares it (``state_resolution``).
        resolved = np.abs(u)
        declared = getattr(problem, 'state_resolution', None)
        if declared is not None:
            resolved = np.maximum(resolved, np.abs(np.asarray(declared(u), dtype=float)).reshape(resolved.shape))
        built = abs(form.weights.alpha1.real) * resolved
        floor = FLOOR_SAFETY * float(np.finfo(float).eps) * np.asarray(magnitudes[1] @ built).ravel()
        own = ADMISSION_BAR * scale
        # A HOMOGENEOUS row (``u_i = 0`` beside a coupled system) holds only the solve's rounding, and its
        # own bar ``ADMISSION_BAR |u_i|`` sits below its own residual for every nonzero iterate.
        own = _noise_judged(problem, form, u, own, scale, system_resolution(scale))
        self.bars = np.maximum(own, floor)
        self.judged, self.judged_rate = u.copy(), rate.copy()
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.where(self.bars > 0, residual / self.bars, np.where(residual > 0, np.inf, 0.0))
        row = int(np.argmax(ratios)) if ratios.size else 0
        owner = getattr(problem, 'row_owner', None)
        self.worst = RowVerdict(
            row=row,
            owner=None if owner is None else str(owner(row)),
            residual=float(residual[row]) if residual.size else 0.0,
            bar=float(self.bars[row]) if residual.size else 0.0,
            own_bar=float(own[row]) if residual.size else 0.0,
            floor=float(floor[row]) if residual.size else 0.0,
            scale=float(scale[row]) if residual.size else 0.0,
        )
        return bool(np.all(residual <= self.bars))


def _near(value, judged, relative: float) -> bool:
    """``value`` within ``relative`` of ``judged`` in their own max norm (a zero vector is near only to itself)."""
    return bool(np.max(np.abs(value - judged), initial=0.0) <= relative * np.max(np.abs(value), initial=0.0))


def _locally_stable(local: LocalIterationCheck, u, t, observed, tangent) -> bool:
    """The local check's verdict, or a change the residual's own arithmetic cannot resolve.

    A relative per-field bar judges a physically ZERO field against its own rounding noise and
    can never pass there; :func:`change_below_residual_resolution` asks the unit-safe question
    instead -- whether the change moved any residual row above that row's rounding floor. Both
    verdicts need a previous observation, so the first iteration is decided by the check alone.
    Unknown source homogeneity keeps each row's own arithmetic scale.
    """
    stable = local.observe(u, t)
    if stable or observed is None or tangent is None or not local.finite:
        return stable
    return change_below_residual_resolution(tangent, observed, u, None)


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
    row_report: list | None = None,
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
    ``max_newton`` and was passed by the arithmetic floor rather than by ``tol``/``tol_rel``
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
            candidate, converged, *_ = outcome
            # A native solve is speculative too: its norm verdict cannot
            # replace an independently owned affine-plane admission. Decline
            # an off-plane result and solve normally from the original guess.
            if not converged or _affine_admitted(problem, form, candidate, capabilities):
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
    polish_slow = 0
    capped = False
    # The last tangent BUILT (never rebuilt for this purpose): the exhaustion check below needs
    # the matrix, not its norm, because the floor is a per-row ``|J| . |u|``. On a factor-cache
    # HIT no matrix is built, so the cache's own -- valid by its contract that equal keys imply an
    # equal jacobian -- stands in, and a hit still never pays a build.
    tangent = None
    local = LocalIterationCheck(problem, ADMISSION_BAR if row_acceptance else tol_rel)
    observed = None  # the iterate the local check last observed, for the resolution test

    rows = RowAcceptance(problem, form) if row_acceptance else None

    def met(residual, r, u):
        if rows is not None:
            return rows.accepts(u, r)
        return residual <= tol or (tol_rel is not None and residual <= tol_rel * residual_0)

    for _it in range(1, max_newton + 1):
        iterations = _it
        u_arg, u_dot, u_ddot = form.u_arg(u), form.u_dot(u), form.u_ddot(u)
        r = np.asarray(problem.residual(u_arg, u_dot, u_ddot, t), dtype=float)
        residual = float(np.linalg.norm(r))
        stable = _locally_stable(local, u, t, observed, tangent)
        observed = u.copy()
        local_ok = local.accepts(stable, residual, initial=_it == 1)
        local_ok = local_ok and _affine_admitted(problem, form, u, capabilities)
        if _it == 1:
            residual_0 = residual
        # The RELATIVE contract (MANUAL-628): the absolute tol is scale-blind -- a
        # residual that stalls at 1e-1 on a ~5400 A switch-current scale is 2e-5
        # relative, and the Newton cannot reach 1e-12 there by SCALE, not by error.
        # The opt-in ``tol_rel`` accepts a step whose residual dropped below
        # ``tol_rel * r0`` (r0 = the first iteration's norm). Off by default: every
        # existing march keeps the absolute contract unchanged.
        # ROW ACCEPTANCE ADMITS ONLY A SOLVED ITERATE: its bar scales with |F_u| |u|, the STATE's size,
        # so a guess within 1e-10 of a settling state's equilibrium passes it un-solved and the march
        # stops decaying there (a downstream repo's voltage-driven bordered march, 2026-10-24: 30 of 170 checks
        # took the guess; the field froze 1.63e-9 short against a 1e-11 bar, 2.6e-13 once refused).
        if local_ok and (rows is None or _it > 1) and met(residual, r, u):
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
            # it (measured on case_a's chopper: 50 iterations, ~30 s a step, at |r| ~ 4e-9).
            if rows is None and local_ok and _at_roundoff_floor(problem, tangent, u, form, capabilities, r):
                converged = True
                on_floor = True
                break
            # THE POLISH MUST BEAT THE NEWTON IT REPLACED, not merely lower the residual: an alternative
            # that only contracts linearly (two iterations above HELD_STALL_RATIO, the held factor's own
            # bar) is a chord, and the exact Newton it displaced finishes quadratically. Measured on
            # case_i's current loop at t = 12.4 ms: the secant crept at 0.88 to the 50-iteration
            # ceiling, while one exact step from its iterate took 6.7e-5 -> 4.8e-10.
            polish_slow = polish_slow + 1 if (tangent_active and ratio > HELD_STALL_RATIO) else 0
            if polish_slow >= 2:
                tangent_active, polish_spent, slow, polish_slow = False, True, 0, 0
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
            limit = residual if backtrack_band is None else backtrack_band * residual

            def trial_at(lam, u=u, step=step):
                u_try = project_trial(u + lam * step)
                try:
                    r_try = problem.residual(form.u_arg(u_try), form.u_dot(u_try), form.u_ddot(u_try), t)
                except TrialNotEvaluable:
                    # The problem cannot evaluate this trial (an inner solve without a root): the
                    # trial is REJECTED exactly like one whose residual is too large -- halve.
                    return None
                return u_try, float(np.linalg.norm(np.asarray(r_try, dtype=float))), r_try

            def accepts_trial(_lam, trial):
                # An already floor-limited row can mask progress on another
                # equation in the global norm. Select a trial satisfying the
                # SAME all-row floor used by convergence, not a larger bar.
                # The next iteration still owns local-field/convergence checks.
                # UNDER THE ROW VERDICT the norm is not the merit: a large-scale row's rounding can
                # dominate it and hide a small-scale row's real progress (a downstream repo's hysteresis chopper,
                # 2026-10-25: a 4e7-scale circuit row at 1e-9 noise over a 0.6 Wb linkage row; every full
                # step was halved to the cap and the stage exhausted). A trial the row verdict accepts is
                # the answer, whatever its norm.
                if rows is not None:
                    return trial[1] < limit or rows.accepts(trial[0], trial[2])
                return trial[1] < limit or _at_roundoff_floor(problem, tangent, trial[0], form, capabilities, trial[2])

            search = backtracking(trial_at, accepts_trial, tries=backtrack_cap)
            raised = search.accepted and search.trial[1] >= residual
            if search.accepted:
                u = search.trial[0]
            else:
                # Exhausted: take the next, untried step only if the last trial could be evaluated at all.
                if search.trial is not None:
                    u = project_trial(u + search.step * step)
                capped = True
            if tangent_active and (capped or raised):
                # THE POLISH IS A TRIAL, NOT A ONE-WAY SWITCH: an alternative operator whose step
                # does not LOWER the residual has done no better than the direction it replaced, so
                # the default source takes back over and the polish is spent for this step. The
                # test is a strict decrease and not the tolerant band the march accepts a default
                # step within: under that band a bad alternative is never capped, only accepted
                # while the residual climbs. Measured on `case_a --point rated` (vacuum B-H tail):
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
        local_ok = _locally_stable(local, u, t, observed, tangent)
        local_ok = local_ok and _affine_admitted(problem, form, u, capabilities)
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
            # problem declares one, :func:`residual_floor_by_row` derives ``eps * |J| . |u|`` from the
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
    if row_report is not None and rows is not None and rows.worst is not None:
        # The last row verdict read: on a refusal, the row that refused it (``RowVerdict``).
        row_report.append(rows.worst)
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
    worst_row: RowVerdict | None = None


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
    report: list = []
    if solve_kw.get('row_acceptance'):
        solve_kw = {**solve_kw, 'row_report': report}
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
    return StepOutcome(
        state,
        u,
        converged,
        iterations,
        worst,
        on_floor,
        form.u_dot(u),
        form.t_eval,
        candidate,
        worst_row=max(report, key=lambda verdict: verdict.ratio) if report else None,
    )
