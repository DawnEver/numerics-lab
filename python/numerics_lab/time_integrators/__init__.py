"""Physics-blind time-integration runtime -- BDF / Newmark / generalized-alpha (plan R5).

This is the temporal counterpart of ``gauss/nonlinear`` (iteration strategies) and
``gauss/linear`` (linear-solve backends): a registry of pluggable TIME INTEGRATORS that a
transient study marches with, decoupled from any physics. The orthogonality hinge is
:class:`~motronics.contracts.spec.analysis_formulation.DerivativeWeights`: an analysis formulation
supplies ONLY the triple ``(alpha0, alpha1, alpha2)`` weighting the ``u`` / ``u_dot`` /
``u_ddot`` operators; a time integrator is exactly the machinery that PRODUCES those weights
per step (plus the history-dependent affine offsets) and advances a
:class:`~motronics.contracts.spec.analysis_formulation.SemiDiscreteProblem` one step.

The load-bearing idea: an implicit integrator writes each derivative at the new step as an
AFFINE function of the unknown ``u_{n+1}``,

    u_dot   = alpha1 * u_{n+1} + h_dot        (history)
    u_ddot  = alpha2 * u_{n+1} + h_ddot

so the residual root-find for ``u_{n+1}`` is driven purely through ``residual`` /
``jacobian(weights)`` -- the integrator never sees the physics, only these numbers. The
``alpha1`` / ``alpha2`` slopes ARE the :class:`DerivativeWeights` a study would carry; the
``h_dot`` / ``h_ddot`` offsets are the history the backend cannot know. First-order schemes
(BDF1/BDF2) leave ``alpha2 = 0`` (the eddy-current / diffusion form ``u_dot``); second-order
schemes (Newmark, generalized-alpha) carry ``alpha2`` (the structural ``u_ddot``).

This module imports ONLY contracts + numpy/scipy (enforced by
``test_import_purity.py::test_importing_gauss_pulls_no_physics_into_sys_modules``); it touches no
mesh, no material, no solver.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from motronics.contracts.spec.analysis_formulation import (
    AlternativeJacobianProblem,
    DerivativeWeights,
    FactorReusableProblem,
    HistoryProblem,
    ResidualFloorProblem,
    SemiDiscreteProblem,
)
from motronics.contracts.spec.study_spec import TransientStudy
from motronics.gauss.linear import reject_nonfinite_solution, solve_linear_system
from motronics.gauss.time_integrators.factor_reuse import FactorCache, apply_factor, factor_matrix

logger = logging.getLogger(__name__)

__all__ = [
    'HELD_STALL_RATIO',
    'INTEGRATOR_REGISTRY',
    'BDF1Integrator',
    'BDF2Integrator',
    'FactorCache',
    'GeneralizedAlphaIntegrator',
    'HeldFactor',
    'ITimeIntegrator',
    'IntegratorState',
    'NewmarkIntegrator',
    'StepForm',
    'TransientSolution',
    'bdf2_variable_form',
    'get_integrator',
    'integrate_transient',
    'residual_floor',
    'residual_floor_of_magnitude',
]


#: Safety factor on the estimated roundoff floor, DELIBERATELY the same 64 as the DC operating
#: point's ``laplace.circuit.core.mna_solves._FLOOR_SAFETY``, and for the same stated reason: the
#: estimate bounds the rounding error of ONE row sum, while the accumulated error over a row of
#: ``n`` assembled terms grows like ``sqrt(n)`` at best and ``n`` at worst. 64 buys that headroom
#: and still sits ORDERS below any residual a stalled-but-wrong Newton shows -- measured
#: 2026-09-12 at 1e8x on the planted stall, against a 64x allowance.
_FLOOR_SAFETY = 64.0

#: The machine epsilon the floor is written in (double precision -- the marcher is ``float64``).
_EPS = float(np.finfo(float).eps)


def residual_floor(tangent, u: NDArray[np.floating]) -> float:
    """The smallest residual norm this system can express at double precision, in RESIDUAL units.

    THE SAME CRITERION the DC operating point already uses -- ``mna_solves._roundoff_floor`` /
    ``CircuitDAE.op_residual_floor`` -- written here in the norm THIS layer measures its residual
    in. The residual is a row sum of signed contributions ``sum_j J_ij u_j``; summing terms of
    magnitude ``M`` in floating point cannot resolve below ``eps * M``, so ``(|J| . |u|)_i`` is
    the magnitude per row and ``eps`` times it is the smallest residual that row can MEAN. The op
    solve compares ``max|R|`` and so takes the row MAX; this marcher compares ``||R||_2`` and so
    takes the same vector's 2-norm. One criterion, each layer in its own norm.

    THE UNIT. It carries the residual's own units, because ``|J| . |u|`` does -- amperes on a KCL
    row, volts on a branch row. That is what the ABSOLUTE ``tol = 1e-12`` could not do, and
    ``rem/porting.md`` names that constant class as the one that cannot survive a change of scale.
    MEASURED 2026-09-12 on a linear series R-L marched by BDF2: the exact one-iteration solve
    lands at 6.17e-13 on a 24 V supply and 4.95e-11 on a 2400 V one -- the SAME solve, and only
    the second one was reported as having exhausted its Newton ceiling. The op solve's own table
    records the same shape from the other side: one circuit's floor spans 56x across its four
    gate states, so no single constant can sit above the lowest and below the highest.

    COMPUTED ONLY ON EXHAUSTION, never per iteration -- the DC op's rule, for its reason: it costs
    an ``|J| . |u|`` matvec, and it can only ever change the verdict where the plain bars have
    already been missed. So it alters no iteration count and no trajectory: strictly the LABEL.
    """
    u_abs = np.abs(np.asarray(u, dtype=float)).reshape(-1)
    if sp.issparse(tangent):
        # NEVER `.toarray()` here. `op_residual_floor` may densify because an operating point is a
        # circuit; this marcher also carries FEA/MEC tangents, where a dense copy is the whole mesh
        # squared. `abs()` keeps the sparse storage and the matvec stays O(nnz).
        magnitude = abs(tangent) @ u_abs
    else:
        dense = np.abs(np.asarray(tangent, dtype=float))
        magnitude = dense * u_abs if dense.ndim == 0 else dense @ u_abs
    return residual_floor_of_magnitude(magnitude)


def residual_floor_of_magnitude(magnitude: NDArray[np.floating]) -> float:
    """THE FLOOR RULE ITSELF, for a caller that builds ``|J| . |u|`` without ever forming ``J``.

    :func:`residual_floor` is this function applied to ``|J| . |u|``; the bordered field<->circuit
    system (:mod:`motronics.hamilton.coupling.monolithic`) can supply the magnitude but NOT the
    matrix -- ``J`` there is a 2x2 block system whose ``(n1, n1)`` corner is a 52185-DOF sparse
    field tangent, so an assembly is the defect that module exists to avoid. It builds the stacked
    magnitude from its four blocks and hands it here.

    ONE HOME FOR ``_FLOOR_SAFETY * _EPS * ||.|``, because a second spelling of it is exactly the
    drift :data:`~motronics.hamilton.coupling.monolithic._BACKTRACK_CAP`'s comment warns about: a
    criterion written twice is two criteria as soon as one of them is edited.
    """
    magnitude = np.asarray(magnitude, dtype=float).reshape(-1)
    return _FLOOR_SAFETY * _EPS * float(np.linalg.norm(magnitude)) if magnitude.size else 0.0


@dataclass
class IntegratorState:
    """The marching history an integrator carries between steps.

    ``us`` holds past state vectors most-recent-LAST (BDF2 reads the last two); ``v`` / ``a`` are
    the velocity ``u_dot`` and acceleration ``u_ddot`` at the last step (the second-order schemes
    propagate them, the first-order schemes leave them ``None``). ``t`` is the last step's time.
    """

    us: list[NDArray[np.floating]]
    t: float
    v: NDArray[np.floating] | None = None
    a: NDArray[np.floating] | None = None

    @property
    def last_u(self) -> NDArray[np.floating]:
        return self.us[-1]


@dataclass(frozen=True)
class StepForm:
    """The affine per-step form an integrator hands the driver for the unknown ``u_{n+1}``.

    Every implicit derivative is affine in the unknown: ``u_arg = alpha0*u + h_u`` (the state the
    residual is evaluated at -- the identity ``u`` for BDF/Newmark, an intermediate point for
    generalized-alpha), ``u_dot = alpha1*u + h_dot``, ``u_ddot = alpha2*u + h_ddot``. The slopes
    are exactly the :class:`DerivativeWeights` the jacobian consumes; the offsets are the history.
    ``t_eval`` is the time the residual/source is sampled at (shifted by ``alpha_f*dt`` for
    generalized-alpha).
    """

    weights: DerivativeWeights
    h_u: NDArray[np.floating]
    h_dot: NDArray[np.floating]
    h_ddot: NDArray[np.floating]
    t_eval: float

    def u_arg(self, u: NDArray[np.floating]) -> NDArray[np.floating]:
        return self.weights.alpha0.real * u + self.h_u

    def u_dot(self, u: NDArray[np.floating]) -> NDArray[np.floating]:
        return self.weights.alpha1.real * u + self.h_dot

    def u_ddot(self, u: NDArray[np.floating]) -> NDArray[np.floating]:
        return self.weights.alpha2.real * u + self.h_ddot


class ITimeIntegrator(Protocol):
    """The seam a time integrator implements -- mirrors ``gauss.nonlinear.IterationStrategy``.

    ``weights(dt)`` is the textbook steady-state ``(alpha0, alpha1, alpha2)`` triple (what an
    analysis formulation carries); ``start`` seeds the history; ``prepare`` produces the per-step
    :class:`StepForm` from the current history; ``advance`` folds the just-solved ``u_{n+1}`` back
    into the history (BDF pushes the state, the second-order schemes also update ``v`` / ``a``).
    """

    name: str
    order: int

    def weights(self, dt: float) -> DerivativeWeights: ...

    def start(
        self,
        u0: NDArray[np.floating],
        t0: float,
        v0: NDArray[np.floating] | None = None,
        a0: NDArray[np.floating] | None = None,
    ) -> IntegratorState: ...

    def prepare(self, state: IntegratorState, dt: float, t_new: float) -> StepForm: ...

    def advance(
        self, state: IntegratorState, u_new: NDArray[np.floating], dt: float, t_new: float
    ) -> IntegratorState: ...


#: THE CONCRETE INTEGRATORS AND THEIR REGISTRY live in `_integrators.py`, split 2026-08-22
#: because this file crossed the 800-line band. It imports :class:`IntegratorState`,
#: :class:`ITimeIntegrator` and :class:`StepForm` FROM THIS MODULE (a circular import that works
#: because they are defined above this import); the public API is unchanged.
from motronics.gauss.time_integrators._integrators import (  # noqa: E402
    INTEGRATOR_REGISTRY,
    BDF1Integrator,
    BDF2Integrator,
    GeneralizedAlphaIntegrator,
    NewmarkIntegrator,
    get_integrator,
)


@dataclass
class TransientSolution:
    """The time march: parallel ``times`` and ``states`` including the initial condition.

    The CONVERGENCE CONTRACT (the FVS plan's near-term item): a march that exhausted its
    per-step Newton ceiling somewhere is LABELLED, never silent -- the worst step residual
    and the offending step times ride the solution, and :meth:`converged` is the machine-
    readable tag. The structured warning is emitted by the marcher at the first exhaustion
    (AGENTS.md: a log is a signal only when the CALLER can tell -- the caller reads the tag).

    ``nonconverged_times`` names only steps that LANDED in ``times``. The eventful march also
    takes bisection probe steps that ``_rollback`` discards; an exhausted probe costs time and
    slightly degrades the located instant, but its error never enters the trajectory, so it is
    recorded separately in ``nonconverged_probe_times`` and does NOT clear :meth:`converged`.
    Conflating the two made the tag ~9x pessimistic on a switching march (srdab_closed_loop,
    2026-08-15: 2094 reported, 239 landed) -- always False wherever it mattered, so no caller
    could use it.
    """

    times: list[float] = field(default_factory=list)
    states: list[NDArray[np.floating]] = field(default_factory=list)
    worst_step_residual: float = 0.0
    nonconverged_times: list[float] = field(default_factory=list)
    #: Exhausted steps that were DISCARDED by an event rollback (bisection probes). Diagnostic
    #: only -- these never reached ``times``. A large count means the locator worked hard at the
    #: toggles, not that the answer is contaminated.
    nonconverged_probe_times: list[float] = field(default_factory=list)
    #: Steps whose ``converged`` verdict was carried by :func:`residual_floor` rather than by
    #: ``tol``/``tol_rel``: they EXHAUSTED the Newton ceiling and landed at the smallest
    #: residual double precision can express for that system. The floor is the correctly
    #: SCALED bar, so such a step IS converged and this list is not a failure record. It
    #: exists because without it the two verdicts are the same value -- a march that genuinely
    #: stopped converging, but whose residual sits at the arithmetic floor, reads
    #: ``converged=True`` with nothing to distinguish it, and a test asserting convergence
    #: cannot tell the physics from the arithmetic. Like ``worst_step_residual`` and unlike
    #: ``nonconverged_times``, this includes the eventful march's DISCARDED bisection probes:
    #: it is a diagnostic about the SOLVES, not a property of the trajectory.
    floor_converged_times: list[float] = field(default_factory=list)

    @property
    def converged(self) -> bool:
        """The accuracy tag: every step IN THIS SOLUTION met the per-step Newton tolerance."""
        return not self.nonconverged_times

    @property
    def converged_within_the_ceiling(self) -> bool:
        """The STRONGER tag: every step converged on its BAR, none of them on the floor.

        :attr:`converged` cannot distinguish the two, and that is the whole reason both exist:
        a caller that means "the physics converged" rather than "the arithmetic could say no
        more" asks for this one. A step ON the floor exhausted its ceiling, so this is also
        the tag to read when deciding whether ``max_newton`` is set high enough.
        """
        return self.converged and not self.floor_converged_times


def _roundoff_floor(problem, tangent, u: NDArray[np.floating], form) -> float | None:
    """The residual scale below which a converged solve cannot go, or ``None`` if unknowable.

    The problem's own `residual_roundoff_floor` is preferred: it is a statement about that
    system's cancellation, which no generic derivation can see. :func:`residual_floor` is the
    fallback and needs only the tangent. A non-finite reading from either is discarded rather
    than compared -- a floor that is not a number cannot bound anything.
    """
    if isinstance(problem, ResidualFloorProblem):
        # `None` IS AN ANSWER HERE -- "this problem cannot bound its own cancellation" -- and it
        # must fall through to the generic fallback rather than be accepted. It was not expressible
        # before: `float(...)` would have raised on it, so `mna_solves.transient_residual_floor`
        # said "no floor" by returning `0.0`, which is finite, was accepted, and suppressed the
        # fallback entirely while comparing every real residual against zero.
        declared = problem.residual_roundoff_floor(u, form)
        if declared is not None:
            floor = float(declared)
            if np.isfinite(floor):
                return floor
    if tangent is None:
        return None
    floor = float(residual_floor(tangent, u))
    return floor if np.isfinite(floor) else None


def _bdf2_variable_form(state: IntegratorState, dt: float, h: float, t_new: float) -> StepForm:
    """The variable-spacing BDF2 step form; ``h`` is the previous step's size.

    Fixed-weight BDF2 assumes uniform history spacing; a step whose size differs from its
    history's spacing needs the derivative of the quadratic through the unevenly spaced
    points (the events module docstring names the O(1) wrong-equation trap this closes).
    """
    u_n, u_nm1 = state.us[-1], state.us[-2]
    return StepForm(
        weights=DerivativeWeights(alpha0=1.0, alpha1=(2.0 * dt + h) / (dt * (dt + h))),
        h_u=np.zeros_like(u_n),
        h_dot=-(dt + h) / (dt * h) * u_n + dt / (h * (dt + h)) * u_nm1,
        h_ddot=np.zeros_like(u_n),
        t_eval=t_new,
    )


#: The public spelling of the variable-step BDF2 form, for a CHECK that must replay the march's own
#: weights (`hamilton.coupling.terminal_balance`). One function, two names, never two formulas.
bdf2_variable_form = _bdf2_variable_form


def _default_solve(matrix: object, rhs: NDArray[np.floating]) -> NDArray[np.floating]:
    """Solve ``matrix @ x = rhs`` for a dense ndarray or a scipy-sparse jacobian (physics-blind)."""
    b = np.asarray(rhs, dtype=float)
    if sp.issparse(matrix):
        return np.asarray(solve_linear_system(sp.csc_matrix(matrix), b))
    m = np.asarray(matrix, dtype=float)
    if m.ndim == 0:  # scalar jacobian (1-DOF problem written as a plain number)
        return b / float(m)
    return np.linalg.solve(m, b)


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
    :func:`~motronics.laplace.circuit.core.mna_solves._gmin_stepped` follows). Each stage is
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
    for gmin in gmin_stages:
        for _it in range(max(2, max_newton)):
            u_arg, u_dot, u_ddot = form.u_arg(u), form.u_dot(u), form.u_ddot(u)
            r = np.asarray(problem.residual(u_arg, u_dot, u_ddot, t), dtype=float)
            residual = float(np.linalg.norm(r))
            if residual <= tol or (tol_rel is not None and residual <= tol_rel * residual_0):
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
    if residual <= tol or (tol_rel is not None and residual <= tol_rel * residual_0):
        return u, True, residual, False
    if tangent is not None and residual <= residual_floor(tangent, u):
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
    """
    reusable = factor_cache is not None and isinstance(problem, FactorReusableProblem)
    # The stall-polish's phase-2 source (MANUAL-627 (i)): the problem's local-derivative
    # jacobian, engaged when the default jacobian's per-iteration creep marks a stalled
    # direction. The default ``None`` reads the problem's optional ``jacobian_tangent``
    # seam; a problem without it keeps the single-source Newton unchanged. The
    # ``has_tangent_jacobian`` flag (the laplace's channel-switch presence) gates the
    # ENGAGEMENT: a circuit whose tangent equals its default jacobian (the RON/ROFF
    # switches, the diodes -- their stamps already ARE the local derivative) would pay
    # the tangent phase's factor-cache bypass for nothing (measured: the buck's reuse
    # counts broke without the gate).
    alternative = problem if isinstance(problem, AlternativeJacobianProblem) else None
    tangent_source = (
        jacobian_tangent
        if jacobian_tangent is not None
        else (alternative.jacobian_tangent if alternative is not None else None)
    )
    tangent_helpful = jacobian_tangent is not None or (alternative is not None and alternative.has_tangent_jacobian)
    u = np.asarray(guess, dtype=float).copy()
    t = form.t_eval
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
    for _it in range(1, max_newton + 1):
        iterations = _it
        u_arg, u_dot, u_ddot = form.u_arg(u), form.u_dot(u), form.u_ddot(u)
        r = np.asarray(problem.residual(u_arg, u_dot, u_ddot, t), dtype=float)
        residual = float(np.linalg.norm(r))
        if _it == 1:
            residual_0 = residual
        # The RELATIVE contract (MANUAL-628): the absolute tol is scale-blind -- a
        # residual that stalls at 1e-1 on a ~5400 A switch-current scale is 2e-5
        # relative, and the Newton cannot reach 1e-12 there by SCALE, not by error.
        # The opt-in ``tol_rel`` accepts a step whose residual dropped below
        # ``tol_rel * r0`` (r0 = the first iteration's norm). Off by default: every
        # existing march keeps the absolute contract unchanged.
        if residual <= tol or (tol_rel is not None and residual <= tol_rel * residual_0):
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
            if (floor := _roundoff_floor(problem, tangent, u, form)) is not None and residual <= floor:
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
            for _ in range(backtrack_cap):
                u_try = u + step
                u_arg, u_dot, u_ddot = form.u_arg(u_try), form.u_dot(u_try), form.u_ddot(u_try)
                r_try = np.asarray(problem.residual(u_arg, u_dot, u_ddot, t), dtype=float)
                limit = residual if backtrack_band is None else backtrack_band * residual
                trial = float(np.linalg.norm(r_try))
                if trial < limit:
                    u = u_try
                    raised = trial >= residual
                    break
                step = 0.5 * step
            else:
                u = u + step
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
            u = u + step
    else:
        # The ceiling exhausted: re-evaluate the residual at the FINAL u (the last solve's
        # result). The loop's last check read the PRE-solve guess -- reporting that would
        # label a step that actually converged as exhausted, the contract's own lie.
        u_arg, u_dot, u_ddot = form.u_arg(u), form.u_dot(u), form.u_ddot(u)
        r = np.asarray(problem.residual(u_arg, u_dot, u_ddot, t), dtype=float)
        residual = float(np.linalg.norm(r))
        if residual <= tol or (tol_rel is not None and residual <= tol_rel * residual_0):
            converged = True
        elif (floor := _roundoff_floor(problem, tangent, u, form)) is not None and residual <= floor:
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


def integrate_transient(
    problem: SemiDiscreteProblem,
    study: TransientStudy,
    u0: NDArray[np.floating],
    *,
    v0: NDArray[np.floating] | None = None,
    a0: NDArray[np.floating] | None = None,
    solve=_default_solve,
    max_newton: int = 1,
    tol: float = 1e-12,
    tol_rel: float | None = None,
    factor_cache: FactorCache | None = None,
    backtrack_band: float = 1.5,
    backtrack_cap: int = 4,
    line_search: str = 'backtrack',
    gmin_stages: tuple[float, ...] | None = None,
    adaptive: bool = False,
    atol: float = 1e-6,
    h_min: float | None = None,
    h_max: float | None = None,
    refactor_every: int | None = None,
) -> TransientSolution:
    """March ``problem`` from ``study.t0`` to ``study.t1`` by ``study.dt`` with ``study.integrator``.

    Steps the selected :class:`ITimeIntegrator`: each step ``prepare`` yields the affine
    :class:`StepForm`, :func:`_solve_step` advances the residual/jacobian seam for ``u_{n+1}``, and
    ``advance`` folds it back into the history. ``v0`` / ``a0`` seed the second-order schemes
    (default zero). ``max_newton = 1`` is the exact linear step; raise it (or swap ``solve``) for a
    nonlinear problem. Returns the full march including the initial condition.

    ``factor_cache`` (W6) reuses one factorization per reuse key across the march (see
    :mod:`~motronics.gauss.time_integrators.factor_reuse`); the default ``None`` builds a fresh
    cache when the problem exposes ``factor_reuse_key`` and none otherwise, so a keyed problem
    reuses by default and an unkeyed one is untouched. Pass a cache in to READ the counters.
    """
    # The MODIFIED Newton (ruling 3A): one held factor across the whole march, or None (full Newton).
    held = HeldFactor(refactor_every) if refactor_every is not None else None
    if factor_cache is None and isinstance(problem, FactorReusableProblem):
        factor_cache = FactorCache()
    integrator = get_integrator(study.integrator)
    if adaptive and integrator.name != 'bdf2':
        msg = f'adaptive stepping is implemented for BDF2 only, got integrator {integrator.name}'
        raise ValueError(msg)
    u0 = np.asarray(u0, dtype=float)
    state = integrator.start(u0, study.t0, v0, a0)
    dt = study.dt
    if adaptive:
        # The ADAPTIVE march (W6, OTSF from the simba notes): the BDF2's local truncation
        # error is estimated by the difference between the BDF2 step and the linear
        # extrapolation from the history (the second-order difference is O(h^3), so the
        # controller h *= (atol/err)^(1/3) drives the LTE to atol). A rejected step halves
        # and retries; an accepted step's size is grown by the controller's own gain. The
        # event locator's bracket-and-bisect machinery is step-agnostic, so the adaptive
        # steps and the eventful march compose -- the events' own adaptive wiring is a
        # follow-up (the events.py cells are fixed-grid by construction).
        h = dt
        h_prev = dt  # the last ACCEPTED step's size (the history's spacing, for the LTE)
        h_lo = dt / 1000.0 if h_min is None else h_min
        h_hi = dt * 1000.0 if h_max is None else h_max
        t_end = study.t1
        solution = TransientSolution(times=[study.t0], states=[u0])
        while state.t < t_end - 1e-14:
            h = min(h, t_end - state.t)
            t_new = state.t + h
            # The variable-spacing BDF2 form whenever the step differs from the history's
            # spacing: the uniform prepare would differentiate against the WRONG history
            # (measured 2026-08-07: the first retry after a rejection was off by |du|, and
            # the controller spun at the floor chasing a phantom error).
            if len(state.us) >= 2 and h != h_prev:
                form = _bdf2_variable_form(state, h, h_prev, t_new)
            else:
                form = integrator.prepare(state, h, t_new)
            # RETIRED 2026-09-12: this call used to raise the bar to ``max(tol, 1e-10)`` for the
            # adaptive march ALONE, on the correct diagnosis that "at the tiny steps the variable
            # form's history offsets cancel at the 1/h scale, so the residual's noise floor is
            # ~eps*u/h". The diagnosis was right and the remedy was a NUMBER -- right at h=1e-6,
            # wrong at h=1e-9, and blind to the supply besides. :func:`residual_floor` IS that
            # noise floor computed, and ``|J|`` carries the 1/h weights, so the same statement now
            # holds at every h and every voltage AND on the fixed-grid path, which never had it.
            u_new, converged, _it, residual, on_floor = _solve_step(
                problem,
                form,
                state.last_u,
                solve=solve,
                max_newton=max_newton,
                tol=tol,
                tol_rel=tol_rel,
                factor_cache=factor_cache,
                backtrack_band=backtrack_band,
                backtrack_cap=backtrack_cap,
                gmin_stages=gmin_stages,
                held=held,
            )
            # The LTE: |u_new - u_extrap| where u_extrap is the linear continuation from the
            # last two states, SCALED by the spacing ratio h/h_prev -- the naive 2u1-u0 form
            # assumes h == h_prev and overstates the error by |du|*|1 - h/h_prev| on the
            # first retry after a rejection (measured 2026-08-07: the controller then spun
            # at the floor).
            if len(state.us) >= 2:
                u_prev = np.asarray(state.us[-2], dtype=float)
                u_extrap = state.last_u + (state.last_u - u_prev) * (h / h_prev)
                err = float(np.linalg.norm(u_new - u_extrap))
                err = max(err, 1e-300)
            else:
                err = 0.0  # the first step: no history to extrapolate from -- accept it
            if h <= h_lo or (converged and (err == 0.0 or err <= atol)):
                # accept the step -- at the h_lo floor REGARDLESS of the convergence (the
                # step is taken with the contract's label: the nonconverged_times + the
                # warning -- a step that cannot converge at the floor must not spin forever,
                # measured 2026-08-07: the retry loop at h_lo with conv=False never exited)
                solution.worst_step_residual = max(solution.worst_step_residual, residual)
                if not converged:
                    solution.nonconverged_times.append(t_new)
                elif on_floor:
                    # ONE spelling of this rule, shared with the fixed-grid march below: a step is
                    # recorded against exactly one of the two lists, because `on_floor` is only
                    # ever set where the floor MADE the verdict converged.
                    solution.floor_converged_times.append(t_new)
                state = integrator.advance(state, u_new, h, t_new)
                solution.times.append(t_new)
                solution.states.append(u_new)
                if isinstance(problem, HistoryProblem):
                    problem.commit_history(u_new, t_new)
                h_prev = h
                if err > 0.0:
                    h = min(h_hi, max(h_lo, h * min(2.0, 1.2 * (atol / err) ** (1.0 / 3.0))))
            else:
                # reject: halve and retry (bounded by h_lo -- at the floor the step is taken
                # with the contract's label rather than spun forever)
                h = max(h_lo, h / 2.0)
        return solution
    n_steps = round((study.t1 - study.t0) / dt)
    solution = TransientSolution(times=[study.t0], states=[u0])
    for k in range(n_steps):
        t_new = study.t0 + (k + 1) * dt
        form = integrator.prepare(state, dt, t_new)
        u_new, converged, _iterations, residual, on_floor = _solve_step(
            problem,
            form,
            state.last_u,
            solve=solve,
            max_newton=max_newton,
            tol=tol,
            tol_rel=tol_rel,
            factor_cache=factor_cache,
            backtrack_band=backtrack_band,
            backtrack_cap=backtrack_cap,
            line_search=line_search,
            gmin_stages=gmin_stages,
            held=held,
        )
        solution.worst_step_residual = max(solution.worst_step_residual, residual)
        if on_floor:
            solution.floor_converged_times.append(t_new)
        elif not converged:
            solution.nonconverged_times.append(t_new)
            # The CONVERGENCE CONTRACT, label-don't-block (AGENTS.md first principle): the
            # march continues, the result carries the measured deviation, and the caller
            # reads the tag -- a silent accept is the forbidden shape. One warning per
            # exhaustion event, carrying the measured residual.
            logger.warning(
                'step at t=%s did not converge in %s Newton iteration(s): residual %s exceeds '
                'both the absolute tol %s and the arithmetic floor of this system',
                t_new,
                max_newton,
                f'{residual:.3e}',
                f'{tol:.3e}',
            )
        state = integrator.advance(state, u_new, dt, t_new)
        solution.times.append(t_new)
        solution.states.append(u_new)
        # THE STEP JOINED THE TRAJECTORY, so its material history is the one the next step starts
        # from -- including a step labelled non-converged: the march continues from `u_new`, and a
        # history left at the previous step would be a trajectory the material never followed.
        if isinstance(problem, HistoryProblem):
            problem.commit_history(u_new, t_new)
    if held is not None:
        logger.info('held-factor march: factorizations=%d solves=%d', held.factorizations, held.solves)
    return solution
