"""Physics-blind time-integration runtime -- BDF / Newmark / generalized-alpha (plan R5).

This is the temporal counterpart of ``gauss/nonlinear`` (iteration strategies) and
``gauss/linear`` (linear-solve backends): a registry of pluggable TIME INTEGRATORS that a
transient study marches with, decoupled from any physics. The orthogonality hinge is
:class:`~motronics.contracts.analysis_formulation.DerivativeWeights`: an analysis formulation
supplies ONLY the triple ``(alpha0, alpha1, alpha2)`` weighting the ``u`` / ``u_dot`` /
``u_ddot`` operators; a time integrator is exactly the machinery that PRODUCES those weights
per step (plus the history-dependent affine offsets) and advances a
:class:`~motronics.contracts.analysis_formulation.SemiDiscreteProblem` one step.

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
``test_architecture_layering::test_gauss_sources_import_no_physics_package``); it touches no
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
from scipy.sparse.linalg import spsolve

from motronics.contracts.analysis_formulation import DerivativeWeights, SemiDiscreteProblem
from motronics.contracts.study_spec import IntegratorKind, TransientStudy
from motronics.gauss.time_integrators.factor_reuse import FactorCache

logger = logging.getLogger(__name__)

__all__ = [
    'INTEGRATOR_REGISTRY',
    'BDF1Integrator',
    'BDF2Integrator',
    'FactorCache',
    'GeneralizedAlphaIntegrator',
    'ITimeIntegrator',
    'IntegratorState',
    'NewmarkIntegrator',
    'StepForm',
    'TransientSolution',
    'get_integrator',
    'integrate_transient',
]


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


def _zeros_like(u: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.zeros_like(np.asarray(u, dtype=float))


class BDF1Integrator:
    """Backward Euler (BDF1) -- first order, unconditionally stable, for the ``u_dot`` (eddy) form.

    ``u_dot = (u_{n+1} - u_n) / dt`` gives the current-step slope ``alpha1 = 1/dt`` and history
    ``h_dot = -u_n/dt``; ``alpha2 = 0`` (no ``u_ddot`` term). One implicit solve per step; the
    local truncation error is ``O(dt^2)`` so the global error is ``O(dt)`` (verified by the
    analytic convergence test).
    """

    name = 'bdf1'
    order = 1

    def weights(self, dt: float) -> DerivativeWeights:
        return DerivativeWeights(alpha0=1.0, alpha1=1.0 / dt)

    def start(self, u0, t0, v0=None, a0=None) -> IntegratorState:
        return IntegratorState(us=[np.asarray(u0, dtype=float)], t=t0)

    def prepare(self, state: IntegratorState, dt: float, t_new: float) -> StepForm:
        u_n = state.last_u
        return StepForm(
            weights=self.weights(dt),
            h_u=_zeros_like(u_n),
            h_dot=-u_n / dt,
            h_ddot=_zeros_like(u_n),
            t_eval=t_new,
        )

    def advance(self, state: IntegratorState, u_new, dt: float, t_new: float) -> IntegratorState:
        return IntegratorState(us=[np.asarray(u_new, dtype=float)], t=t_new)


class BDF2Integrator:
    """Second-order backward difference (BDF2) -- ``u_dot = (3u_{n+1} - 4u_n + u_{n-1})/(2 dt)``.

    Slope ``alpha1 = 3/(2 dt)``, history ``h_dot = (-4u_n + u_{n-1})/(2 dt)``; ``alpha2 = 0``.
    Uniform-step BDF2 needs two past states, so the FIRST step (only ``u_0`` known) falls back to
    a single BDF1 step -- the standard bootstrap, which keeps the scheme second order globally.
    """

    name = 'bdf2'
    order = 2

    def __init__(self) -> None:
        self._bdf1 = BDF1Integrator()

    def weights(self, dt: float) -> DerivativeWeights:
        return DerivativeWeights(alpha0=1.0, alpha1=1.5 / dt)

    def start(self, u0, t0, v0=None, a0=None) -> IntegratorState:
        return IntegratorState(us=[np.asarray(u0, dtype=float)], t=t0)

    def prepare(self, state: IntegratorState, dt: float, t_new: float) -> StepForm:
        if len(state.us) < 2:  # bootstrap: not enough history yet -> one BDF1 step
            return self._bdf1.prepare(state, dt, t_new)
        u_n, u_nm1 = state.us[-1], state.us[-2]
        return StepForm(
            weights=self.weights(dt),
            h_u=_zeros_like(u_n),
            h_dot=(-4.0 * u_n + u_nm1) / (2.0 * dt),
            h_ddot=_zeros_like(u_n),
            t_eval=t_new,
        )

    def advance(self, state: IntegratorState, u_new, dt: float, t_new: float) -> IntegratorState:
        us = [*state.us, np.asarray(u_new, dtype=float)][-2:]  # keep only the last two states
        return IntegratorState(us=us, t=t_new)


# Newmark defaults: average-acceleration (unconditionally stable, second order). Match the
# contracts formulation_for() defaults so the study-level and integrator-level weights agree.
_NEWMARK_BETA = 0.25
_NEWMARK_GAMMA = 0.5


class NewmarkIntegrator:
    """Newmark-beta for the second-order (structural ``u_ddot``) system ``M ddot + C dot + K u = f``.

    Given ``(u_n, v_n, a_n)`` the Newmark relations make ``a_{n+1}`` and ``v_{n+1}`` affine in the
    unknown ``u_{n+1}``:

        a_{n+1} = (u_{n+1} - u_n - dt v_n - dt^2/2 (1-2 beta) a_n) / (beta dt^2)
        v_{n+1} = v_n + dt[(1-gamma) a_n + gamma a_{n+1}]

    so ``alpha2 = 1/(beta dt^2)`` and ``alpha1 = gamma/(beta dt)`` (the average-acceleration
    default gives ``alpha1 = 2/dt``, ``alpha2 = 4/dt^2``). One implicit solve per step; the
    driver folds the new ``v`` / ``a`` back into the history in :meth:`advance`.
    """

    name = 'newmark'
    order = 2

    def __init__(self, beta: float = _NEWMARK_BETA, gamma: float = _NEWMARK_GAMMA) -> None:
        self.beta = beta
        self.gamma = gamma

    def weights(self, dt: float) -> DerivativeWeights:
        return DerivativeWeights(
            alpha0=1.0,
            alpha1=self.gamma / (self.beta * dt),
            alpha2=1.0 / (self.beta * dt * dt),
        )

    def start(self, u0, t0, v0=None, a0=None) -> IntegratorState:
        u0 = np.asarray(u0, dtype=float)
        v0 = _zeros_like(u0) if v0 is None else np.asarray(v0, dtype=float)
        a0 = _zeros_like(u0) if a0 is None else np.asarray(a0, dtype=float)
        return IntegratorState(us=[u0], t=t0, v=v0, a=a0)

    def prepare(self, state: IntegratorState, dt: float, t_new: float) -> StepForm:
        beta, gamma = self.beta, self.gamma
        u_n, v_n, a_n = state.last_u, state.v, state.a
        # a_{n+1} = alpha2 u_{n+1} + h_ddot
        h_ddot = -(u_n + dt * v_n + 0.5 * dt * dt * (1.0 - 2.0 * beta) * a_n) / (beta * dt * dt)
        # v_{n+1} = v_n + dt(1-gamma) a_n + dt gamma a_{n+1} = alpha1 u_{n+1} + h_dot
        h_dot = v_n + dt * (1.0 - gamma) * a_n + dt * gamma * h_ddot
        return StepForm(weights=self.weights(dt), h_u=_zeros_like(u_n), h_dot=h_dot, h_ddot=h_ddot, t_eval=t_new)

    def advance(self, state: IntegratorState, u_new, dt: float, t_new: float) -> IntegratorState:
        u_new = np.asarray(u_new, dtype=float)
        w = self.weights(dt)
        form = self.prepare(state, dt, t_new)
        a_new = w.alpha2.real * u_new + form.h_ddot
        v_new = w.alpha1.real * u_new + form.h_dot
        return IntegratorState(us=[u_new], t=t_new, v=v_new, a=a_new)


class GeneralizedAlphaIntegrator:
    """Chung-Hulbert generalized-alpha -- second order with controllable high-frequency damping.

    The residual is evaluated at the intermediate states ``u_{n+1-alpha_f}``, ``v_{n+1-alpha_f}``,
    ``a_{n+1-alpha_m}`` (each affine in the unknown ``u_{n+1}``, via the same Newmark relations),
    and the time is sampled at ``t_{n+1} - alpha_f dt``. The effective jacobian weights are

        alpha0 = 1 - alpha_f
        alpha1 = (1 - alpha_f) gamma / (beta dt)
        alpha2 = (1 - alpha_m) / (beta dt^2)

    with ``alpha_m = (2 rho - 1)/(rho + 1)``, ``alpha_f = rho/(rho + 1)``,
    ``beta = 1/4 (1 - alpha_m + alpha_f)^2``, ``gamma = 1/2 - alpha_m + alpha_f`` for a chosen
    spectral radius ``rho = rho_inf in [0, 1]`` (``rho = 1`` recovers undamped Newmark
    average-acceleration; ``rho = 0`` gives asymptotic annihilation). Fully driven -- the
    intermediate-state evaluation makes the step exact for a linear tangent.
    """

    name = 'generalized_alpha'
    order = 2

    def __init__(self, rho_inf: float = 0.8) -> None:
        if not 0.0 <= rho_inf <= 1.0:
            msg = f'generalized-alpha spectral radius rho_inf must be in [0, 1], got {rho_inf}'
            raise ValueError(msg)
        self.rho_inf = rho_inf
        self.alpha_m = (2.0 * rho_inf - 1.0) / (rho_inf + 1.0)
        self.alpha_f = rho_inf / (rho_inf + 1.0)
        self.beta = 0.25 * (1.0 - self.alpha_m + self.alpha_f) ** 2
        self.gamma = 0.5 - self.alpha_m + self.alpha_f

    def weights(self, dt: float) -> DerivativeWeights:
        return DerivativeWeights(
            alpha0=1.0 - self.alpha_f,
            alpha1=(1.0 - self.alpha_f) * self.gamma / (self.beta * dt),
            alpha2=(1.0 - self.alpha_m) / (self.beta * dt * dt),
        )

    def start(self, u0, t0, v0=None, a0=None) -> IntegratorState:
        u0 = np.asarray(u0, dtype=float)
        v0 = _zeros_like(u0) if v0 is None else np.asarray(v0, dtype=float)
        a0 = _zeros_like(u0) if a0 is None else np.asarray(a0, dtype=float)
        return IntegratorState(us=[u0], t=t0, v=v0, a=a0)

    def _newmark_form(self, state: IntegratorState, dt: float):
        """The Newmark end-point slopes/offsets for ``a_{n+1}`` and ``v_{n+1}`` (shared machinery)."""
        beta, gamma = self.beta, self.gamma
        u_n, v_n, a_n = state.last_u, state.v, state.a
        a_slope = 1.0 / (beta * dt * dt)
        a_off = -(u_n + dt * v_n + 0.5 * dt * dt * (1.0 - 2.0 * beta) * a_n) / (beta * dt * dt)
        v_slope = gamma / (beta * dt)
        v_off = v_n + dt * (1.0 - gamma) * a_n + dt * gamma * a_off
        return a_slope, a_off, v_slope, v_off

    def prepare(self, state: IntegratorState, dt: float, t_new: float) -> StepForm:
        af, am = self.alpha_f, self.alpha_m
        u_n, v_n, a_n = state.last_u, state.v, state.a
        _a_slope, a_off, _v_slope, v_off = self._newmark_form(state, dt)
        # Intermediate (alpha-averaged) states, each affine in the unknown u_{n+1}.
        # u_{n+1-af} = (1-af) u_{n+1} + af u_n
        h_u = af * u_n
        # v_{n+1-af} = (1-af) v_{n+1} + af v_n
        h_dot = (1.0 - af) * v_off + af * v_n
        # a_{n+1-am} = (1-am) a_{n+1} + am a_n
        h_ddot = (1.0 - am) * a_off + am * a_n
        return StepForm(weights=self.weights(dt), h_u=h_u, h_dot=h_dot, h_ddot=h_ddot, t_eval=t_new - af * dt)

    def advance(self, state: IntegratorState, u_new, dt: float, t_new: float) -> IntegratorState:
        u_new = np.asarray(u_new, dtype=float)
        a_slope, a_off, v_slope, v_off = self._newmark_form(state, dt)
        a_new = a_slope * u_new + a_off
        v_new = v_slope * u_new + v_off
        return IntegratorState(us=[u_new], t=t_new, v=v_new, a=a_new)


INTEGRATOR_REGISTRY: dict[IntegratorKind, ITimeIntegrator] = {
    IntegratorKind.BDF1: BDF1Integrator(),
    IntegratorKind.BDF2: BDF2Integrator(),
    IntegratorKind.NEWMARK: NewmarkIntegrator(),
    IntegratorKind.GENERALIZED_ALPHA: GeneralizedAlphaIntegrator(),
}


def get_integrator(kind: IntegratorKind) -> ITimeIntegrator:
    """Resolve an :class:`IntegratorKind` to its :class:`ITimeIntegrator` implementation."""
    try:
        return INTEGRATOR_REGISTRY[kind]
    except KeyError as e:
        msg = f'unknown integrator {kind!r}; known: {sorted(k.value for k in INTEGRATOR_REGISTRY)}'
        raise KeyError(msg) from e


@dataclass
class TransientSolution:
    """The time march: parallel ``times`` and ``states`` including the initial condition.

    The CONVERGENCE CONTRACT (the FVS plan's near-term item): a march that exhausted its
    per-step Newton ceiling somewhere is LABELLED, never silent -- the worst step residual
    and the offending step times ride the solution, and :meth:`converged` is the machine-
    readable tag. The structured warning is emitted by the marcher at the first exhaustion
    (AGENTS.md: a log is a signal only when the CALLER can tell -- the caller reads the tag).
    """

    times: list[float] = field(default_factory=list)
    states: list[NDArray[np.floating]] = field(default_factory=list)
    worst_step_residual: float = 0.0
    nonconverged_times: list[float] = field(default_factory=list)

    @property
    def converged(self) -> bool:
        """The accuracy tag: every step met the per-step Newton tolerance."""
        return not self.nonconverged_times


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


def _default_solve(matrix: object, rhs: NDArray[np.floating]) -> NDArray[np.floating]:
    """Solve ``matrix @ x = rhs`` for a dense ndarray or a scipy-sparse jacobian (physics-blind)."""
    b = np.asarray(rhs, dtype=float)
    if sp.issparse(matrix):
        return np.asarray(spsolve(sp.csc_matrix(matrix), b))
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
    jacobian_tangent: Callable | None = None,
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

    Returns ``(u, converged, iterations, residual)`` -- the CONVERGENCE CONTRACT: the caller
    (the marcher) labels an exhausted step rather than taking it silently (the FVS plan's
    replacement for the old silent accept; the DC-op precedent is :class:`OpNonConvergence`).
    """
    key_fn = getattr(problem, 'factor_reuse_key', None) if factor_cache is not None else None
    # The stall-polish's phase-2 source (MANUAL-627 (i)): the problem's local-derivative
    # jacobian, engaged when the default jacobian's per-iteration creep marks a stalled
    # direction. The default ``None`` reads the problem's optional ``jacobian_tangent``
    # seam; a problem without it keeps the single-source Newton unchanged.
    tangent_source = jacobian_tangent if jacobian_tangent is not None else getattr(problem, 'jacobian_tangent', None)
    u = np.asarray(guess, dtype=float).copy()
    t = form.t_eval
    converged = False
    iterations = 0
    residual = float('inf')
    residual_0 = float('inf')
    prev_residual = None
    slow = 0
    tangent_active = False
    capped = False
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
            slow = slow + 1 if (capped or ratio > _STALL_RATIO) else 0
            if slow >= 2 and tangent_source is not None:
                tangent_active = True
        prev_residual = residual
        capped = False
        if tangent_active:
            jac = tangent_source(u_arg, u_dot, u_ddot, t, form.weights)
            du = np.asarray(solve(jac, -r), dtype=float)
        elif key_fn is None:
            jac = problem.jacobian(u_arg, u_dot, u_ddot, t, form.weights)
            du = np.asarray(solve(jac, -r), dtype=float)
        else:
            key = key_fn(u_arg, u_dot, u_ddot, t, form.weights)
            du = np.asarray(
                factor_cache.solve(key, lambda: problem.jacobian(u_arg, u_dot, u_ddot, t, form.weights), -r),
                dtype=float,
            )
        step = du.reshape(u.shape)
        if max_newton > 1 and backtrack_cap > 0:
            for _ in range(backtrack_cap):
                u_try = u + step
                u_arg, u_dot, u_ddot = form.u_arg(u_try), form.u_dot(u_try), form.u_ddot(u_try)
                r_try = np.asarray(problem.residual(u_arg, u_dot, u_ddot, t), dtype=float)
                limit = residual if backtrack_band is None else backtrack_band * residual
                if float(np.linalg.norm(r_try)) < limit:
                    u = u_try
                    break
                step = 0.5 * step
            else:
                u = u + step
                capped = True
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
    return u, converged, iterations, residual


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
    adaptive: bool = False,
    atol: float = 1e-6,
    h_min: float | None = None,
    h_max: float | None = None,
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
    if factor_cache is None and callable(getattr(problem, 'factor_reuse_key', None)):
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
            # The convergence threshold's FLOAT FLOOR: at the tiny steps the variable form's
            # history offsets cancel at the 1/h scale, so the residual's noise floor is
            # ~eps*u/h -- a 1e-12 threshold is unreachable at h = 1e-6 (measured 2026-08-07:
            # every small step was labelled non-converged by the noise). Genuine failures sit
            # well above the floor; the label keeps its meaning.
            u_new, converged, _it, residual = _solve_step(
                problem,
                form,
                state.last_u,
                solve=solve,
                max_newton=max_newton,
                tol=max(tol, 1e-10),
                tol_rel=tol_rel,
                factor_cache=factor_cache,
                backtrack_band=backtrack_band,
                backtrack_cap=backtrack_cap,
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
                state = integrator.advance(state, u_new, h, t_new)
                solution.times.append(t_new)
                solution.states.append(u_new)
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
        u_new, converged, _iterations, residual = _solve_step(
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
        )
        solution.worst_step_residual = max(solution.worst_step_residual, residual)
        if not converged:
            solution.nonconverged_times.append(t_new)
            # The CONVERGENCE CONTRACT, label-don't-block (AGENTS.md first principle): the
            # march continues, the result carries the measured deviation, and the caller
            # reads the tag -- a silent accept is the forbidden shape. One warning per
            # exhaustion event, carrying the measured residual.
            logger.warning(
                'step at t=%s did not converge in %s Newton iteration(s): residual %s > tol %s',
                t_new,
                max_newton,
                f'{residual:.3e}',
                f'{tol:.3e}',
            )
        state = integrator.advance(state, u_new, dt, t_new)
        solution.times.append(t_new)
        solution.states.append(u_new)
    return solution
