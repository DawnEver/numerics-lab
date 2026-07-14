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

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse.linalg import spsolve

from motronics.contracts.analysis_formulation import DerivativeWeights, SemiDiscreteProblem
from motronics.contracts.study_spec import IntegratorKind, TransientStudy

__all__ = [
    'INTEGRATOR_REGISTRY',
    'BDF1Integrator',
    'BDF2Integrator',
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
    """The time march: parallel ``times`` and ``states`` including the initial condition."""

    times: list[float] = field(default_factory=list)
    states: list[NDArray[np.floating]] = field(default_factory=list)


def _default_solve(matrix: object, rhs: NDArray[np.floating]) -> NDArray[np.floating]:
    """Solve ``matrix @ x = rhs`` for a dense ndarray or a scipy-sparse jacobian (physics-blind)."""
    b = np.asarray(rhs, dtype=float)
    if sp.issparse(matrix):
        return np.asarray(spsolve(sp.csc_matrix(matrix), b))
    m = np.asarray(matrix, dtype=float)
    if m.ndim == 0:  # scalar jacobian (1-DOF problem written as a plain number)
        return b / float(m)
    return np.linalg.solve(m, b)


def _solve_step(
    problem: SemiDiscreteProblem,
    form: StepForm,
    guess: NDArray[np.floating],
    *,
    solve,
    max_newton: int,
    tol: float,
) -> NDArray[np.floating]:
    """Advance one step by a Newton loop on ``residual(u_arg, u_dot, u_ddot, t)`` for ``u_{n+1}``.

    Each iteration evaluates the residual at the affine step form and asks the problem for the
    weighted tangent ``jacobian(..., form.weights)``, then takes ``du = -J^{-1} R``. For a LINEAR
    problem one iteration is exact (``max_newton = 1``, the default first cut); a NONLINEAR problem
    is where a ``gauss.nonlinear`` strategy plugs in -- this local loop is the minimal Newton that
    the richer registry-driven iteration would replace, driving the SAME residual/jacobian seam.
    """
    u = np.asarray(guess, dtype=float).copy()
    t = form.t_eval
    for _ in range(max_newton):
        u_arg, u_dot, u_ddot = form.u_arg(u), form.u_dot(u), form.u_ddot(u)
        r = np.asarray(problem.residual(u_arg, u_dot, u_ddot, t), dtype=float)
        if float(np.linalg.norm(r)) <= tol:
            break
        jac = problem.jacobian(u_arg, u_dot, u_ddot, t, form.weights)
        du = np.asarray(solve(jac, -r), dtype=float)
        u = u + du.reshape(u.shape)
    return u


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
) -> TransientSolution:
    """March ``problem`` from ``study.t0`` to ``study.t1`` by ``study.dt`` with ``study.integrator``.

    Steps the selected :class:`ITimeIntegrator`: each step ``prepare`` yields the affine
    :class:`StepForm`, :func:`_solve_step` advances the residual/jacobian seam for ``u_{n+1}``, and
    ``advance`` folds it back into the history. ``v0`` / ``a0`` seed the second-order schemes
    (default zero). ``max_newton = 1`` is the exact linear step; raise it (or swap ``solve``) for a
    nonlinear problem. Returns the full march including the initial condition.
    """
    integrator = get_integrator(study.integrator)
    u0 = np.asarray(u0, dtype=float)
    state = integrator.start(u0, study.t0, v0, a0)
    dt = study.dt
    n_steps = round((study.t1 - study.t0) / dt)
    solution = TransientSolution(times=[study.t0], states=[u0])
    for k in range(n_steps):
        t_new = study.t0 + (k + 1) * dt
        form = integrator.prepare(state, dt, t_new)
        u_new = _solve_step(problem, form, state.last_u, solve=solve, max_newton=max_newton, tol=tol)
        state = integrator.advance(state, u_new, dt, t_new)
        solution.times.append(t_new)
        solution.states.append(u_new)
    return solution
