"""Concrete time integrators (BDF1/BDF2/trapezoid/TR-BDF2/Newmark/generalized-alpha) and their
registry, split from `time_integrators/__init__.py` because that module crossed the 800-line band.
See its docstring for the affine-weights contract these implement.

Every scheme here is valid at ARBITRARY step spacing: the history spacing ``state.h`` rides the
state, so BDF2 differentiates against the quadratic through its actual points and a marcher never
picks a form for it. Each carries ``error_constant`` -- the ``C`` in ``LTE = C h^(p+1) u^(p+1)`` --
which the step-size controller scales its predictor difference by.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from numerics_lab.formulation import NEWMARK_BETA, NEWMARK_GAMMA, DerivativeWeights, IntegratorKind
from numerics_lab.time_integrators._forms import IntegratorState, ITimeIntegrator, StepForm

__all__ = [
    'INTEGRATOR_REGISTRY',
    'BDF1Integrator',
    'BDF2Integrator',
    'GeneralizedAlphaIntegrator',
    'NewmarkIntegrator',
    'TRBDF2Integrator',
    'TrapezoidIntegrator',
    'get_integrator',
]

#: TR-BDF2's stage split. ``2 - sqrt(2)`` is the unique choice giving both stages the same
#: jacobian weight ``2/(gamma dt) = (2 - gamma)/((1 - gamma) dt)``, so one factor serves the step.
TR_BDF2_GAMMA = 2.0 - 2.0**0.5


def _zeros_like(u: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.zeros_like(np.asarray(u, dtype=float))


@lru_cache(maxsize=64)
def _bdf1_weights(dt: float) -> DerivativeWeights:
    return DerivativeWeights(alpha0=1.0, alpha1=1.0 / dt)


@lru_cache(maxsize=64)
def _bdf2_weights(dt: float) -> DerivativeWeights:
    return DerivativeWeights(alpha0=1.0, alpha1=1.5 / dt)


def _bdf2_variable_form(state: IntegratorState, dt: float, h: float, t_new: float) -> StepForm:
    """The variable-spacing BDF2 step form; ``h`` is the previous step's size.

    The derivative at ``t_new`` of the quadratic through the three unevenly spaced points; it
    reduces to the uniform ``(3u - 4u_n + u_{n-1})/(2dt)`` at ``h = dt`` exactly. Fixed weights
    against an uneven history are an O(1) wrong equation, not an order loss (the events module
    docstring records the measurement).
    """
    u_n, u_nm1 = state.us[-1], state.us[-2]
    return StepForm(
        weights=DerivativeWeights(alpha0=1.0, alpha1=(2.0 * dt + h) / (dt * (dt + h))),
        h_u=np.zeros_like(u_n),
        h_dot=dt / (h * (dt + h)) * (u_nm1 - u_n),
        h_ddot=np.zeros_like(u_n),
        t_eval=t_new,
        rate_origin=u_n,
    )


def _bdf1_form(u_n: NDArray[np.floating], dt: float, t_new: float) -> StepForm:
    return StepForm(
        weights=_bdf1_weights(dt),
        h_u=_zeros_like(u_n),
        h_dot=_zeros_like(u_n),
        h_ddot=_zeros_like(u_n),
        t_eval=t_new,
        rate_origin=u_n,
    )


class BDF1Integrator:
    """Backward Euler (BDF1) -- first order, unconditionally stable, for the ``u_dot`` (eddy) form.

    ``u_dot = (u_{n+1} - u_n) / dt`` gives the current-step slope ``alpha1 = 1/dt`` and history
    ``h_dot = -u_n/dt``; ``alpha2 = 0`` (no ``u_ddot`` term). One implicit solve per step; the
    local truncation error is ``O(dt^2)`` so the global error is ``O(dt)`` (verified by the
    analytic convergence test).
    """

    name = 'bdf1'
    order = 1
    stages = 1
    error_constant = 0.5

    def weights(self, dt: float) -> DerivativeWeights:
        return _bdf1_weights(dt)

    def start(self, u0, t0, v0=None, a0=None) -> IntegratorState:
        return IntegratorState(us=[np.asarray(u0, dtype=float)], t=t0)

    def prepare(self, state: IntegratorState, dt: float, t_new: float) -> StepForm:
        return _bdf1_form(state.last_u, dt, t_new)

    def advance(self, state: IntegratorState, u_new, dt: float, t_new: float, form: StepForm) -> IntegratorState:
        return IntegratorState(us=[np.asarray(u_new, dtype=float)], t=t_new, h=dt)

    def restart(self, state: IntegratorState, u, t: float) -> IntegratorState:
        return self.start(u, t)

    def restart_at_pair(self, state, u, t, _rate, _acceleration) -> IntegratorState:
        return self.restart(state, u, t)


class BDF2Integrator:
    """Second-order backward difference (BDF2) -- ``u_dot = (3u_{n+1} - 4u_n + u_{n-1})/(2 dt)``.

    Slope ``alpha1 = 3/(2 dt)``, history ``h_dot = (-4u_n + u_{n-1})/(2 dt)``; ``alpha2 = 0``.
    Uniform-step BDF2 needs two past states, so the FIRST step (only ``u_0`` known) falls back to
    a single BDF1 step -- the standard bootstrap, which keeps the scheme second order globally.
    """

    name = 'bdf2'
    order = 2
    stages = 1
    error_constant = 2.0 / 9.0

    def weights(self, dt: float) -> DerivativeWeights:
        return _bdf2_weights(dt)

    def start(self, u0, t0, v0=None, a0=None) -> IntegratorState:
        return IntegratorState(us=[np.asarray(u0, dtype=float)], t=t0)

    def prepare(self, state: IntegratorState, dt: float, t_new: float) -> StepForm:
        if len(state.us) < 2:  # bootstrap: not enough history yet -> one BDF1 step
            return _bdf1_form(state.last_u, dt, t_new)
        if state.h is not None and state.h != dt:
            return _bdf2_variable_form(state, dt, state.h, t_new)
        u_n, u_nm1 = state.us[-1], state.us[-2]
        return StepForm(
            weights=self.weights(dt),
            h_u=_zeros_like(u_n),
            h_dot=(u_nm1 - u_n) / (2.0 * dt),
            h_ddot=_zeros_like(u_n),
            t_eval=t_new,
            rate_origin=u_n,
        )

    def advance(self, state: IntegratorState, u_new, dt: float, t_new: float, form: StepForm) -> IntegratorState:
        us = [*state.us, np.asarray(u_new, dtype=float)][-2:]  # keep only the last two states
        return IntegratorState(us=us, t=t_new, h=dt)

    def restart(self, state: IntegratorState, u, t: float) -> IntegratorState:
        return self.start(u, t)

    def restart_at_pair(self, state, u, t, _rate, _acceleration) -> IntegratorState:
        return self.restart(state, u, t)


class TrapezoidIntegrator:
    """The trapezoidal rule -- ``u_{n+1} = u_n + dt/2 (u_dot_n + u_dot_{n+1})``, A-stable, order 2.

    Affine form: ``u_dot_{n+1} = 2/dt u_{n+1} - 2/dt u_n - u_dot_n``, so ``alpha1 = 2/dt`` and the
    history is the previous RATE ``v``. With no rate known (a start without ``v0``, or a restart
    at an event, where the pre-event rate belongs to a different problem) the first step is one
    BDF1 step -- a single ``O(dt^2)`` local error, so the scheme stays second order globally. Not
    L-stable: a stiff mode rings at ``(-1)^n``; :class:`TRBDF2Integrator` is the damped sibling.
    """

    name = 'trapezoid'
    order = 2
    stages = 1
    error_constant = 1.0 / 12.0

    def weights(self, dt: float) -> DerivativeWeights:
        return DerivativeWeights(alpha0=1.0, alpha1=2.0 / dt)

    def start(self, u0, t0, v0=None, a0=None) -> IntegratorState:
        v = None if v0 is None else np.asarray(v0, dtype=float)
        return IntegratorState(us=[np.asarray(u0, dtype=float)], t=t0, v=v)

    def prepare(self, state: IntegratorState, dt: float, t_new: float) -> StepForm:
        u_n = state.last_u
        if state.v is None:
            return _bdf1_form(u_n, dt, t_new)
        return StepForm(
            weights=self.weights(dt),
            h_u=_zeros_like(u_n),
            h_dot=-state.v,
            h_ddot=_zeros_like(u_n),
            t_eval=t_new,
            rate_origin=u_n,
        )

    def advance(self, state: IntegratorState, u_new, dt: float, t_new: float, form: StepForm) -> IntegratorState:
        u_new = np.asarray(u_new, dtype=float)
        rate = form.u_dot(u_new)
        return IntegratorState(us=[u_new], t=t_new, v=rate, h=dt)

    def restart(self, state: IntegratorState, u, t: float) -> IntegratorState:
        return self.start(u, t)

    def restart_at_pair(self, _state, u, t, rate, _acceleration) -> IntegratorState:
        return self.start(u, t, rate)


class TRBDF2Integrator:
    """TR-BDF2 (Bank et al. 1985) -- a trapezoid stage to ``t_n + gamma dt``, then BDF2 to ``t_{n+1}``.

    Second order and L-stable: the BDF2 stage damps the stiff modes the trapezoid stage would let
    ring. The second stage IS the variable-spacing BDF2 form through ``(t_n, t_n + gamma dt,
    t_{n+1})``, so no new formula exists here. With ``gamma = 2 - sqrt(2)`` both stages carry the
    same jacobian weight ``(2 + sqrt(2))/dt``. The stage is tracked on ``state.stage``; the
    intermediate state never joins a trajectory. With no rate known the trapezoid stage is a BDF1
    stage (as for :class:`TrapezoidIntegrator`).
    """

    name = 'tr_bdf2'
    order = 2
    stages = 2
    # |(-3 gamma^2 + 4 gamma - 2) / (12 (2 - gamma))|
    error_constant = abs(-3.0 * TR_BDF2_GAMMA**2 + 4.0 * TR_BDF2_GAMMA - 2.0) / (12.0 * (2.0 - TR_BDF2_GAMMA))

    def weights(self, dt: float) -> DerivativeWeights:
        return DerivativeWeights(alpha0=1.0, alpha1=2.0 / (TR_BDF2_GAMMA * dt))

    def start(self, u0, t0, v0=None, a0=None) -> IntegratorState:
        v = None if v0 is None else np.asarray(v0, dtype=float)
        return IntegratorState(us=[np.asarray(u0, dtype=float)], t=t0, v=v)

    def prepare(self, state: IntegratorState, dt: float, t_new: float) -> StepForm:
        if state.stage == 1:  # the BDF2 stage, from the stage point to t_new
            return _bdf2_variable_form(state, t_new - state.t, float(state.h), t_new)
        u_n, h_g = state.last_u, TR_BDF2_GAMMA * dt
        if state.v is None:
            return _bdf1_form(u_n, h_g, state.t + h_g)
        return StepForm(
            weights=self.weights(dt),
            h_u=_zeros_like(u_n),
            h_dot=-state.v,
            h_ddot=_zeros_like(u_n),
            t_eval=state.t + h_g,
            rate_origin=u_n,
        )

    def advance(self, state: IntegratorState, u_new, dt: float, t_new: float, form: StepForm) -> IntegratorState:
        u_new = np.asarray(u_new, dtype=float)
        if state.stage == 0:
            h_g = TR_BDF2_GAMMA * dt
            return IntegratorState(us=[state.last_u, u_new], t=state.t + h_g, v=state.v, h=h_g, stage=1)
        rate = form.u_dot(u_new)
        return IntegratorState(us=[u_new], t=t_new, v=rate, h=dt)

    def restart(self, state: IntegratorState, u, t: float) -> IntegratorState:
        return self.start(u, t)

    def restart_at_pair(self, _state, u, t, rate, _acceleration) -> IntegratorState:
        return self.start(u, t, rate)


# Newmark defaults: average-acceleration (unconditionally stable, second order). IMPORTED from
# `formulation`, which owns them, not re-declared here.


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
    stages = 1
    error_constant = 1.0 / 12.0

    def __init__(self, beta: float = NEWMARK_BETA, gamma: float = NEWMARK_GAMMA) -> None:
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

    def advance(self, state: IntegratorState, u_new, dt: float, t_new: float, form: StepForm) -> IntegratorState:
        u_new = np.asarray(u_new, dtype=float)
        w = self.weights(dt)
        a_new = w.alpha2.real * u_new + form.h_ddot
        v_new = form.u_dot(u_new)
        return IntegratorState(us=[u_new], t=t_new, v=v_new, a=a_new, h=dt)

    def restart(self, state: IntegratorState, u, t: float) -> IntegratorState:
        return self.start(u, t, state.v, state.a)  # u, v, a are continuous; only spacing is dropped

    def restart_at_pair(self, _state, u, t, rate, acceleration) -> IntegratorState:
        if rate is None or acceleration is None:
            msg = 'second-order projected restart requires actual outgoing rate and acceleration'
            raise ValueError(msg)
        return self.start(u, t, rate, acceleration)


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
    stages = 1
    error_constant = 1.0 / 12.0

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

    def advance(self, state: IntegratorState, u_new, dt: float, t_new: float, form: StepForm) -> IntegratorState:
        u_new = np.asarray(u_new, dtype=float)
        a_slope, a_off, v_slope, v_off = self._newmark_form(state, dt)
        a_new = a_slope * u_new + a_off
        v_new = v_slope * u_new + v_off
        return IntegratorState(us=[u_new], t=t_new, v=v_new, a=a_new, h=dt)

    def restart(self, state: IntegratorState, u, t: float) -> IntegratorState:
        return self.start(u, t, state.v, state.a)

    def restart_at_pair(self, _state, u, t, rate, acceleration) -> IntegratorState:
        if rate is None or acceleration is None:
            msg = 'second-order projected restart requires actual outgoing rate and acceleration'
            raise ValueError(msg)
        return self.start(u, t, rate, acceleration)


INTEGRATOR_REGISTRY: dict[IntegratorKind, ITimeIntegrator] = {
    IntegratorKind.BDF1: BDF1Integrator(),
    IntegratorKind.BDF2: BDF2Integrator(),
    IntegratorKind.TRAPEZOID: TrapezoidIntegrator(),
    IntegratorKind.TR_BDF2: TRBDF2Integrator(),
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
