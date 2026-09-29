"""Concrete time integrators (BDF1/BDF2/Newmark/generalized-alpha) and their registry, split from
`time_integrators/__init__.py` because that module crossed the 800-line band. See its docstring
for the affine-weights contract these implement.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from motronics.contracts.spec.analysis_formulation import NEWMARK_BETA, NEWMARK_GAMMA, DerivativeWeights
from motronics.contracts.spec.study_spec import IntegratorKind
from motronics.gauss.time_integrators._forms import IntegratorState, ITimeIntegrator, StepForm

__all__ = [
    'INTEGRATOR_REGISTRY',
    'BDF1Integrator',
    'BDF2Integrator',
    'GeneralizedAlphaIntegrator',
    'NewmarkIntegrator',
    'get_integrator',
]


def _zeros_like(u: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.zeros_like(np.asarray(u, dtype=float))


@lru_cache(maxsize=64)
def _bdf1_weights(dt: float) -> DerivativeWeights:
    return DerivativeWeights(alpha0=1.0, alpha1=1.0 / dt)


@lru_cache(maxsize=64)
def _bdf2_weights(dt: float) -> DerivativeWeights:
    return DerivativeWeights(alpha0=1.0, alpha1=1.5 / dt)


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
        return _bdf1_weights(dt)

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
        return _bdf2_weights(dt)

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


# Newmark defaults: average-acceleration (unconditionally stable, second order). IMPORTED, not
# re-declared: contracts is the dependency leaf and owns them, and until 2026-09-03 this module
# carried its own copy under a comment saying it "matched" the contracts defaults -- an agreement
# nothing enforced. Now one edit moves both.


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
