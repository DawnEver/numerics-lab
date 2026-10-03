"""Outer residual-march and load-continuation drivers over the nonlinear system protocol."""

from __future__ import annotations

import numpy as np

from motronics.gauss.nonlinear import (
    _RTOL_ENERGY_PCT,
    AssembledState,
    IterationResult,
    IterationStrategy,
    NonlinearSystem,
)
from motronics.gauss.nonlinear._strategies_picard import DampedPicardStrategy

__all__ = ['ContinuationStrategy', 'ResidualMarchStrategy']


class ResidualMarchStrategy:
    """A Newton / secant march judged on the RESIDUAL against the system's own arithmetic floor.

    For an equation whose convergence is a statement about its residual rather than an energy --
    the 3-D H(curl) curl-curl system, whose ``R = K(nu(B)) A - f`` is the measured quantity and
    whose floor is ``eps * || |J| . |u| ||`` -- the system itself decides when the residual is
    settled (:meth:`residual_settled`) and how long each step is (:meth:`step_length`); the
    strategy owns only the march: assemble, ask, solve ``tangent dx = -r``, step. It is not in
    ``ITERATION_REGISTRY`` because no energy-judged system can answer :meth:`residual_settled`.
    """

    name = 'residual-march'

    def solve(
        self, system: NonlinearSystem, *, max_iter: int, rtol_energy_pct: float = _RTOL_ENERGY_PCT
    ) -> IterationResult:
        del rtol_energy_pct  # the system's own residual rule decides
        x = system.initial_guess()
        state: AssembledState | None = None
        converged = False
        ran = 0
        for i in range(max_iter):
            ran = i + 1
            state = system.assemble(x)
            system.on_iteration(i, state, 0.0, system.step_length(i), 0.0)
            if system.residual_settled(state):
                converged = True
                break
            dx = system.solve_system(system.tangent(x, state), -system.residual(x, state))
            x = x + system.step_length(i) * dx
        return IterationResult(x=x, state=state, iterations=ran, converged=converged)


class ContinuationStrategy:
    """Load / source stepping: solve a ramp ``0 < lam_1 < ... < lam_n = 1`` of the source (sec 1).

    An OUTER driver for strong-saturation / large-current operating points where a single solve of the
    full source diverges. It scales the source (conductor current + magnet Br) by ``lam`` via the
    equation's ``scale_source`` hook, solving each sub-problem with an INNER strategy seeded by the
    previous solution (natural continuation). The last sub-solve (``lam=1``) is the full problem, so
    the converged field is the true solution. Degrades to a single full-source inner solve when the
    equation cannot rescale its source (``scale_source`` raises ``NotImplementedError``).
    """

    name = 'continuation'

    def __init__(self, steps: int = 4, inner: IterationStrategy | None = None) -> None:
        self.steps = max(1, steps)
        self.inner = inner or DampedPicardStrategy()

    def solve(
        self, system: NonlinearSystem, *, max_iter: int, rtol_energy_pct: float = _RTOL_ENERGY_PCT
    ) -> IterationResult:
        lambdas = [(k + 1) / self.steps for k in range(self.steps)]
        result: IterationResult | None = None
        seed: np.ndarray | None = None
        for lam in lambdas:
            try:
                sub = system if lam == 1.0 and self.steps == 1 else system.scale_source(lam)
            except NotImplementedError:
                # The equation cannot rescale its source: solve the full problem directly (no ramp).
                return self.inner.solve(system, max_iter=max_iter, rtol_energy_pct=rtol_energy_pct)
            if seed is not None:
                sub = _SeededSystem(sub, seed)
            result = self.inner.solve(sub, max_iter=max_iter, rtol_energy_pct=rtol_energy_pct)
            seed = result.x
        # ``result.state`` came from the scaled sub-system; re-assemble on the FULL system at the
        # converged iterate so the returned payload/energy are the true (lam=1) fields.
        final_state = system.assemble(result.x)
        return IterationResult(x=result.x, state=final_state, iterations=result.iterations, converged=result.converged)


class _SeededSystem:
    """Wrap a :class:`NonlinearSystem` to start from a supplied iterate (continuation warm start).

    Delegates every method to the wrapped system except :meth:`initial_guess`, which returns the seed
    (the previous continuation step's converged field) so each sub-solve starts near its solution.
    """

    def __init__(self, base: NonlinearSystem, seed: np.ndarray) -> None:
        self._base = base
        self._seed = seed

    def initial_guess(self) -> np.ndarray:
        return self._seed

    @property
    def unwrapped(self) -> NonlinearSystem:
        return self._base

    @property
    def is_linear(self) -> bool:
        return bool(self._base.is_linear)

    def assemble(self, x, /):
        return self._base.assemble(x)

    def solve_linear(self, state, /):
        return self._base.solve_linear(state)

    def residual(self, x, state, /):
        return self._base.residual(x, state)

    def residual_norm(self, x, state, /):
        return self._base.residual_norm(x, state)

    def tangent(self, x, state, /):
        return self._base.tangent(x, state)

    def solve_system(self, matrix, rhs, /):
        return self._base.solve_system(matrix, rhs)

    def solve_secant(self, state, rhs, /):
        return self._base.solve_secant(state, rhs)

    def on_iteration(self, i, state, err, alpha, increment, /):
        return self._base.on_iteration(i, state, err, alpha, increment)

    def scale_source(self, lam, /):
        return self._base.scale_source(lam)
