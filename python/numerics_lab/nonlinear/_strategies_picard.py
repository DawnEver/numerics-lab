"""The Picard family: ONE successive-substitution loop on two orthogonal axes.

* WHAT is relaxed -- the iterate (``x += alpha*(solve - x)``) or the coefficients (mix the raw
  secant arrays before the boundary fold, then ``x = solve``).
* HOW alpha moves -- constant 1, or the damped schedule of :class:`_AlphaSchedule`.

Everything else -- assemble at the top, the one-step-delayed break, :func:`_has_converged` or
:class:`_StagnationWatch`, ``on_iteration`` -- is shared. See ``nonlinear/__init__.py`` for the
NonlinearSystem/IterationStrategy protocol.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from numerics_lab.nonlinear._seam import (
    _ALPHA_CALM_STEPS,
    _ALPHA_GROW,
    _ALPHA_MIN,
    _RTOL_ENERGY_PCT,
    AssembledState,
    IterationResult,
    NonlinearSystem,
    RawStateSystem,
    _energy_error_pct,
    _has_converged,
    _IncrementTrend,
    _StagnationWatch,
    increment_ratio,
    unwrap_system,
)

__all__ = [
    'CoefficientDampedPicardStrategy',
    'DampedPicardStrategy',
    'PicardStrategy',
]


class _AlphaSchedule:
    """The under-relaxation factor: constant 1, or damped.

    The damped rules, in order each step:

    * shrink x0.5 when :class:`_IncrementTrend` claims damping (the iterate overshot);
    * shrink x0.5 when the energy error grows, and x0.6 on an energy-delta sign flip -- both only
      ABOVE ``rtol``: below it the energy is at its noise floor and neither is a limit cycle;
    * after :data:`_ALPHA_CALM_STEPS` calm steps grow x :data:`_ALPHA_GROW`, capped at the step
      size that last misbehaved; once alpha has reached the cap, the cap itself grows.

    A shrink that did not move alpha (already at :data:`_ALPHA_MIN`) sets no cap, and a step whose
    increment grew is not calm. The fixed point and the convergence test do not depend on alpha.
    """

    def __init__(self, *, damped: bool) -> None:
        self.damped = damped
        self.alpha = 1.0
        self._cap = 1.0
        self._calm = 0
        self._prev_err = np.inf
        self._prev_sign = 0

    def step(self, err: float, delta: float | None, rtol: float, trend: _IncrementTrend) -> float:
        """Update alpha from this step's energy error and energy change; return it."""
        if not self.damped:
            return self.alpha
        meaningful = err >= rtol
        before, disturbed = self.alpha, False
        if trend.claim_damping():
            self.alpha, disturbed = max(_ALPHA_MIN, self.alpha * 0.5), True
        if meaningful and np.isfinite(self._prev_err) and err > self._prev_err:
            self.alpha, disturbed = max(_ALPHA_MIN, self.alpha * 0.5), True
        if delta is not None:
            sign = (delta > 0) - (delta < 0)
            if meaningful and self._prev_sign != 0 and sign not in (0, self._prev_sign):
                self.alpha, disturbed = max(_ALPHA_MIN, self.alpha * 0.6), True
            self._prev_sign = sign
        self._prev_err = err
        if disturbed:
            if self.alpha < before:
                self._cap = before
            self._calm = 0
        elif not trend.permits_growth():
            self._calm = 0
        else:
            self._calm += 1
            if self._calm >= _ALPHA_CALM_STEPS:
                if self.alpha >= self._cap:
                    self._cap = min(1.0, self._cap * _ALPHA_GROW)
                self.alpha = min(self._cap, self.alpha * _ALPHA_GROW)
                self._calm = 0
        return self.alpha


class _PicardLoop:
    """Successive substitution, parametrised by relax target and alpha schedule.

    ``relax='coefficients'`` rests on one contract the system declares by exposing ``raw_state``:
    every raw matrix and source entry is ``c_e * <fixed factor>`` for a per-entry coefficient
    ``c_e`` and assembly is linear, so mixing the coefficients ``c_k = (1-a) c_(k-1) + a c(x_k)``
    equals mixing the assembled raw arrays ``S_k = (1-a) S_(k-1) + a S(c(x_k))`` (likewise the
    source). That holds ONLY for a GLOBAL scalar alpha; a per-entry alpha would break it silently.
    The mix must precede the boundary fold, hence the ``raw_state``/``fold_raw`` requirement. A
    magnetostatic system satisfies it with the reluctivity as ``c``; that reading is its own.
    """

    name: str
    relax: Literal['x', 'coefficients']
    damped: bool

    def solve(
        self, system: NonlinearSystem, *, max_iter: int, rtol_energy_pct: float = _RTOL_ENERGY_PCT
    ) -> IterationResult:
        schedule = _AlphaSchedule(damped=self.damped)
        assemble = self._coefficient_assembler(system, schedule) if self.relax == 'coefficients' else system.assemble
        trend = _IncrementTrend()
        watch = _StagnationWatch(rtol_energy_pct)
        x = system.initial_guess()
        state: AssembledState | None = None
        last_energy: float | None = None
        converged = False
        ran = 0
        for i in range(max_iter):
            ran = i + 1
            state = assemble(x)
            if converged:
                break
            err = _energy_error_pct(state.total_energy, last_energy)
            delta = None if last_energy is None else state.total_energy - last_energy
            alpha = schedule.step(err, delta, rtol_energy_pct, trend)
            x_solve = system.solve_linear(state)
            increment = increment_ratio(x, x_solve)
            converged = _has_converged(
                err, x, x_solve, rtol_energy_pct, material_change=state.material_change
            ) or watch.observe(err, increment, state.material_change)
            last_energy = state.total_energy
            trend.observe(increment)
            x = x + alpha * (x_solve - x) if self.relax == 'x' and self.damped else x_solve
            if system.is_linear:
                converged = True
            system.on_iteration(i, state, err, alpha, increment)
        return IterationResult(x=x, state=state, iterations=ran, converged=converged)

    def _coefficient_assembler(self, system: NonlinearSystem, schedule: _AlphaSchedule):
        """``assemble`` for ``relax='coefficients'``: raw arrays mixed with the current alpha, then folded."""
        inner = unwrap_system(system)
        if not isinstance(inner, RawStateSystem):
            named = type(inner).__name__
            if inner is not system:
                named = f'{named} (inside {type(system).__name__})'
            msg = f"'{self.name}' requires a system exposing raw_state/fold_raw; got {named}"
            raise TypeError(msg)
        prev: list = []  # last iteration's MIXED (s_val, t_mat); empty before the first pass

        def assemble(x):
            s_row, s_col, s_val, t_mat, total_energy = inner.raw_state(x)
            if prev:
                a = schedule.alpha
                s_val = (1.0 - a) * prev[0] + a * s_val
                t_mat = (1.0 - a) * prev[1] + a * t_mat
            prev[:] = [s_val, t_mat]
            return inner.fold_raw(s_row, s_col, s_val, t_mat, total_energy)

        return assemble


class DampedPicardStrategy(_PicardLoop):
    """Relax the iterate on the damped schedule -- the MEC default."""

    name, relax, damped = 'picard-damped', 'x', True


class CoefficientDampedPicardStrategy(_PicardLoop):
    """Relax the assembly coefficients on the damped schedule."""

    name, relax, damped = 'coefficient-damped', 'coefficients', True


class PicardStrategy(_PicardLoop):
    """Plain substitution, alpha 1. Kept so its divergence on saturating B-H stays a labelled datum."""

    name, relax, damped = 'picard', 'x', False
