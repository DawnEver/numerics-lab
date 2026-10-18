"""The one unconstrained minimiser: limited-memory BFGS on :class:`LbfgsMemory` and :func:`strong_wolfe`.

``minimize_lbfgs(fun_and_grad, x0)`` minimises a smooth ``f`` with ``fun_and_grad(x) -> (f, grad)``.
The direction is the two-loop recursion over the last ``memory`` secant pairs on ``H0 = gamma I``
(``gamma = s.y / y.y`` of the newest pair); the step is a strong-Wolfe search. A pair whose curvature
``y.s`` is not positive is skipped, and a direction that is not descent is replaced by ``-grad``.

It stops CONVERGED when ``|grad|_inf <= rtol |grad(x0)|_inf`` (reason ``'rtol'``) or ``<= atol``
(``'atol'``), both declared by the caller; otherwise on ``max_iter`` or when the line search fails
(:class:`WolfeSearchFailed`), each named in ``reason`` and never reported as convergence. A step the
search returns ``capped`` is taken like any other (it satisfies sufficient decrease).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from numerics_lab.line_search import WolfeSearchFailed, strong_wolfe
from numerics_lab.quasi_newton import LbfgsMemory

__all__ = ['MinimizeResult', 'minimize_lbfgs']

Vector = NDArray[np.float64]


@dataclass(frozen=True)
class MinimizeResult:
    """Where the descent stopped, ``f`` and its gradient there, the work spent, and why it stopped."""

    x: Vector
    fun: float
    grad: Vector
    iterations: int
    evaluations: int
    converged: bool
    reason: str


def minimize_lbfgs(
    fun_and_grad: Callable[[Vector], tuple[float, Vector]],
    x0: Vector,
    *,
    memory: int = 10,
    rtol: float = 1e-8,
    atol: float = 0.0,
    max_iter: int = 1000,
) -> MinimizeResult:
    """Minimise ``f`` from *x0*; see the module docstring for the stop rules."""
    model = LbfgsMemory(memory)
    if rtol < 0 or atol < 0 or (rtol == 0 and atol == 0):
        msg = f'rtol and atol are >= 0 and at least one is > 0, got rtol={rtol!r} atol={atol!r}'
        raise ValueError(msg)
    x = np.asarray(x0, dtype=float).copy()
    f, g = fun_and_grad(x)
    f, g = float(f), np.asarray(g, dtype=float)
    evaluations = 1
    bar_relative = rtol * float(np.max(np.abs(g), initial=0.0))

    def result(iterations: int, converged: bool, reason: str) -> MinimizeResult:
        return MinimizeResult(x, f, g, iterations, evaluations, converged, reason)

    for iteration in range(max_iter):
        worst = float(np.max(np.abs(g), initial=0.0))
        if worst <= bar_relative:
            return result(iteration, True, 'rtol')
        if worst <= atol:
            return result(iteration, True, 'atol')
        d = -model.inverse_apply(g)
        slope = float(g @ d)
        if not slope < 0:
            model.clear()
            d, slope = -g, -float(g @ g)
        seen: dict[float, tuple[float, Vector]] = {}

        def along(a: float, x=x, d=d, seen=seen) -> tuple[float, float]:
            value, grad = fun_and_grad(x + a * d)
            grad = np.asarray(grad, dtype=float)
            seen[a] = (float(value), grad)
            return float(value), float(grad @ d)

        first = 1.0 if model.pairs else min(1.0, 1.0 / worst)
        try:
            step = strong_wolfe(along, f, slope, step=first)
        except WolfeSearchFailed:
            evaluations += len(seen)
            return result(iteration, False, 'line search failed')
        evaluations += len(seen)
        f_new, g_new = seen[step.step]
        s, y = step.step * d, g_new - g
        model.push(s, y)
        x, f, g = x + s, f_new, g_new
    worst = float(np.max(np.abs(g), initial=0.0))
    if worst <= max(bar_relative, atol):
        return result(max_iter, True, 'rtol' if worst <= bar_relative else 'atol')
    return result(max_iter, False, 'max_iter')
