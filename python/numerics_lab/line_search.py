"""Line searches along a given direction: backtracking, Armijo, strong Wolfe, and the energy-slope search.

Every search here asks only for the function along the ray, ``phi(step)`` (and ``phi'(step)`` where
the condition needs it); none knows what a residual, an energy or a design variable is. The Newton
strategies and the implicit time step are callers, not owners.

- :func:`backtracking` -- try ``1, shrink, shrink**2, ...`` and take the first trial the caller's
  predicate accepts. A trial may be UNEVALUABLE (the caller returns ``None``): it is rejected like
  any other and the search shrinks.
- :func:`sufficient_decrease` / :func:`armijo` -- the Armijo predicate
  ``phi(step) <= phi(0) + c1 step phi'(0)``, and the backtracking search over it.
- :func:`strong_wolfe` -- a step satisfying the strong Wolfe conditions (Nocedal and Wright,
  Algorithms 3.5 and 3.6, cubic interpolation in the zoom with a bisection safeguard).
- :func:`directional_line_search` -- the energy-slope search: the step minimising a convex potential
  along the ray, judged on its derivative ``phi'(s) = R(u + s step) . step`` alone.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

__all__ = [
    'Backtrack',
    'TrialNotEvaluable',
    'WolfeSearchFailed',
    'WolfeStep',
    'armijo',
    'backtracking',
    'directional_line_search',
    'strong_wolfe',
    'sufficient_decrease',
]

T = TypeVar('T')


class TrialNotEvaluable(RuntimeError):
    """Raised by a problem's ``residual`` when a line-search TRIAL point has no value to give.

    A trial is speculative: an inner root that does not exist there (the condensed field-circuit residual), or a
    constitutive law that cannot certify its evaluation there. Every search here REJECTS such a trial exactly as
    one that fails its test, and shrinks -- only a point a search returns must evaluate.
    """


@dataclass(frozen=True)
class Backtrack[T]:
    """What a backtracking search found.

    ``accepted``: the predicate accepted ``trial`` at ``step``. Otherwise the search EXHAUSTED its
    tries: ``step`` is the NEXT, untried length (``shrink**tries``), and ``trial`` is the last trial
    evaluated, ``None`` when that one could not be evaluated. ``first`` is the full step's trial
    (``None`` if it could not be evaluated); a caller that falls back to the full step reads it here
    rather than evaluating it twice.
    """

    step: float
    trial: T | None
    accepted: bool
    first: T | None


def backtracking[T](
    trial: Callable[[float], T | None],
    accept: Callable[[float, T], bool],
    *,
    tries: int,
    shrink: float = 0.5,
) -> Backtrack[T]:
    """Evaluate ``trial(step)`` at ``1, shrink, shrink**2, ...`` (``tries`` of them); the first accepted wins.

    ``trial`` returns ``None`` for a point it cannot evaluate; that trial is rejected and the search
    shrinks. The step lengths are formed by repeated multiplication, so with ``shrink = 0.5`` every
    length is exact.
    """
    if tries < 1:
        msg = f'a backtracking search needs at least one try, got {tries!r}'
        raise ValueError(msg)
    if not 0.0 < shrink < 1.0:
        msg = f'the shrink factor must lie in (0, 1), got {shrink!r}'
        raise ValueError(msg)
    step = 1.0
    first: T | None = None
    value: T | None = None
    for attempt in range(tries):
        value = trial(step)
        if attempt == 0:
            first = value
        if value is not None and accept(step, value):
            return Backtrack(step, value, True, first)
        step *= shrink
    return Backtrack(step, value, False, first)


def sufficient_decrease(phi0: float, slope0: float, *, c1: float = 1e-4) -> Callable[[float, float], bool]:
    """The Armijo predicate ``(step, phi) -> phi`` finite and ``<= phi0 + c1 step slope0``.

    ``slope0`` is ``phi'(0)``, never positive: along an ascent direction no step decreases. A zero slope
    (a merit already at zero) asks only that the trial not rise; a non-finite one accepts nothing.
    """
    if slope0 > 0.0:
        msg = f"the Armijo condition needs a descent direction, phi'(0) = {slope0!r}"
        raise ValueError(msg)
    if not 0.0 < c1 < 1.0:
        msg = f'the Armijo constant must lie in (0, 1), got {c1!r}'
        raise ValueError(msg)

    def holds(step: float, phi: float) -> bool:
        return bool(np.isfinite(phi)) and phi <= phi0 + c1 * step * slope0

    return holds


def armijo(
    phi: Callable[[float], float | None],
    phi0: float,
    slope0: float,
    *,
    c1: float = 1e-4,
    shrink: float = 0.5,
    tries: int = 30,
) -> Backtrack[float]:
    """Backtrack along a descent direction until the Armijo sufficient-decrease condition holds."""
    holds = sufficient_decrease(phi0, slope0, c1=c1)
    return backtracking(phi, holds, tries=tries, shrink=shrink)


@dataclass(frozen=True)
class WolfeStep:
    """A step satisfying the strong Wolfe conditions, with ``phi`` and ``phi'`` there.

    ``capped`` marks More and Thuente's ``stpmax`` exit: the step IS ``max_step``, sufficient
    decrease holds there and ``phi`` still falls, so no larger step may be tried and the curvature
    condition is not met. A box-capped method (L-BFGS-B) takes it; an unbounded caller never sees it
    unless it set ``max_step``.
    """

    step: float
    phi: float
    slope: float
    evaluations: int
    capped: bool = False


class WolfeSearchFailed(RuntimeError):
    """No step satisfying the strong Wolfe conditions was found within the evaluation budget."""


def _cubic_minimizer(a: float, fa: float, ga: float, b: float, fb: float, gb: float) -> float | None:
    """The minimiser of the cubic interpolating ``(a, fa, ga)`` and ``(b, fb, gb)``, or ``None``."""
    d1 = ga + gb - 3.0 * (fa - fb) / (a - b)
    radicand = d1 * d1 - ga * gb
    if radicand < 0.0:
        return None
    d2 = math.copysign(math.sqrt(radicand), b - a)
    denominator = gb - ga + 2.0 * d2
    if denominator == 0.0:
        return None
    return b - (b - a) * (gb + d2 - d1) / denominator


def strong_wolfe(
    phi_and_slope: Callable[[float], tuple[float, float]],
    phi0: float,
    slope0: float,
    *,
    c1: float = 1e-4,
    c2: float = 0.9,
    step: float = 1.0,
    max_step: float = 1e10,
    max_evaluations: int = 30,
) -> WolfeStep:
    """A step ``a`` with ``phi(a) <= phi0 + c1 a phi'(0)`` and ``|phi'(a)| <= c2 |phi'(0)|``.

    Nocedal and Wright, Algorithm 3.5 (bracketing, the step doubling up to ``max_step``) and 3.6
    (zoom: a cubic interpolant kept inside the central 80 % of the bracket, else bisection).
    A step that reaches ``max_step`` still falling is returned ``capped`` (see :class:`WolfeStep`).
    Raises :class:`WolfeSearchFailed` when ``phi'(0)`` is not strictly negative (``-0.0``, ``0.0``,
    positive or NaN alike: there is no descent to search along -- whether the point is converged is
    the caller's question, not the search's), when the budget runs out, or when the zoom bracket
    collapses (no float lies strictly inside it, as at a jump in ``phi``); never returns an uncapped
    step that fails a condition.
    """
    if not slope0 < 0.0:
        msg = f"no descent to search along, phi'(0) = {slope0!r}"
        raise WolfeSearchFailed(msg)
    if not 0.0 < c1 < c2 < 1.0:
        msg = f'the Wolfe constants must satisfy 0 < c1 < c2 < 1, got c1={c1!r} c2={c2!r}'
        raise ValueError(msg)
    evaluations = 0

    def evaluate(a: float) -> tuple[float, float]:
        nonlocal evaluations
        if evaluations >= max_evaluations:
            msg = f'no strong Wolfe step within {max_evaluations} evaluations'
            raise WolfeSearchFailed(msg)
        evaluations += 1
        value, slope = phi_and_slope(a)
        return float(value), float(slope)

    def armijo_holds(a: float, value: float) -> bool:
        return math.isfinite(value) and value <= phi0 + c1 * a * slope0

    def zoom(lo: float, f_lo: float, g_lo: float, hi: float, f_hi: float, g_hi: float) -> WolfeStep:
        while True:
            left, right = min(lo, hi), max(lo, hi)
            if not left < 0.5 * (left + right) < right:
                msg = f'the zoom bracket collapsed at a={lo!r} after {evaluations} evaluations'
                raise WolfeSearchFailed(msg)
            trial = _cubic_minimizer(lo, f_lo, g_lo, hi, f_hi, g_hi) if math.isfinite(f_hi) else None
            margin = 0.1 * (right - left)
            if trial is None or not left + margin <= trial <= right - margin:
                trial = 0.5 * (lo + hi)
            f, g = evaluate(trial)
            if not armijo_holds(trial, f) or f >= f_lo:
                hi, f_hi, g_hi = trial, f, g
                continue
            if abs(g) <= -c2 * slope0:
                return WolfeStep(trial, f, g, evaluations)
            if g * (hi - lo) >= 0.0:
                hi, f_hi, g_hi = lo, f_lo, g_lo
            lo, f_lo, g_lo = trial, f, g

    previous, f_previous, g_previous = 0.0, float(phi0), float(slope0)
    a = float(step)
    while True:
        f, g = evaluate(a)
        if not armijo_holds(a, f) or (evaluations > 1 and f >= f_previous):
            return zoom(previous, f_previous, g_previous, a, f, g)
        if abs(g) <= -c2 * slope0:
            return WolfeStep(a, f, g, evaluations)
        if g >= 0.0:
            return zoom(a, f, g, previous, f_previous, g_previous)
        if a >= max_step:
            return WolfeStep(a, f, g, evaluations, capped=True)
        previous, f_previous, g_previous = a, f, g
        a = min(2.0 * a, max_step)


#: Secant (Illinois) evaluations :func:`directional_line_search` spends bracketing the line minimum.
_DIRECTIONAL_EVALS = 12


def directional_line_search(residual, u, step, r0, *, curvature: float = 0.5):
    """Minimize the step's ENERGY along ``step`` using only residuals: ``phi'(s) = R(u + s step) . step``.

    For a residual that IS the gradient of a convex potential (a magnetostatic step with a monotone
    law: the variational play, a monotone B-H curve, ``sigma/dt`` mass), ``phi`` is convex along the
    ray, so its minimizer is the root of the nondecreasing ``phi'``. Judging a step by ``|R|``
    instead is not a merit of that problem -- on a law with kinks the Newton direction need not
    lower ``|R|`` at all, measured on case_a with the hysteretic steel: full steps 4e2 -> 5e3,
    100 iterations at ~10 % each. Differences of the potential itself would cancel to round-off
    near the root; its DERIVATIVE does not. The full step is taken when ``phi'(1) <= 0``, or when
    ``0 < phi'(1) <= curvature |phi'(0)|`` and it lowers ``|R|`` (a smooth Newton finish); otherwise
    a bracketed Illinois secant stops at a LOW-side point ``-curvature |phi'(0)| <= phi'(s) <= 0``.
    A non-descent direction (``phi'(0) >= 0``) is taken whole: there is nothing to minimize along it.

    A trial the problem cannot evaluate (:class:`TrialNotEvaluable`) is REJECTED like an overshoot: it
    closes the bracket from above without a slope, and the next trial bisects toward the low end. Only a
    point that evaluated is ever returned; a step none of whose trials evaluates raises.
    """
    g0 = float(np.dot(np.ravel(r0), np.ravel(step)))
    if not g0 < 0.0:
        return u + step

    def slope(s):
        try:
            return float(np.dot(np.ravel(residual(u + s * step)), np.ravel(step)))
        except TrialNotEvaluable:
            return None

    try:
        r1 = np.ravel(residual(u + step))
    except TrialNotEvaluable:
        r1 = None
    g1 = None if r1 is None else float(np.dot(r1, np.ravel(step)))
    if g1 is not None and g1 <= 0.0:
        return u + step
    # A full step a SLIVER past the minimum that lowers |R| is a smooth law's Newton finish, not a kink's
    # overshoot (which raises |R| by orders): taken whole, as the band below would take a low-side point.
    if g1 is not None and g1 <= curvature * abs(g0) and np.linalg.norm(r1) < np.linalg.norm(np.ravel(r0)):
        return u + step
    lo, glo, hi, ghi = 0.0, g0, 1.0, g1
    evaluated = None if g1 is None else 1.0
    side = 0
    for _ in range(_DIRECTIONAL_EVALS):
        s = 0.5 * (lo + hi) if ghi is None else lo - glo * (hi - lo) / (ghi - glo)
        g = slope(s)
        if g is None:
            hi, ghi, side = s, None, 0
            continue
        evaluated = s
        if -curvature * abs(g0) <= g <= 0.0:
            return u + s * step
        if g < 0.0:
            lo, glo = s, g
            if side == -1 and ghi is not None:
                ghi *= 0.5
            side = -1
        else:
            hi, ghi = s, g
            if side == 1:
                glo *= 0.5
            side = 1
    # Unclosed: the bracket's LOW end, where phi' < 0 and so the energy fell; a point past the
    # minimum may sit above phi(0) however small its slope.
    if lo > 0.0:
        return u + lo * step
    if evaluated is None:
        msg = 'no trial along the step evaluates'
        raise TrialNotEvaluable(msg)
    return u + evaluated * step
