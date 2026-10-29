"""Waveform relaxation: the ONE outer fixed point a staggered (window) coupling iterates -- physics-blind.

A staggered couple exchanges a whole WINDOW between two marchers: march one side over the window
against the other's last waveform, hand the result back, repeat until the exchanged waveform stops
moving. The loop is the same whatever the two sides are, so it lives here once:

    x_{k+1} = x_k + omega * (G(x_k) - x_k),   stop when  max|G(x_k) - x_k| < tol + tol_rel * max|G(x_k)|

``G`` is the caller's sweep (march both sides once and return the proposal), ``x`` the exchanged
quantity -- a scalar operating-point parameter or a sampled waveform. THE STOPPING TEST IS THE
FIXED-POINT GAP ``G(x) - x``, NEVER THE RELAXED STEP: the step is ``omega`` times the gap, so a test
on it could be met by lowering ``omega`` alone, a self-consistency the couple never reached.

A proposal whose shape differs from the state (the first sweep against an empty seed, a march that
landed a different grid) is measured against ZERO of its own shape, and the relaxed update starts
from that zero -- the waveform the circuit has never been handed is the unexcited one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

__all__ = ['RelaxedFixedPoint', 'waveform_relaxation']


@dataclass(frozen=True)
class RelaxedFixedPoint[P]:
    """The outcome: the relaxed state, the LAST sweep's payload and the gap trace.

    ``residuals[k]`` is the unrelaxed gap ``max|G(x_k) - x_k|`` of sweep ``k``; ``converged`` is
    whether it fell below ``tol`` within ``max_iter``. ``state`` is relaxed once more after the last
    sweep, exactly as every earlier one was.
    """

    state: object
    payload: P
    residuals: list[float] = field(default_factory=list)
    converged: bool = False
    iterations: int = 0


def _against(state: object, proposal: object) -> object:
    """The state the proposal is measured against: itself, or zero when it has no matching shape."""
    if state is None or np.shape(state) != np.shape(proposal):
        return np.zeros_like(proposal) if np.ndim(proposal) else 0.0
    return state


def waveform_relaxation[P](
    sweep: Callable[[object], tuple[object, P]],
    seed: object,
    *,
    relaxation: float = 1.0,
    tol: float = 0.0,
    tol_rel: float = 0.0,
    max_iter: int,
) -> RelaxedFixedPoint[P]:
    """Iterate ``sweep`` to its fixed point under relaxation ``0 < relaxation <= 1``.

    ``sweep(x) -> (proposal, payload)``: the proposal is what the sweep says ``x`` should be; the
    payload is whatever the caller needs back from the last sweep (the marched trajectories). ``tol``
    is in the unit of ``x``; ``tol_rel`` is dimensionless, against the proposal's own peak
    ``max|G(x)|`` -- the bar a waveform whose scale is not known before the first sweep can declare.
    The gap is the max-norm over every component.

    Raises:
        ValueError: ``relaxation`` outside ``(0, 1]``, ``max_iter < 1``, or no positive bar.

    """
    if not 0.0 < relaxation <= 1.0:
        msg = f'relaxation must be in (0, 1]; got {relaxation}'
        raise ValueError(msg)
    if max_iter < 1:
        msg = f'max_iter must be at least 1; got {max_iter}'
        raise ValueError(msg)
    if not (tol > 0.0 or tol_rel > 0.0) or tol < 0.0 or tol_rel < 0.0:
        msg = f'a fixed point needs a positive bar: tol={tol}, tol_rel={tol_rel}'
        raise ValueError(msg)
    state = seed
    residuals: list[float] = []
    payload: P | None = None
    for iteration in range(1, max_iter + 1):
        proposal, payload = sweep(state)
        base = _against(state, proposal)
        residual = float(np.max(np.abs(proposal - base)))
        residuals.append(residual)
        state = base + relaxation * (proposal - base)
        if residual < tol + tol_rel * float(np.max(np.abs(proposal))):
            return RelaxedFixedPoint(state, payload, residuals, True, iteration)
    return RelaxedFixedPoint(state, payload, residuals, False, max_iter)
