"""Shared fixtures for the split ``test_events_*`` files -- the physics-blind W2 event seam.

``_RampWithToggle`` is the minimal stateful ``SemiDiscreteProblem`` used across the event-location
and solver-internals splits: ``u_dot = slope`` with a callback-flipped slope, so the trajectory is
a triangle turning exactly at ``level``.
"""

# the test/fixture, keeping collection cheap and loading platform-conditional deps (win32com,
# OCP, FEMM) only on the path that needs them. Hoist a file's imports and DELETE this header;
# the population is ratcheted by tests/architecture/ratchets/test_suppression_ratchet.py.

from __future__ import annotations

import numpy as np


class _RampWithToggle:
    """``u_dot = slope`` with a callback-flipped slope -- the minimal stateful ``SemiDiscreteProblem``.

    The event ``u - level`` crosses zero at a known time (``level`` with slope +1 from ``u(0)=0``);
    the callback negates the slope, so the trajectory is a triangle turning exactly at ``level``.
    """

    def __init__(self, level: float) -> None:
        self.slope = 1.0
        self.level = level
        self.toggles = 0

    def residual(self, u, u_dot, u_ddot, t):
        return np.asarray(u_dot) - self.slope

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.array([[weights.alpha1.real]])

    def event_fn(self, u, t):
        # One-sided by construction: once toggled, u falls away from the level, so the same
        # function cannot re-trigger -- the state-dependent event is the laplace switch's shape.
        return float(u[0]) - self.level if self.slope > 0.0 else -1.0

    def toggle(self):
        self.slope = -1.0
        self.toggles += 1
