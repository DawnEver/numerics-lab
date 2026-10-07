"""RRE finds the fixed point of a slow period map in a handful of periods, and refuses a map with none."""

from __future__ import annotations

import numpy as np
import pytest

from motronics.gauss.time_integrators.periodic import periodic_fixed_point


def test_a_slow_affine_map_is_solved_exactly_in_a_few_periods() -> None:
    rng = np.random.default_rng(7)
    q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    a = q @ np.diag([0.999, 0.99, 0.5, -0.3]) @ q.T  # plain iteration: ~2e4 periods to 1e-9
    b = rng.standard_normal(4)
    exact = np.linalg.solve(np.eye(4) - a, b)
    fixed = periodic_fixed_point(lambda x: a @ x + b, np.zeros(4), tol=1e-10)
    assert np.max(np.abs(fixed.state - exact)) <= 1e-7 * np.max(np.abs(exact))
    assert fixed.map_calls <= 20


def test_a_map_without_a_fixed_point_raises_with_its_residual() -> None:
    with pytest.raises(RuntimeError, match='no periodic steady state'):
        periodic_fixed_point(lambda x: x + 1.0, np.zeros(2), max_cycles=3)
