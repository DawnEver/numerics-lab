"""The explicit sampled march: classic RK4 between task hits, every hit and declared edge landed.

Anchored on a closed form -- ``u' = -u/tau + v`` with ``v`` held by a task -- whose piecewise
exponential solution is exact. The planted control is the same edge marched by a step that
STRADDLES it: landing the edge is what makes the march exact to RK4 order, and straddling it costs
first-order accuracy, which the test measures.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from numerics_lab.time_integrators import SampledTask, ScheduledTask, integrate_explicit_sampled

_TAU = 0.2


def _exact(times: np.ndarray, edge: float, v_before: float, v_after: float) -> np.ndarray:
    """``u(0) = 0``, ``v = v_before`` until ``edge`` then ``v_after``: the exact piecewise solution."""
    u_edge = v_before * _TAU * (1.0 - math.exp(-edge / _TAU))
    after = v_after * _TAU + (u_edge - v_after * _TAU) * np.exp(-(times - edge) / _TAU)
    before = v_before * _TAU * (1.0 - np.exp(-times / _TAU))
    return np.where(times <= edge, before, after)


def _march(dt: float, edge: float, *, declared: bool):
    held = [1.0]

    def switch(_t, _u):
        held[0] = -2.0

    def derivative(u, _t):
        return -u / _TAU + held[0]

    if declared:
        tasks = [ScheduledTask([edge], switch)]
        rhs = derivative
    else:  # the edge is read off the clock inside the derivative: every step that holds it straddles it
        tasks = []

        def rhs(u, t):
            return -u / _TAU + (1.0 if t < edge else -2.0)

    return integrate_explicit_sampled(rhs, 0.0, 1.0, dt, np.zeros(1), tasks)


def test_a_declared_edge_is_landed_and_the_march_is_fourth_order() -> None:
    edge = 0.33  # off every step grid below
    errors = []
    for dt in (0.02, 0.01):
        solution = _march(dt, edge, declared=True)
        times = np.asarray(solution.times)
        assert edge in solution.times, 'the declared edge must be a marched instant'
        errors.append(float(np.max(np.abs(np.asarray(solution.states)[:, 0] - _exact(times, edge, 1.0, -2.0)))))
    assert errors[0] < 1e-6
    assert errors[0] / errors[1] == pytest.approx(16.0, rel=0.15, abs=0.0), f'RK4 order not seen: {errors}'


def test_a_straddled_edge_is_first_order_the_planted_control() -> None:
    edge = 0.33
    landed = _march(0.02, edge, declared=True)
    straddled = _march(0.02, edge, declared=False)
    exact = _exact(np.asarray(straddled.times), edge, 1.0, -2.0)
    straddle_error = float(np.max(np.abs(np.asarray(straddled.states)[:, 0] - exact)))
    landed_error = float(
        np.max(np.abs(np.asarray(landed.states)[:, 0] - _exact(np.asarray(landed.times), edge, 1.0, -2.0)))
    )
    assert straddle_error > 100.0 * landed_error, (straddle_error, landed_error)


def test_tasks_fire_on_the_state_reached_at_their_hits_and_project_maps_every_step() -> None:
    seen = []

    def watch(t, u):
        seen.append((t, float(u[0])))

    solution = integrate_explicit_sampled(
        lambda _u, _t: np.array([1.0]),
        0.0,
        1.0,
        0.1,
        np.zeros(1),
        [SampledTask(0.25, watch)],
        project=lambda u: np.minimum(u, 0.6),
    )
    assert [t for t, _ in seen] == pytest.approx([0.0, 0.25, 0.5, 0.75])
    assert [u for _, u in seen] == pytest.approx([0.0, 0.25, 0.5, 0.6])
    assert max(float(s[0]) for s in solution.states) == pytest.approx(0.6)
    assert solution.times[-1] == 1.0
    assert float(np.min(np.diff(solution.times))) > 0.0


def test_a_non_positive_step_is_refused() -> None:
    with pytest.raises(ValueError, match='dt must be finite and positive'):
        integrate_explicit_sampled(lambda u, _t: u, 0.0, 1.0, 0.0, np.zeros(1))
