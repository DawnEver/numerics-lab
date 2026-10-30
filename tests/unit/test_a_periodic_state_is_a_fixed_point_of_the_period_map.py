"""RRE finds the fixed point of a slow period map in a handful of periods, and refuses a map with none."""

from __future__ import annotations

import numpy as np
import pytest

from numerics_lab.time_integrators.periodic import periodic_fixed_point


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


def test_an_in_place_map_cannot_overwrite_its_residual_history() -> None:
    initial = np.zeros(2)

    def drift(x):
        x += 1.0
        return x

    with pytest.raises(RuntimeError, match='no periodic steady state'):
        periodic_fixed_point(drift, initial, max_cycles=3)
    np.testing.assert_array_equal(initial, np.zeros(2))


@pytest.mark.parametrize('tol', [0.0, -1.0, np.nan, np.inf])
def test_an_invalid_periodic_tolerance_is_refused(tol) -> None:
    with pytest.raises(ValueError, match='tol'):
        periodic_fixed_point(lambda x: x, np.ones(1), tol=tol)


@pytest.mark.parametrize('period', [0.0, -1.0, np.nan, np.inf])
def test_an_invalid_forcing_period_is_refused_before_marching(period) -> None:
    from numerics_lab.time_integrators.periodic import smooth_phase  # noqa: PLC0415

    with pytest.raises(ValueError, match='period'):
        smooth_phase(period, ())


# serial: the closed form is reached only at rtol=1e-8 over full periods; a looser or shorter march cannot land it.
@pytest.mark.serial
@pytest.mark.parametrize('integrator', ['bdf1', 'bdf2', 'trapezoid', 'tr_bdf2'])
def test_the_steady_state_driver_restarts_off_every_edge_and_lands_the_closed_form(integrator) -> None:
    """``u' = -u + s(t)``, ``s`` a 0/1 square wave of period 1 (on for [0, 0.5)): exact periodic orbit."""
    from numerics_lab.formulation import TimeWindow  # noqa: PLC0415
    from numerics_lab.time_integrators import StepTolerance, integrate_transient  # noqa: PLC0415
    from numerics_lab.time_integrators.periodic import periodic_steady_state, smooth_phase  # noqa: PLC0415

    class _Square:
        def residual(self, u, u_dot, u_ddot, t):
            return u_dot + u - (1.0 if (t % 1.0) < 0.5 else 0.0)

        def jacobian(self, u, u_dot, u_ddot, t, weights):
            return np.atleast_2d(weights.alpha0.real + weights.alpha1.real)

    assert smooth_phase(1.0, (0.0, 0.5)) == pytest.approx(0.75)  # the middle of a widest gap, never an edge

    def march(t0, t1, x, bps, _first):
        study = TimeWindow(t0=t0, t1=t1, dt=0.01, integrator=integrator)
        return integrate_transient(
            _Square(), study, x, time_breakpoints=bps, step_control=StepTolerance(rtol=1e-8, atol=1e-10)
        )

    fixed, orbit = periodic_steady_state(march, 1.0, np.zeros(1), edges=(0.0, 0.5), tol=1e-10)
    # closed form at the ON edge t=0: u0 = e^{-1/2}(1 - e^{-1/2}) / (1 - e^{-1}) after the off half
    a = np.exp(-0.5)
    u_on_start = (1.0 - a) * a / (1.0 - a * a)
    t = np.asarray(orbit.times)
    k = int(np.argmin(np.abs((t + 0.5) % 1.0 - 0.5)))  # an ON edge of whichever period the orbit is
    assert float(orbit.states[k][0]) == pytest.approx(u_on_start, abs=1e-5)
    assert fixed.map_calls <= 16
