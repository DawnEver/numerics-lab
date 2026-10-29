"""Bounded-error output decimation: fewer stored points, interpolation error audited against the bound.

Planted control: a uniform stride keeping the SAME number of points misses the bound -- so meeting it
is the greedy chord test's doing, not a property of any thinning that size.
"""

from __future__ import annotations

import numpy as np
import pytest

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import TransientSolution, integrate_transient
from numerics_lab.time_integrators.decimate import decimate, interpolation_error


class _Driven:
    """``u_0' = cos(8 t^2)`` (a chirp -- smooth early, fast late) and ``u_1' = -u_1``."""

    def residual(self, u, u_dot, u_ddot, t):
        return u_dot - np.array([np.cos(8.0 * t * t), -u[1]])

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.diag([weights.alpha1.real, weights.alpha1.real + weights.alpha0.real])


@pytest.fixture(scope='module')
def marched():
    study = TimeWindow(t0=0.0, t1=2.0, dt=1e-3, integrator=IntegratorKind.TRAPEZOID)
    return integrate_transient(_Driven(), study, np.array([0.0, 1.0]))


@pytest.mark.parametrize('bound', [1e-3, np.array([1e-4, 1e-2])])
def test_points_drop_and_every_dropped_point_is_within_the_bound(marched, bound) -> None:
    thin = decimate(marched, bound)
    err = interpolation_error(marched.times, marched.states, thin.times, thin.states)
    print(f'bound {bound}: {len(marched.times)} -> {len(thin.times)} points, max error {err}')  # noqa: T201
    assert len(thin.times) < len(marched.times) / 5
    assert np.all(err <= np.broadcast_to(bound, err.shape))
    assert thin.times[0] == marched.times[0]
    assert thin.times[-1] == marched.times[-1]


def test_planted_control_a_uniform_stride_of_the_same_size_misses_the_bound(marched) -> None:
    thin = decimate(marched, 1e-3)
    idx = np.unique(np.linspace(0, len(marched.times) - 1, len(thin.times)).round().astype(int))
    stride_t = [marched.times[i] for i in idx]
    stride_u = [marched.states[i] for i in idx]
    assert np.max(interpolation_error(marched.times, marched.states, stride_t, stride_u)) > 1e-3


def test_an_event_instant_is_always_kept() -> None:
    class _Ramp:
        def residual(self, u, u_dot, u_ddot, t):
            return u_dot - 1.0

        def jacobian(self, u, u_dot, u_ddot, t, weights):
            return np.atleast_2d(weights.alpha1.real)

    sol = integrate_transient(
        _Ramp(),
        TimeWindow(t0=0.0, t1=1.0, dt=0.01),
        np.zeros(1),
        event_fns=(lambda u, _t: float(u[0]) - 0.333,),
    )
    thin = decimate(sol, 1.0)
    t_event = sol.events[0][1]
    assert t_event in thin.times
    assert len(thin.times) == 3


@pytest.mark.parametrize('bound', [-1.0, float('nan'), float('inf')])
def test_an_error_bound_must_be_finite_and_nonnegative(bound) -> None:
    sol = TransientSolution(times=[0.0, 1.0], states=[np.zeros(1), np.ones(1)])
    with pytest.raises(ValueError, match='bound'):
        decimate(sol, bound)


@pytest.mark.parametrize('times', [[0.0, 0.0], [1.0, 0.0], [0.0, float('nan')]])
def test_a_compression_certificate_requires_finite_increasing_times(times) -> None:
    sol = TransientSolution(times=times, states=[np.zeros(1), np.ones(1)])
    with pytest.raises(ValueError, match='times'):
        decimate(sol, 1.0)


def test_compression_preserves_landed_diagnostics_and_residual_side_states() -> None:
    sol = TransientSolution(
        times=list(np.arange(6, dtype=float)),
        states=[np.array([float(i)]) for i in range(6)],
        state_rates=[np.ones(1) for _ in range(5)],
        rate_state_overrides={1: np.array([-7.0])},
        nonconverged_times=[3.0],
        floor_converged_times=[4.0],
        events=[(0, 2.0)],
    )
    thin = decimate(sol, 10.0)
    assert set(sol.nonconverged_times + sol.floor_converged_times) <= set(thin.times)
    assert thin.rate_state_overrides is not None
    index = thin.times.index(2.0) - 1
    np.testing.assert_array_equal(thin.rate_state_overrides[index], [-7.0])
    assert not thin.converged
    assert sol.rate_state_overrides == {1: np.array([-7.0])}


def test_a_one_point_trajectory_and_discarded_probe_labels_remain_honest() -> None:
    sol = TransientSolution(
        times=[0.0],
        states=[np.ones(1)],
        state_rates=[],
        rate_state_overrides={},
        floor_converged_times=[0.25],
    )
    thin = decimate(sol, 0.0)
    assert thin.times == [0.0]
    assert thin.state_rates == []
    assert thin.rate_state_overrides == {}
    assert thin.floor_converged_times == [0.25]


def test_nonfinite_states_cannot_receive_an_error_certificate() -> None:
    sol = TransientSolution(times=[0.0, 1.0], states=[np.zeros(1), np.array([np.nan])])
    with pytest.raises(ValueError, match='finite'):
        decimate(sol, 1.0)
