"""Bounded-error output decimation: fewer stored points, interpolation error audited against the bound.

Planted control: a uniform stride keeping the SAME number of points misses the bound -- so meeting it
is the greedy chord test's doing, not a property of any thinning that size.
"""

from __future__ import annotations

import numpy as np
import pytest

from motronics.contracts.spec.study_spec import IntegratorKind, TransientStudy
from motronics.gauss.time_integrators import integrate_transient
from motronics.gauss.time_integrators.decimate import decimate, interpolation_error


class _Driven:
    """``u_0' = cos(8 t^2)`` (a chirp -- smooth early, fast late) and ``u_1' = -u_1``."""

    def residual(self, u, u_dot, u_ddot, t):
        return u_dot - np.array([np.cos(8.0 * t * t), -u[1]])

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.diag([weights.alpha1.real, weights.alpha1.real + weights.alpha0.real])


@pytest.fixture(scope='module')
def marched():
    study = TransientStudy(t0=0.0, t1=2.0, dt=1e-3, integrator=IntegratorKind.TRAPEZOID)
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
        TransientStudy(t0=0.0, t1=1.0, dt=0.01),
        np.zeros(1),
        event_fns=(lambda u, _t: float(u[0]) - 0.333,),
    )
    thin = decimate(sol, 1.0)
    t_event = sol.events[0][1]
    assert t_event in thin.times
    assert len(thin.times) == 3
