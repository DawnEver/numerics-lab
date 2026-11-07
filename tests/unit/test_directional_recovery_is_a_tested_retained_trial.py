"""UNVERIFIED: affine directional recovery owns evaluated-point provenance, not idempotence."""

from asyncio import CancelledError as AsyncCancelledError
from concurrent.futures import CancelledError as FutureCancelledError

import numpy as np
import pytest
from tests.unit.test_newton_trials_use_the_problem_owned_limit import _advance

import numerics_lab.time_integrators._step as step_module
from numerics_lab.line_search import TrialNotEvaluable

pytestmark = pytest.mark.unit


class _Projected:
    size = 2

    def residual(self, u, rate, acceleration, time):
        return u - np.array([1.0, 10.0])

    def jacobian(self, u, rate, acceleration, time, weights):
        return np.eye(2) * weights.alpha0.real

    def project_affine_state(self, u, time):
        u[0] = 1.0 if u[0] < 1.0 else np.nextafter(u[0], np.inf)
        return u

    def limit_newton_trial(self, previous, proposed, time):
        proposed[1] = previous[1] + np.clip(proposed[1] - previous[1], -1.0, 1.0)
        return proposed


def test_near_idempotent_projector_reaches_real_directional_search_after_tested_recovery(monkeypatch):
    problem, starts = _Projected(), []
    search = step_module.directional_line_search

    def observed(residual, previous, step, r0, **options):
        starts.append(previous.copy())
        return search(residual, previous, step, r0, **options)

    monkeypatch.setattr(step_module, 'directional_line_search', observed)
    result = _advance(problem, line_search='directional')
    assert result.converged and starts
    np.testing.assert_array_equal(starts[0], [np.nextafter(1.0, np.inf), 1.0])
    assert problem.project_affine_state(starts[0].copy(), 1.0)[0] != starts[0][0]


@pytest.mark.parametrize('cap', [0, 3])
@pytest.mark.parametrize('unevaluable', [False, True])
def test_rejected_recovery_cannot_replace_the_retained_point_even_at_the_small_budget(cap, unevaluable):
    class Refuses(_Projected):
        def residual(self, u, rate, acceleration, time):
            if unevaluable and u[0] != 0.0:
                msg = 'projected endpoint is outside this law domain'
                raise TrialNotEvaluable(msg)
            return np.array([100.0 * u[0] + 1.0, u[1] - 1.0])

        def jacobian(self, u, rate, acceleration, time, weights):
            return np.diag([100.0, 1.0]) * weights.alpha0.real

    result = _advance(Refuses(), line_search='directional', backtrack_cap=cap, max_newton=2)
    assert not result.converged
    np.testing.assert_array_equal(result.u, [0.0, 0.0])
    assert result.residual > 1.0


@pytest.mark.parametrize(
    'failure_type', [ValueError, KeyboardInterrupt, SystemExit, AsyncCancelledError, FutureCancelledError]
)
def test_recovery_does_not_convert_actual_callback_failure_or_stop_into_an_unevaluable_trial(failure_type):
    stop = failure_type('actual callback failure')
    cause = RuntimeError('original provenance')
    stop.__cause__ = cause

    class Stops(_Projected):
        def residual(self, u, rate, acceleration, time):
            if u[0] != 0.0:
                raise stop from cause
            return super().residual(u, rate, acceleration, time)

    with pytest.raises(failure_type) as caught:
        _advance(Stops(), line_search='directional', max_newton=2)
    assert caught.value is stop and stop.__cause__ is cause
