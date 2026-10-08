"""Accepted lattice commands restart BDF history without off-grid observations."""

import numpy as np
import pytest

from motronics.contracts.spec.study_spec import IntegratorKind, TransientStudy
from motronics.gauss.time_integrators import integrate_transient

pytestmark = pytest.mark.unit


class _Forced:
    forcing = 0.0

    def residual(self, u, u_dot, u_ddot, t):
        return u_dot - self.forcing

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.eye(1) * weights.alpha1.real


def test_landing_dispatches_only_committed_states_and_restarts_history():
    problem = _Forced()
    observed = []

    def landing(time, state):
        observed.append((time, state.copy()))
        assert not state.flags.writeable
        if time == pytest.approx(0.2):
            problem.forcing = 3.0
            return (7,)
        return ()

    solution = integrate_transient(
        problem,
        TransientStudy(t0=0.0, t1=0.4, dt=0.1, integrator=IntegratorKind.BDF2),
        np.zeros(1),
        on_landing=landing,
    )
    assert [time for time, _ in observed] == solution.times[1:]
    assert solution.landed_events == [(7, pytest.approx(0.2))]
    # BE restart gives .3; retaining pre-command BDF2 history would incorrectly give .2.
    assert np.array(solution.states).ravel() == pytest.approx([0.0, 0.0, 0.0, 0.3, 0.6])
    assert np.array(solution.state_rates).ravel() == pytest.approx([0.0, 0.0, 3.0, 3.0])


@pytest.mark.parametrize('indices', [(True,), (-1,), (0, 0), (1.5,)])
def test_invalid_landing_identifiers_are_not_published(indices):
    with pytest.raises(ValueError, match='event identifiers'):
        integrate_transient(
            _Forced(),
            TransientStudy(t0=0.0, t1=0.1, dt=0.1, integrator=IntegratorKind.BDF1),
            np.zeros(1),
            on_landing=lambda _time, _state: indices,
        )
