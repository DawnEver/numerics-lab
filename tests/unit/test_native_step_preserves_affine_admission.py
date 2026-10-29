"""A speculative native Newton result cannot bypass owned affine admission."""

import numpy as np
import pytest

from numerics_lab.formulation import IntegratorKind
from numerics_lab.time_integrators import get_integrator
from numerics_lab.time_integrators._step import take_step

pytestmark = pytest.mark.unit


class _NativeHeldSource:
    target = 5e-11

    def __init__(self, admitted) -> None:
        self.native_admitted = admitted
        self.native_calls = 0
        self.residual_calls = 0
        self.admissions = []

    def residual(self, state, rate, acceleration, time):
        self.residual_calls += 1
        return state - self.target

    def jacobian(self, state, rate, acceleration, time, weights):
        return weights.alpha0.real * np.eye(1)

    def affine_state_is_admitted(self, state, time):
        eps = np.finfo(float).eps
        result = abs(state[0] - self.target) <= eps / (1 - eps) * (abs(state[0]) + abs(self.target))
        self.admissions.append(bool(result))
        return bool(result)

    def native_step(self, form, guess, **options):
        self.native_calls += 1
        candidate = np.full(1, self.target) if self.native_admitted else np.asarray(guess).copy()
        # The native contract returns a speculative Newton result, not accepted
        # history. No source or history is advanced by this pure calculation.
        return candidate, True, 1, float(abs(candidate[0] - self.target)), False


@pytest.mark.parametrize('admitted', [False, True])
def test_native_candidate_requires_the_same_owned_affine_admission(admitted):
    problem = _NativeHeldSource(admitted)
    integrator = get_integrator(IntegratorKind.BDF1)
    initial = np.zeros(1)
    outcome = take_step(
        problem,
        integrator,
        integrator.start(initial, 0.0),
        0.1,
        0.1,
        solve=np.linalg.solve,
        max_newton=2,
        tol=1e-10,
    )
    assert outcome.converged
    np.testing.assert_array_equal(outcome.u, [problem.target])
    np.testing.assert_array_equal(initial, [0.0])
    assert problem.native_calls == 1
    assert problem.admissions and problem.admissions[-1]
    assert bool(problem.residual_calls) != admitted
