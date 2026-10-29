"""A loose solve bar cannot erase an explicitly owned affine source change."""

import numpy as np
import pytest

from numerics_lab.formulation import IntegratorKind
from numerics_lab.time_integrators import get_integrator
from numerics_lab.time_integrators._step import take_step

pytestmark = pytest.mark.unit


class _HeldVoltage:
    size = 2

    def __init__(self, target, *, steep=False) -> None:
        self.target, self.steep = target, steep
        self.evaluated = []
        self.admissions = []

    def residual(self, state, rate, acceleration, time):
        self.evaluated.append(np.asarray(state).copy())
        difference = state[1] - state[0]
        return np.array([state[0] - self.target, np.sinh(100 * difference) if self.steep else difference])

    def jacobian(self, state, rate, acceleration, time, weights):
        slope = 100 * np.cosh(100 * (state[1] - state[0])) if self.steep else 1.0
        return weights.alpha0.real * np.array([[1.0, 0.0], [-slope, slope]])

    def project_affine_state(self, state, time):
        projected = np.asarray(state).copy()
        projected[0] = self.target
        return projected

    def affine_state_is_admitted(self, state, time):
        # One actual subtraction, with the target treated as the declared
        # held float. This owner permits its own counted arithmetic error,
        # not a guessed solver tolerance or an exact projected-state match.
        eps = np.finfo(float).eps
        bound = eps / (1 - eps) * (abs(state[0]) + abs(self.target))
        accepted = abs(state[0] - self.target) <= bound
        self.admissions.append(bool(accepted))
        return bool(accepted)


def _advance(problem, initial):
    integrator = get_integrator(IntegratorKind.BDF1)
    return take_step(
        problem,
        integrator,
        integrator.start(initial, 0.0),
        0.1,
        0.1,
        solve=np.linalg.solve,
        max_newton=8,
        tol=1e-10,
    )


def test_tiny_actual_held_source_change_is_not_accepted_from_the_old_guess():
    problem = _HeldVoltage(5e-11)
    outcome = _advance(problem, np.zeros(2))
    assert outcome.converged
    np.testing.assert_array_equal(outcome.u, [problem.target, problem.target])
    assert problem.admissions and not problem.admissions[0] and problem.admissions[-1]


def test_a_counted_roundoff_admitted_plane_does_not_require_bitwise_projection_identity():
    target = 1.0
    value = target + np.spacing(target)
    problem = _HeldVoltage(target)
    outcome = _advance(problem, np.full(2, value))
    assert outcome.converged
    np.testing.assert_array_equal(outcome.u, np.full(2, value))
    assert problem.admissions == [True]


def test_steep_clamp_keeps_the_original_unprojected_first_linearization():
    problem = _HeldVoltage(0.1, steep=True)
    outcome = _advance(problem, np.zeros(2))
    assert outcome.converged
    np.testing.assert_array_equal(problem.evaluated[0], np.zeros(2))
    np.testing.assert_array_equal(outcome.u, [0.1, 0.1])
