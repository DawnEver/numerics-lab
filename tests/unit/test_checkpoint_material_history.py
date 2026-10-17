"""A checkpoint restores committed material history with the numerical BDF history."""

import json

import numpy as np
import pytest

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import integrate_transient
from numerics_lab.time_integrators.checkpoint import TransientCheckpoint, restore_history_checkpoint


class RememberingProblem:
    has_history = True

    def __init__(self) -> None:
        self.memory = 0.0
        self.last_time = 0.0

    def residual(self, u, u_dot, u_ddot, t):
        return u_dot + u - self.memory - t

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.eye(1) * (weights.alpha0.real + weights.alpha1.real)

    def commit_history(self, u, t):
        self.memory += float(u[0]) * (t - self.last_time)
        self.last_time = t

    def checkpoint_history(self):
        return {'memory': self.memory, 'last_time': self.last_time}

    def restore_checkpoint_history(self, data):
        self.memory, self.last_time = data['memory'], data['last_time']


def test_checkpoint_preserves_actual_previous_accepted_endpoint_bits():
    source_edge = 0.0003
    motion_edge = 0.0008854084409738354
    assert motion_edge - (motion_edge - source_edge) != source_edge
    solved = integrate_transient(
        RememberingProblem(),
        TimeWindow(0.0, motion_edge, 0.001),
        np.zeros(1),
        step_times=(0.0, source_edge, motion_edge),
        checkpoint_key='exact-source-clock',
    )
    restored = TransientCheckpoint.from_mapping(json.loads(json.dumps(solved.checkpoint.to_mapping())))
    assert solved.times[-2] == source_edge
    assert restored.previous_time == source_edge
    assert restored.time == motion_edge


def test_committed_material_and_bdf_history_restart_together():
    whole = integrate_transient(RememberingProblem(), TimeWindow(0.0, 1.0, 0.1), np.zeros(1))
    first = integrate_transient(RememberingProblem(), TimeWindow(0.0, 0.5, 0.1), np.zeros(1), checkpoint_key='memory')
    checkpoint = TransientCheckpoint.from_mapping(json.loads(json.dumps(first.checkpoint.to_mapping())))
    resumed = integrate_transient(
        RememberingProblem(), TimeWindow(0.5, 1.0, 0.1), np.zeros(1), restart=checkpoint, checkpoint_key='memory'
    )
    np.testing.assert_allclose(resumed.states, whole.states[5:], rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize('payload', [{'state': float('nan')}, {'state': object()}, {1: 2.0}])
def test_material_history_rejects_non_json_or_nonfinite_payloads(payload):
    with pytest.raises(ValueError, match='checkpoint'):
        TransientCheckpoint('memory', 'bdf2', 0.5, 0.1, ((0.0,), (1.0,)), material_history=payload)


def test_history_preserves_json_boolean_metadata_without_numeric_coercion():
    payload = {'rate_is_unique': [True, False], 'model': {'conducting': True}}
    checkpoint = TransientCheckpoint('memory', 'bdf2', 0.5, 0.1, ((0.0,), (1.0,)), material_history=payload)
    restored = TransientCheckpoint.from_mapping(json.loads(json.dumps(checkpoint.to_mapping())))
    assert restored.material_history == payload
    assert restored.material_history['rate_is_unique'][0] is True
    assert restored.material_history['rate_is_unique'][1] is False


@pytest.mark.parametrize('states', [((True,), (1.0,)), ((0.0,), (False,))])
def test_boolean_metadata_permission_does_not_admit_boolean_numeric_states(states):
    with pytest.raises(ValueError, match='not booleans'):
        TransientCheckpoint('memory', 'bdf2', 0.5, 0.1, states)


class Oscillator:
    def residual(self, u, u_dot, u_ddot, t):
        return u_ddot + 4 * u

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.eye(1) * (weights.alpha2.real + 4 * weights.alpha0.real)


def test_newmark_restart_keeps_velocity_and_acceleration():
    whole = integrate_transient(
        Oscillator(),
        TimeWindow(0.0, 1.0, 0.01, IntegratorKind.NEWMARK),
        np.ones(1),
        v0=np.zeros(1),
        a0=np.array([-4.0]),
    )
    first = integrate_transient(
        Oscillator(),
        TimeWindow(0.0, 0.5, 0.01, IntegratorKind.NEWMARK),
        np.ones(1),
        v0=np.zeros(1),
        a0=np.array([-4.0]),
        checkpoint_key='oscillator',
    )
    resumed = integrate_transient(
        Oscillator(),
        TimeWindow(0.5, 1.0, 0.01, IntegratorKind.NEWMARK),
        np.zeros(1),
        restart=first.checkpoint,
        checkpoint_key='oscillator',
    )
    np.testing.assert_allclose(resumed.states, whole.states[50:], rtol=1e-12, atol=1e-13)


def test_second_order_checkpoint_cannot_omit_derivative_history():
    with pytest.raises(ValueError, match='integrator order'):
        TransientCheckpoint('oscillator', 'newmark', 0.5, 0.01, ((1.0,),))


@pytest.mark.parametrize('stateful', [False, True])
def test_history_presence_refusal_does_not_mutate_actual_physical_context(stateful):
    class ContextualProblem(RememberingProblem):
        def __init__(self) -> None:
            super().__init__()
            self.has_history = stateful
            self.context = {'origin': 0.0}

        def checkpoint_context(self, time):
            return self.context.copy()

        def restore_checkpoint_context(self, data):
            self.context = data.copy()

    problem = ContextualProblem()
    checkpoint = TransientCheckpoint(
        'memory',
        'bdf2',
        0.5,
        0.1,
        ((0.0,), (1.0,)),
        material_history=None if stateful else {'memory': 1.0, 'last_time': 0.5},
        problem_context={'origin': 0.5},
    )
    with pytest.raises(ValueError, match='material history'):
        restore_history_checkpoint(problem, checkpoint)
    assert problem.context == {'origin': 0.0}
    assert problem.memory == 0.0 and problem.last_time == 0.0
