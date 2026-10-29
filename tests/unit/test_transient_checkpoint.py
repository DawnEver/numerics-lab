"""A restart preserves numerical history and refuses a different initial-value problem."""

import json

import numpy as np
import pytest
from tests.unit.test_time_integrators import DecayProblem

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import integrate_transient
from numerics_lab.time_integrators.checkpoint import TransientCheckpoint, model_digest


def test_checkpoint_roundtrips_without_executable_payload():
    record = TransientCheckpoint('decay', 'bdf2', 0.2, 0.1, ((1.0,), (0.9,)))
    restored = TransientCheckpoint.from_mapping(json.loads(json.dumps(record.to_mapping())))
    assert restored == record
    with pytest.raises(ValueError, match='does not match'):
        restored.validate_restart('another', 'bdf2', 0.2, 1)


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('schema_version', True),
        ('schema_version', 1),
        ('time', True),
        ('time', float('nan')),
        ('previous_dt', 0.0),
        ('previous_dt', '0.1'),
        ('previous_time', -1.0),
        ('states', [[True]]),
        ('states', [[float('nan')]]),
        ('states', [[1.0], [1.0], [1.0]]),
    ],
)
def test_checkpoint_refuses_malformed_json(field, value):
    data = TransientCheckpoint('decay', 'bdf2', 0.2, 0.1, ((1.0,), (0.9,))).to_mapping()
    data[field] = value
    with pytest.raises(ValueError, match='checkpoint'):
        TransientCheckpoint.from_mapping(data)


def test_changed_model_cannot_keep_the_same_checkpoint_identity():
    original = model_digest({'namespace': 'same', 'operator': np.array([[1.0]]), 'load': [1.0]})
    changed = model_digest({'namespace': 'same', 'operator': np.array([[2.0]]), 'load': [1.0]})
    record = TransientCheckpoint(original, 'bdf2', 0.2, 0.1, ((1.0,), (0.9,)))
    with pytest.raises(ValueError, match='does not match'):
        record.validate_restart(changed, 'bdf2', 0.2, 1)


@pytest.mark.parametrize('context', [{'origin': True}, {'clock': float('nan')}, {'large': list(range(4097))}])
def test_problem_context_is_finite_numeric_and_bounded(context):
    with pytest.raises(ValueError, match='checkpoint'):
        TransientCheckpoint('decay', 'bdf2', 0.2, 0.1, ((1.0,), (0.9,)), problem_context=context)


def test_checkpoint_callback_receives_each_accepted_step():
    records = []
    result = integrate_transient(
        DecayProblem(1.0),
        TimeWindow(0.0, 0.3, 0.1),
        np.array([1.0]),
        checkpoint_key='decay',
        on_checkpoint=records.append,
    )
    assert [record.time for record in records] == result.times[1:]
    assert records[-1] == result.checkpoint


def test_bdf2_restart_matches_an_uninterrupted_march():
    problem = DecayProblem(1.0)
    whole = integrate_transient(problem, TimeWindow(0.0, 1.0, 0.1), np.array([1.0]))
    first = integrate_transient(
        problem,
        TimeWindow(0.0, 0.5, 0.1),
        np.array([1.0]),
        checkpoint_key='decay',
    )
    restored = TransientCheckpoint.from_mapping(first.checkpoint.to_mapping())
    last = integrate_transient(
        problem,
        TimeWindow(0.5, 1.0, 0.1),
        first.states[-1],
        restart=restored,
        checkpoint_key='decay',
    )
    np.testing.assert_allclose(last.states, whole.states[5:], rtol=1e-13, atol=1e-14)


def test_fixed_json_restart_preserves_absolute_parent_products_exactly():
    dt = 0.25e-6
    split = 10 * dt
    end = 20 * dt
    whole = integrate_transient(DecayProblem(1.0), TimeWindow(0.0, end, dt), np.array([1.0]), checkpoint_key='decay')
    first = integrate_transient(DecayProblem(1.0), TimeWindow(0.0, split, dt), np.array([1.0]), checkpoint_key='decay')
    restored = TransientCheckpoint.from_mapping(json.loads(json.dumps(first.checkpoint.to_mapping())))
    resumed = integrate_transient(
        DecayProblem(1.0),
        TimeWindow(split, end, dt),
        first.states[-1],
        checkpoint_key='decay',
        restart=restored,
    )
    assert restored.step_origin == 0.0
    assert resumed.times == whole.times[10:]
    assert np.array_equal(resumed.states, whole.states[10:])
    assert resumed.checkpoint.to_mapping() == whole.checkpoint.to_mapping()


def test_restart_refuses_a_different_integrator():
    record = TransientCheckpoint('decay', 'bdf2', 0.2, 0.1, ((1.0,), (0.9,)))
    with pytest.raises(ValueError, match='does not match'):
        integrate_transient(
            DecayProblem(1.0),
            TimeWindow(0.2, 0.4, 0.1, integrator=IntegratorKind.BDF1),
            np.array([0.9]),
            restart=record,
            checkpoint_key='decay',
        )


def test_an_array_identity_is_its_values_whatever_its_memory_layout():
    """The digest reads the array's VALUES: a Fortran-ordered or strided copy of the same values is the same model."""
    values = np.arange(12.0).reshape(3, 4)
    assert model_digest({'k': values}) == model_digest({'k': np.asfortranarray(values)})
    changed = values.copy()
    changed[2, 3] = np.nextafter(changed[2, 3], 0.0)
    assert model_digest({'k': values}) != model_digest({'k': changed})


def test_a_non_finite_array_value_has_no_identity():
    with pytest.raises(ValueError, match='non-finite'):
        model_digest({'k': np.array([1.0, np.nan])})
