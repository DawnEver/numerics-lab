"""Nonlinear EEC resumes its actual integrated tangent, not an affine surrogate."""

import json

import numpy as np
import pytest
from tests.unit.test_nonlinear_eec_cycle_tangent import Cubic

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import integrate_transient
from numerics_lab.time_integrators.checkpoint import TransientCheckpoint


def _run(start, end, restart=None):
    return integrate_transient(
        Cubic(),
        TimeWindow(start, end, 0.05, IntegratorKind.BDF1),
        np.array([0.7]),
        max_newton=30,
        eec_period=1.0,
        eec_nonlinear_first_order=True,
        checkpoint_key='nonlinear-eec',
        restart=restart,
    )


# THE SPLIT IS A LATTICE TICK (`7 * 0.05`, not the literal 0.35): the uninterrupted march lands
# `origin + k dt`, and a first run stopped at a nearby declared float takes a last step one ULP short,
# so a bit-for-bit resume claim is only meaningful at the tick the whole run actually visits.
@pytest.mark.parametrize('split', [7 * 0.05, 20 * 0.05])
def test_nonlinear_eec_json_restart_preserves_the_real_cycle_tangent_and_future_states(split):
    whole = _run(0.0, 2.1)
    first = _run(0.0, split)
    mapping = json.loads(json.dumps(first.checkpoint.to_mapping(), allow_nan=False))
    tangent = mapping['eec_state']['cycle_tangent']
    assert tangent['shape'] == [1, 1]
    assert tangent['indptr'] == [0, len(tangent['data'])]
    if split < 1.0:
        expected = 0.05 * sum(Cubic.stiffness + 3 * Cubic.cubic * state[0] ** 2 for state in first.states[1:])
        assert tangent['data'] == pytest.approx([expected])
    else:
        assert tangent['data'] == []  # canonical reset at the corrected boundary
    saved = TransientCheckpoint.from_mapping(mapping)
    resumed = _run(split, 2.1, saved)
    index = round(split / 0.05)
    np.testing.assert_array_equal(resumed.states[1:], whole.states[index + 1 :])
    assert resumed.checkpoint.to_mapping() == whole.checkpoint.to_mapping()


@pytest.mark.parametrize(
    'change',
    [
        {'shape': [2, 2]},
        {'indptr': [0, 2]},
        {'indices': [1]},
        {'data': [float('nan')]},
        {'indices': [0, 0], 'data': [1.0, 2.0], 'indptr': [0, 2]},
    ],
)
def test_malformed_nonlinear_eec_csr_is_rejected_before_restore(change):
    mapping = _run(0.0, 0.35).checkpoint.to_mapping()
    mapping['eec_state']['cycle_tangent'].update(change)
    with pytest.raises(ValueError, match='CSR|tangent|dimension'):
        TransientCheckpoint.from_mapping(mapping)


def test_nonlinear_checkpoint_cannot_silently_switch_to_affine_eec():
    saved = _run(0.0, 0.35).checkpoint
    with pytest.raises(ValueError, match='nonlinear|model|mode'):
        integrate_transient(
            Cubic(),
            TimeWindow(0.35, 2.1, 0.05, IntegratorKind.BDF1),
            np.array([0.7]),
            eec_period=1.0,
            eec_affine_first_order=True,
            restart=saved,
            checkpoint_key='nonlinear-eec',
        )


def test_canonical_tangent_csr_refuses_explicit_zero_entries():
    mapping = _run(0.0, 0.35).checkpoint.to_mapping()
    mapping['eec_state']['cycle_tangent']['data'] = [0.0]
    with pytest.raises(ValueError, match='CSR|canonical'):
        TransientCheckpoint.from_mapping(mapping)


def test_restart_keeps_a_failed_cycle_flag_and_does_not_correct_its_converged_endpoint():
    class FailingStep(Cubic):
        def residual(self, u, v, a, t):
            if np.isclose(t, 0.1, rtol=0.0, atol=1e-14):
                return np.ones_like(u)
            return super().residual(u, v, a, t)

    options = {
        'max_newton': 30,
        'eec_period': 1.0,
        'eec_nonlinear_first_order': True,
        'checkpoint_key': 'failed-cycle',
        'on_nonconverged': 'label',
    }
    first = integrate_transient(
        FailingStep(), TimeWindow(0.0, 7 * 0.05, 0.05, IntegratorKind.BDF1), np.array([0.7]), **options
    )
    saved = TransientCheckpoint.from_mapping(json.loads(json.dumps(first.checkpoint.to_mapping(), allow_nan=False)))
    assert not saved.eec_state.cycle_converged
    resumed = integrate_transient(
        FailingStep(),
        TimeWindow(7 * 0.05, 1.05, 0.05, IntegratorKind.BDF1),
        np.array([0.7]),
        restart=saved,
        **options,
    )
    plain = integrate_transient(
        FailingStep(),
        TimeWindow(0.0, 1.05, 0.05, IntegratorKind.BDF1),
        np.array([0.7]),
        max_newton=30,
        on_nonconverged='label',
    )
    assert resumed.converged
    assert resumed.corrected_times == []
    np.testing.assert_array_equal(resumed.states[1:], plain.states[8:])
    assert resumed.checkpoint.eec_state.cycle_converged
