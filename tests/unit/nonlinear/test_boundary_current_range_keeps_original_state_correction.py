"""A rank-one stored field does not freeze its independent circulating currents."""

import logging
from importlib import import_module

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear.storage_border import storage_border_columns, storage_border_rows
from numerics_lab.nonlinear.finite_storage import finite_storage_actual_rate_initial_pair


def _observed_initial_pair(monkeypatch, tmp_path):
    """Capture only operations the original solve actually performs."""
    border = import_module('numerics_lab.linear.initial_storage_border')
    payload = {}

    def retain(prefix, value):
        if sp.issparse(value):
            matrix = value.tocsr()
            for name in ('data', 'indices', 'indptr', 'shape'):
                payload[prefix + '_' + name] = np.array(getattr(matrix, name), copy=True)
        elif isinstance(value, np.ndarray) and value.dtype.kind in 'biuf':
            payload[prefix] = value.copy()
        elif isinstance(value, (bool, int, float, np.number)):
            payload[prefix] = value

    original_sparse, original_weighted = border._sparse_joint_border, border._weighted_by_row_bars
    original_affine, original_anchors = border.border_affine_space, border.homogeneous_zero_coordinates

    def sparse(matrix, rhs, *args, **options):
        result = original_sparse(matrix, rhs, *args, **options)
        retain('border_matrix', matrix)
        retain('border_rhs', rhs)
        retain('border_directions', options.get('state_directions'))
        retain('border_change', result[0])
        retain('border_rate_directions', result[1])
        return result

    def weighted(matrix, rhs, answer, bars):
        result = original_weighted(matrix, rhs, answer, bars)
        for name, value in (('matrix', matrix), ('rhs', rhs), ('before', answer), ('after', result), ('bars', bars)):
            retain('weighted_' + name, value)
        return result

    def affine(matrix, rhs, **options):
        result = original_affine(matrix, rhs, **options)
        retain('affine_matrix', matrix)
        retain('affine_rhs', rhs)
        retain('affine_answer', result[0])
        retain('affine_directions', result[1])
        return result

    def anchors(matrix, rhs, **options):
        result = original_anchors(matrix, rhs, **options)
        retain('anchor_matrix', matrix)
        retain('anchor_rhs', rhs)
        retain('anchor_unanchored', result[0])
        return result

    monkeypatch.setattr(border, '_sparse_joint_border', sparse)
    monkeypatch.setattr(border, '_weighted_by_row_bars', weighted)
    monkeypatch.setattr(border, 'border_affine_space', affine)
    monkeypatch.setattr(border, 'homogeneous_zero_coordinates', anchors)

    def solve(*args, **options):
        try:
            return finite_storage_actual_rate_initial_pair(*args, **options)
        except ValueError as error:
            if str(error) != 'original boundary state/rate equations did not converge':
                raise
            trace, index = error.__traceback__, 0
            while trace is not None:
                frame = trace.tb_frame
                if frame.f_code.co_filename.endswith('finite_storage.py'):
                    for name, value in frame.f_locals.items():
                        retain(f'frame_{index}_' + name, value)
                    index += 1
                trace = trace.tb_next
            path = tmp_path / 'failed-boundary-current-range.npz'
            np.savez(path, **payload)
            logging.warning('boundary_current_range_diagnostic=%s', path)
            raise

    return solve


@pytest.mark.parametrize('wired', [False, True])
@pytest.mark.parametrize('cubic', [0.0, 20.0])
@pytest.mark.parametrize('mixed_gates', [False, True])
@pytest.mark.parametrize('stored_field', [0.0, -0.000712846])
def test_original_source_rows_can_change_the_bounded_current_state_range(
    wired, cubic, mixed_gates, stored_field, monkeypatch, tmp_path
):
    turns = np.array([1.0, -2.0, 0.5])
    resistance = np.full(3, 1.1)
    voltage = np.full(3, 150.0)
    size = 7  # field, three currents, three physical linkage coordinates
    stiffness, mass = np.zeros((size, size)), np.zeros((size, size))
    stiffness[0, 0], stiffness[0, 1:4] = 2.0, -turns
    stiffness[1:4, 1:4] = np.diag(resistance)
    stiffness[4:, 0], stiffness[4:, 4:] = -turns, np.eye(3)
    mass[1:4, 4:] = np.eye(3)
    mass = sp.csr_matrix(mass)
    source = np.r_[0.0, voltage, np.zeros(3)]
    initial = np.zeros(size)
    current_rows = np.arange(1, 4)
    if wired:
        # Explicit rail sources and conducting/off-state channels add only
        # algebraic coordinates. They must not freeze the same free currents.
        size, current_rows = 17, np.arange(11, 14)
        stiffness, raw_mass = np.zeros((size, size)), np.zeros((size, size))
        stiffness[0, 0], stiffness[0, current_rows] = 2.0, -turns
        off, on = 1e-6, 1e3
        upper = np.array([on, off, off]) if mixed_gates else np.full(3, off)
        lower = np.full(3, off) if mixed_gates else np.full(3, on)
        stiffness[1, 1], stiffness[1, 3:6], stiffness[1, 9] = upper.sum(), -upper, 1.0
        stiffness[2, 2], stiffness[2, 3:6], stiffness[2, 10] = lower.sum(), -lower, -1.0
        for phase, current in enumerate(current_rows):
            node, load = 3 + phase, 6 + phase
            stiffness[node, [1, 2, node, current]] = [-upper[phase], -lower[phase], upper[phase] + lower[phase], 1.0]
            stiffness[load, [load, current]] = [1.0, -1.0]
            stiffness[current, [node, load, current]] = [-1.0, 1.0, 0.1]
            raw_mass[current, 14 + phase] = 1.0
            stiffness[14 + phase, [0, 14 + phase]] = [-turns[phase], 1.0]
        stiffness[9, 1], stiffness[10, 2] = -1.0, 1.0
        source = np.zeros(size)
        source[9:11] = -150.0
        mass = sp.csr_matrix(raw_mass)
        initial = np.zeros(size)
        initial[1:3], initial[3:6] = [150.0, -150.0], -150.0
        resistance = 1.1 + 1.0 / (upper + lower)
        voltage = (upper * 150.0 - lower * 150.0) / (upper + lower)
    bulk, border = np.empty(0, dtype=int), np.arange(size)
    initial[0] = stored_field
    initial[-3:] = turns * stored_field

    def residual(state, rate):
        result = stiffness @ state + mass @ rate - source
        result[0] += cubic * state[0] ** 3
        return result

    def jacobian(state, _rate):
        result = stiffness.copy()
        result[0, 0] += 3 * cubic * state[0] ** 2
        return sp.csr_matrix(result)

    pair = _observed_initial_pair(monkeypatch, tmp_path)(
        initial,
        residual_at=residual,
        jacobian_at=jacobian,
        mass_at=lambda _state, _rate: mass,
        storage_at=lambda state: mass @ state,
        partial_time_at=lambda _state, _rate: np.zeros(size),
        storage_rows_at=lambda _state, _rate: storage_border_rows(mass, bulk_indices=bulk, border_rows=border),
        stored_columns_at=lambda _state, _rate, rows: storage_border_columns(
            mass, bulk_indices=bulk, stored_rows=rows.stored_rows
        ),
        rate_constraint_jacobian_at=jacobian,
        # Q and the three linkage rows fix field state at zero; the bounded
        # current directions have no field component, hence actual mixed G=0.
        rate_constraint_state_columns_at=lambda _state, _rate, directions: sp.csr_matrix((size, directions.shape[1])),
        sparse_state_indices=np.array([0]),
    )
    field_rate = (turns @ (voltage / resistance)) / (turns @ (turns / resistance))
    field_force = 2.0 * stored_field + cubic * stored_field**3
    field_rate -= field_force / (turns @ (turns / resistance))
    expected_current = (voltage - turns * field_rate) / resistance
    np.testing.assert_allclose(pair.state[current_rows], expected_current, rtol=1e-10, atol=0.0)
    assert pair.rate_is_unique[0]
    np.testing.assert_allclose(pair.physical_rate[0], field_rate, rtol=1e-10)
    np.testing.assert_array_equal(mass @ pair.state, mass @ initial)
    np.testing.assert_allclose(residual(pair.state, pair.rate), 0.0, rtol=0.0, atol=1e-10)
