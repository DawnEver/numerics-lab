"""Outgoing numerical pairs cannot manufacture real values from booleans."""

import numpy as np
import pytest

from numerics_lab.time_integrators._landing import AcceptedProjectionResult


@pytest.mark.parametrize('name', ['state', 'rate', 'acceleration'])
@pytest.mark.parametrize('value', [[True, False], np.array([True, False]), [0.0, True], [np.bool_(False), 1.0]])
def test_outgoing_pair_rejects_boolean_coordinates_before_numeric_coercion(name, value):
    fields = {'state': np.array([0.0, 1.0]), name: value}
    with pytest.raises(ValueError, match='finite real'):
        AcceptedProjectionResult(**fields)


def test_real_integer_pairs_remain_owned_without_coercing_restart_provenance():
    token = object()
    source = np.array([0, 1])
    pair = AcceptedProjectionResult(source, [2, 3], [4, 5], prepared_restart=token)
    source[:] = 7
    np.testing.assert_array_equal(pair.state, [0.0, 1.0])
    np.testing.assert_array_equal(pair.rate, [2.0, 3.0])
    np.testing.assert_array_equal(pair.acceleration, [4.0, 5.0])
    assert pair.prepared_restart is token
    assert not pair.state.flags.writeable
