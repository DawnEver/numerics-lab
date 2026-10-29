"""Actual differentiated-row admission consumes the explicit time producer once."""

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.formulation import ResidualAssemblyBound
from numerics_lab.linear.storage_border import StorageRows
from numerics_lab.nonlinear.finite_storage import finite_storage_actual_rate_initial_pair

pytestmark = pytest.mark.unit


def _pair(evaluation, *, coupling=1.0, **storage):
    mass = sp.diags([1.0, 0.0], format='csr')
    jacobian = sp.csr_matrix([[0.0, -1.0], [1.0, 0.0]])

    def plain_partial(_state, _rate):
        pytest.fail('the supplied evaluation owns both value and bound; do not evaluate Ft twice')

    return finite_storage_actual_rate_initial_pair(
        np.array([1.0, 0.0]),
        residual_at=lambda state, rate: np.array([rate[0] - state[1], state[0] - 1.0]),
        jacobian_at=lambda _state, _rate: jacobian,
        mass_at=lambda _state, _rate: mass,
        storage_at=storage.pop('storage_at', lambda state: mass @ state),
        partial_time_at=plain_partial,
        partial_time_evaluation_at=evaluation,
        storage_rows_at=lambda _state, _rate: StorageRows(np.array([0]), sp.csr_matrix([[0.0, 1.0]])),
        rate_constraint_jacobian_at=lambda _state, _rate: jacobian,
        rate_constraint_state_columns_at=lambda _state, _rate, directions: (
            sp.csr_matrix([[0.0, 0.0], [0.0, coupling]]) @ directions
        ),
        stored_columns_at=lambda _state, _rate, _storage: np.array([0]),
        sparse_state_indices=np.array([0]),
        max_newton=8,
        **storage,
    )


@pytest.mark.parametrize('known', [True, False])
def test_time_evaluation_owns_its_actual_value_without_duplicate_plain_evaluation(known):
    calls = []

    def evaluation(state, rate):
        calls.append((state.copy(), rate.copy()))
        return SimpleNamespace(
            residual=np.array([0.0, state[1] - 2.0]),
            bound=ResidualAssemblyBound(np.zeros(2), np.full(2, known)),
        )

    pair = _pair(evaluation)
    assert calls
    np.testing.assert_array_equal(pair.state, [1.0, 1.0])
    assert pair.rate[0] == 1.0


@pytest.mark.parametrize('invalid', ['value_shape', 'nonfinite_value', 'bound_shape'])
def test_time_evaluation_rejects_malformed_actual_values_or_bound_support(invalid):
    def evaluation(_state, _rate):
        value = np.zeros(1 if invalid == 'value_shape' else 2)
        if invalid == 'nonfinite_value':
            value[1] = np.inf
        size = 1 if invalid == 'bound_shape' else 2
        return SimpleNamespace(residual=value, bound=ResidualAssemblyBound(np.zeros(size), np.ones(size, dtype=bool)))

    with pytest.raises(ValueError, match='time|partial|finite|shape|align|dimension'):
        _pair(evaluation)


@pytest.mark.parametrize('known', [True, False])
def test_time_assembly_allowance_cannot_hide_a_genuine_mixed_physical_contradiction(known):
    def evaluation(state, _rate):
        return SimpleNamespace(
            residual=np.array([0.0, -state[1] - 2.0]),
            bound=ResidualAssemblyBound(np.full(2, 1e-15), np.full(2, known)),
        )

    # The contradiction is refused by the AUTHORITY check, before any allowance is applied -- so the bound,
    # known or not, never reaches it. Pinned to that one message: any other failure is not this proof.
    with pytest.raises(ValueError, match='require additional physical state authority'):
        _pair(evaluation, coupling=-1.0)


def _exact_time(_state, _rate):
    return SimpleNamespace(residual=np.zeros(2), bound=ResidualAssemblyBound(np.zeros(2), np.ones(2, dtype=bool)))


def _exact_storage(seen=None):
    mass = sp.diags([1.0, 0.0], format='csr')

    def evaluation(state):
        if seen is not None:
            seen.append(state.tobytes())
        return SimpleNamespace(storage=mass @ state, bound=ResidualAssemblyBound(np.zeros(2), np.ones(2, dtype=bool)))

    return evaluation


def test_a_storage_evaluation_owns_the_charge_and_is_evaluated_once_per_state():
    """The evaluation, not the plain ``storage_at``, is the charge; ``charge_at`` and the allowance share it."""
    seen = []

    def plain_storage(_state):
        pytest.fail('the supplied storage evaluation owns the charge; do not evaluate Q twice')

    _pair(_exact_time, storage_at=plain_storage, storage_evaluation_at=_exact_storage(seen))
    assert seen
    assert len(seen) == len(set(seen)), 'one storage evaluation per distinct state'


def test_a_zero_storage_bound_admits_exactly_the_plain_pair():
    """PLANTED CONTROL: an exact storage producer changes nothing against the path that has none."""
    plain = _pair(_exact_time)
    bounded = _pair(_exact_time, storage_evaluation_at=_exact_storage())
    np.testing.assert_array_equal(bounded.state, plain.state)
    np.testing.assert_array_equal(bounded.rate, plain.rate)
