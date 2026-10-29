"""Successful floating factorization does not prove original boundary state rank."""

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear.storage_border import StorageRows
from numerics_lab.nonlinear import finite_storage as runtime


@pytest.mark.parametrize('value', [1.0, 1e-30])
@pytest.mark.parametrize('admitted', [True, False])
def test_sparse_boundary_correction_preserves_only_admitted_stored_particular(monkeypatch, value, admitted):
    initial = np.array([value, 0.0])
    particular = initial.copy()
    particular[0] = np.nextafter(value, np.inf) if admitted else 2 * value
    mass = sp.diags([1.0, 0.0], format='csr')
    storage = StorageRows(np.array([0]), sp.csr_matrix([[0.0, 1.0]]))

    def state_authority(*_args):
        msg = 'explicit boundary direction'
        raise runtime.StorageRateAuthorityRequired(
            msg,
            state=particular,
            directions=sp.csr_matrix([[0.0], [1.0]]),
        )

    monkeypatch.setattr(runtime, '_actual_finite_storage_state', state_authority)
    kwargs = {
        'residual_at': lambda state, rate: state + mass @ rate,
        'jacobian_at': lambda _state, _rate: sp.eye(2, format='csr'),
        'mass_at': lambda _state, _rate: mass,
        'storage_at': lambda state: mass @ state,
        'partial_time_at': lambda _state, _rate: np.zeros(2),
        'storage_rows_at': lambda _state, _rate: storage,
        'rate_constraint_jacobian_at': lambda _state, _rate: sp.eye(2, format='csr'),
        'sparse_state_indices': np.array([0]),
    }
    if not admitted:
        with pytest.raises(ValueError, match='boundary particular contradicts original finite charge'):
            runtime.finite_storage_actual_rate_initial_pair(initial, **kwargs)
        return
    pair = runtime.finite_storage_actual_rate_initial_pair(initial, **kwargs)
    np.testing.assert_array_equal(pair.state, particular)
    np.testing.assert_array_equal(mass @ pair.state, mass @ particular)
    assert pair.state[0] != 0.0


def test_boundary_tangent_direction_does_not_authorize_changing_nonlinear_finite_charge():
    with pytest.raises(ValueError, match='boundary state direction contradicts original finite charge'):
        runtime.finite_storage_actual_rate_initial_pair(
            np.zeros(2),
            residual_at=lambda state, rate: np.array([2 * state[0] * rate[0] + rate[1] + state[0] - 1, state[1]]),
            jacobian_at=lambda _state, rate: sp.diags([2 * rate[0] + 1, 1], format='csr'),
            mass_at=lambda state, _rate: sp.csr_matrix([[2 * state[0], 1.0], [0.0, 0.0]]),
            storage_at=lambda state: np.array([state[0] ** 2 + state[1], 0.0]),
            partial_time_at=lambda _state, _rate: np.zeros(2),
            storage_rows_at=lambda _state, _rate: StorageRows(np.array([0]), sp.csr_matrix([[0.0, 1.0]])),
            stored_columns_at=lambda _state, _rate, _storage: np.array([1]),
            rate_constraint_jacobian_at=lambda _state, _rate: sp.eye(2, format='csr'),
            sparse_state_indices=np.array([], dtype=int),
        )


@pytest.mark.parametrize('declared', [False, True])
def test_genuinely_tiny_independent_boundary_is_not_declared_dependent(declared):
    tangent = sp.diags([1.0, 1e-30], format='csr')
    storage = StorageRows(np.array([], dtype=int), sp.eye(2, format='csr'))
    pair = runtime.finite_storage_actual_rate_initial_pair(
        np.zeros(2),
        residual_at=lambda state, _rate: tangent @ state,
        jacobian_at=lambda _state, _rate: tangent,
        mass_at=lambda _state, _rate: sp.csr_matrix((2, 2)),
        storage_at=lambda _state: np.zeros(2),
        partial_time_at=lambda _state, _rate: np.zeros(2),
        storage_rows_at=lambda _state, _rate: storage,
        rate_constraint_jacobian_at=lambda _state, _rate: tangent,
        sparse_state_indices=np.array([0]) if declared else None,
    )
    np.testing.assert_array_equal(pair.state, np.zeros(2))
    assert pair.rate_is_unique.all()


@pytest.mark.parametrize('source', [0.1, -0.2])
def test_exact_boundary_dependency_requires_source_selected_pair_even_when_lu_succeeds(monkeypatch, source):
    jacobian = sp.csr_matrix([
        [1.0, -1.0, 0.5, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 5.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 1.0, 0.0],
        [0.5, 0.0, 0.0, 0.0, 1.0],
    ])
    mass = sp.csr_matrix((np.ones(2), ([1, 2], [3, 4])), shape=(5, 5))
    storage = StorageRows(np.array([1, 2]), sp.eye(5, format='csr')[[0, 3, 4]])
    initial = np.array([0.045, 0.05, 0.01, 0.045, -0.0225])
    forcing = np.array([0.0, source, source, 0.0, 0.0])
    actual_lu = runtime.splu

    def spurious_lu(matrix):
        if matrix.shape == (3, 3):
            return actual_lu(matrix + 1e-12 * sp.eye(3, format='csc'))
        return actual_lu(matrix)

    monkeypatch.setattr(runtime, 'splu', spurious_lu)

    def refuse_unnecessary_bulk_correction(*_args, **_kwargs):
        msg = 'proved current-only state freedom must not perturb the admitted field particular'
        raise AssertionError(msg)

    monkeypatch.setattr(runtime, 'sparse_initial_pair', refuse_unnecessary_bulk_correction)
    pair = runtime.finite_storage_actual_rate_initial_pair(
        initial,
        residual_at=lambda state, rate: jacobian @ state + mass @ rate - forcing,
        jacobian_at=lambda _state, _rate: jacobian,
        mass_at=lambda _state, _rate: mass,
        storage_at=lambda state: mass @ state,
        partial_time_at=lambda _state, _rate: np.zeros(5),
        storage_rows_at=lambda _state, _rate: storage,
        stored_columns_at=lambda _state, _rate, _storage: np.array([3, 4]),
        rate_constraint_jacobian_at=lambda _state, _rate: jacobian,
        sparse_state_indices=np.array([0]),
    )
    np.testing.assert_array_equal(mass @ pair.state, mass @ initial)
    expected_current = (1.5 * source - initial[0]) / 5.5
    np.testing.assert_allclose(pair.state[2], expected_current, rtol=1e-12)
    np.testing.assert_allclose(jacobian @ pair.state + mass @ pair.rate, forcing, atol=1e-14)
    np.testing.assert_allclose(storage.algebraic_rows @ jacobian @ pair.rate, 0.0, atol=1e-14)


def test_a_saturating_state_search_backtracks_instead_of_cycling():
    """Full Newton on ``arctan`` started 1.5 from its root overshoots without bound; the search must not.

    The case_a static march parked in a two-cycle here (projected residual 0.0047 <-> 0.0085,
    the same 0.169 correction every iteration) because the state search took every full step.
    """
    storage = StorageRows(np.array([], dtype=int), sp.eye(1, format='csr'))
    state = runtime._actual_finite_storage_state(
        np.array([2.5]),
        np.zeros(1),
        lambda state, _rate: np.arctan(state - 1.0),
        lambda state, _rate: sp.csr_matrix([[1.0 / (1.0 + (state[0] - 1.0) ** 2)]]),
        lambda _state, _rate: sp.csr_matrix((1, 1)),
        lambda _state: np.zeros(1),
        lambda _state, _rate: storage,
        50,
        1e-10,
        None,
        None,
    )
    np.testing.assert_allclose(state, [1.0], atol=1e-9)
