"""A retired storage LU must not overlap the next algebraic factorization."""

import weakref
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear.storage_border import StorageRows
from numerics_lab.nonlinear import finite_storage


@pytest.mark.parametrize('coupling', [0.0, 0.25])
@pytest.mark.parametrize('already_admitted', [False, True])
def test_storage_factor_retires_before_algebraic_lu_without_changing_solves(monkeypatch, coupling, already_admitted):
    assert Path(finite_storage.__file__).resolve() == (
        Path(__file__).resolve().parents[3] / 'python/numerics_lab/nonlinear/finite_storage.py'
    )
    original = finite_storage.splu
    references, factors, solves = [], [], []
    evaluations = Counter()
    mass = sp.csr_matrix([[1.0, coupling], [0.0, 0.0]])
    jacobian = sp.csr_matrix([[0.0, 0.0], [0.0, 1.0]])
    storage = StorageRows(np.array([0]), sp.csr_matrix([[0.0, 1.0]]))
    expected = np.array([1.0 - 2.0 * coupling, 2.0])

    class Tracked:
        def __init__(self, factor, role) -> None:
            self.factor, self.role = factor, role

        def solve(self, rhs, **kwargs):
            result = self.factor.solve(rhs, **kwargs)
            solves.append((self.role, rhs.copy(), result.copy()))
            return result

    def factor(matrix):
        role = 'storage' if len(factors) % 2 == 0 else 'algebraic'
        if role == 'algebraic':
            assert references[-1]() is None, 'retired storage LU still owned at algebraic factor entry'
        factors.append(role)
        proxy = Tracked(original(matrix), role)
        if role == 'storage':
            references.append(weakref.ref(proxy))  # No strong spy owner and no forced collection.
        return proxy

    def mass_at(_state, _rate):
        evaluations['mass'] += 1
        return mass

    def jacobian_at(_state, _rate):
        evaluations['jacobian'] += 1
        return jacobian

    def storage_at(_state, _rate):
        evaluations['storage'] += 1
        return storage

    def columns_at(_state, _rate, _storage):
        evaluations['columns'] += 1
        return np.array([0])

    def charge_at(state):
        evaluations['charge'] += 1
        return mass @ state

    def residual_at(state, _rate):
        evaluations['residual'] += 1
        return np.array([0.0, state[1] - 2.0])

    monkeypatch.setattr(finite_storage, 'splu', factor)
    result = finite_storage._actual_finite_storage_state(
        expected.copy() if already_admitted else np.zeros(2),
        np.array([1.0, 0.0]),
        residual_at,
        jacobian_at,
        mass_at,
        charge_at,
        storage_at,
        max_newton=3,
        tolerance=1e-10,
        stored_columns_at=columns_at,
        sparse_state_indices=None,
    )
    np.testing.assert_array_equal(result, expected)
    iterations = 1 if already_admitted else 2
    assert factors == ['storage', 'algebraic'] * iterations
    assert Counter(role for role, _rhs, _result in solves) == Counter({
        'storage': iterations * (1 + bool(coupling)),
        'algebraic': 0 if already_admitted else 1,
    })
    assert evaluations == {
        'mass': iterations,
        'jacobian': iterations,
        'storage': iterations,
        'columns': iterations,
        'charge': iterations if already_admitted else 4,
        'residual': iterations if already_admitted else 4,
    }
    for role, rhs, solution in solves:
        if role == 'storage':
            np.testing.assert_array_equal(solution, rhs)  # Stored minor is exactly [1].
        else:
            np.testing.assert_array_equal(rhs, [2.0])
            np.testing.assert_array_equal(solution, [2.0])
    assert all(reference() is None for reference in references)


def test_currently_admitted_state_still_requires_nonsingular_algebraic_minor():
    # Do not move LU below the early-return admission test to save memory.
    with pytest.raises(finite_storage.StorageRateAuthorityRequired, match='actual differentiated rate authority'):
        finite_storage._actual_finite_storage_state(
            np.zeros(2),
            np.zeros(2),
            residual_at=lambda _state, _rate: np.zeros(2),
            jacobian_at=lambda _state, _rate: sp.csr_matrix((2, 2)),
            mass_at=lambda _state, _rate: sp.diags([1.0, 0.0], format='csr'),
            charge_at=lambda state: np.array([state[0], 0.0]),
            storage_rows_at=lambda _state, _rate: StorageRows(np.array([0]), sp.csr_matrix([[0.0, 1.0]])),
            max_newton=1,
            tolerance=1e-10,
            stored_columns_at=lambda *_args: np.array([0]),
            sparse_state_indices=None,
        )
