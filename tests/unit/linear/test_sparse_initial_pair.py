"""Actual storage/state uniqueness without a dense field-sized SVD."""

import numpy as np
import pytest
import scipy.sparse as sp

from motronics.gauss.linear.initial_sparse import sparse_complete_initial_rate, sparse_initial_pair
from motronics.gauss.linear.storage_border import storage_border_rows


def test_an_ideal_source_current_rate_is_not_a_physical_zero_derivative():
    # C v_dot + g v + i_source = 0; v = declared ramp(t).
    # Source current is determined, but its outgoing rate needs ramp''.
    pair = sparse_initial_pair(
        [[3.0, 1.0], [1.0, 0.0]],
        [6.0, 0.0],
        [[4.0, 0.0], [0.0, 0.0]],
        [2.0, 0.0],
        algebraic_rows=[[0.0, 1.0]],
        stored_rows=[0],
        partial_time=[0.0, -5.0],
    )
    np.testing.assert_allclose(pair.state, [2.0, -26.0])
    assert pair.rate[0] == pytest.approx(5.0)
    np.testing.assert_array_equal(pair.rate_is_unique, [True, False])


def test_declared_curvature_proves_source_current_rate_without_an_acceleration_guess():
    jacobian, mass, algebraic = [[3.0, 1.0], [1.0, 0.0]], [[4.0, 0.0], [0.0, 0.0]], [[0.0, 1.0]]
    pair = sparse_initial_pair(
        jacobian,
        [6.0, 0.0],
        mass,
        [2.0, 0.0],
        algebraic_rows=algebraic,
        stored_rows=[0],
        partial_time=[0.0, -5.0],
    )
    proven = sparse_complete_initial_rate(
        jacobian,
        [-20.0, 0.0],
        mass,
        pair,
        algebraic_rows=algebraic,
        partial_time=[0.0, -5.0],
        acceleration_remainder=[0.0, -2.0],
    )
    np.testing.assert_allclose(proven.state, pair.state)
    np.testing.assert_allclose(proven.rate, [5.0, -23.0])
    assert proven.rate_is_unique.all()


def test_sparse_initial_pair_never_densifies_a_large_differential_field(monkeypatch):
    def refuse_dense(*_args, **_kwargs):
        pytest.fail('field-sized dense initial decomposition is forbidden')

    monkeypatch.setattr(sp.csr_matrix, 'toarray', refuse_dense)
    n = 17000
    pair = sparse_initial_pair(
        2 * sp.eye(n, format='csr'),
        np.full(n, 2.0),
        sp.eye(n, format='csr'),
        np.ones(n),
        algebraic_rows=sp.csr_matrix((0, n)),
        stored_rows=np.arange(n),
        partial_time=np.zeros(n),
    )
    np.testing.assert_allclose(pair.state, 1.0)
    np.testing.assert_allclose(pair.rate, -2.0)
    assert pair.rate_is_unique.all()
    proven = sparse_complete_initial_rate(
        2 * sp.eye(n, format='csr'),
        np.full(n, 2.0),
        sp.eye(n, format='csr'),
        pair,
        algebraic_rows=sp.csr_matrix((0, n)),
        partial_time=np.zeros(n),
        acceleration_remainder=np.zeros(n),
    )
    np.testing.assert_allclose(proven.rate, pair.rate)
    assert proven.rate_is_unique.all()


def test_sparse_projection_does_not_guess_an_undetermined_current_split():
    with pytest.raises(ValueError, match='physical authority|nonsingular'):
        sparse_initial_pair(
            [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [0.0, 0.0, 0.0],
            np.diag([1.0, 0.0, 0.0]),
            np.zeros(3),
            algebraic_rows=[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            stored_rows=[0],
            partial_time=np.zeros(3),
        )


def test_sparse_projection_refuses_an_actual_storage_source_contradiction():
    with pytest.raises(ValueError, match='contradicts'):
        sparse_initial_pair(
            [[3.0, 1.0], [1.0, 0.0]],
            [6.0, -1.0],
            [[4.0, 0.0], [0.0, 0.0]],
            [2.0, 0.0],
            algebraic_rows=[[0.0, 1.0]],
            stored_rows=[0],
            partial_time=[0.0, -5.0],
        )


def test_a_storage_border_eliminates_the_bulk_without_dropping_cross_rate_rows():
    mass = sp.csr_matrix([
        [2.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 3.0, 0.0, 0.0, 0.0],
        [4.0, 5.0, 2.0, 4.0, 0.0],
        [6.0, 7.0, 1.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    rows = storage_border_rows(mass, bulk_indices=[0, 1], border_rows=[2, 3])
    assert len(rows.stored_rows) == 3
    assert rows.algebraic_rows.shape == (2, 5)
    np.testing.assert_allclose((rows.algebraic_rows @ mass).toarray(), 0.0, atol=1e-14)
    assert np.linalg.matrix_rank(mass[rows.stored_rows].toarray()) == 3


def test_a_large_storage_bulk_only_densifies_the_explicit_small_border(monkeypatch):
    original = sp.csr_matrix.toarray

    def forbid_square(self, *args, **kwargs):
        assert min(self.shape) <= 2, 'field-sized dense storage rank decomposition'
        return original(self, *args, **kwargs)

    monkeypatch.setattr(sp.csr_matrix, 'toarray', forbid_square)
    n = 17000
    mass = sp.bmat(
        [[sp.eye(n, format='csr'), None], [sp.csr_matrix(np.ones((2, n))), sp.eye(2, format='csr')]],
        format='csr',
    )
    rows = storage_border_rows(mass, bulk_indices=np.arange(n), border_rows=np.array([n, n + 1]))
    assert rows.algebraic_rows.shape == (0, n + 2)
    assert len(rows.stored_rows) == n + 2


def test_storage_border_never_silently_discards_an_undeclared_storage_row():
    with pytest.raises(ValueError, match='omits actual mass'):
        storage_border_rows(sp.eye(3, format='csr'), bulk_indices=[0], border_rows=[1])
