"""Actual storage/state uniqueness without a dense field-sized SVD."""

import numpy as np
import pytest
import scipy.sparse as sp

from motronics.gauss.linear.initial_sparse import sparse_initial_pair


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
