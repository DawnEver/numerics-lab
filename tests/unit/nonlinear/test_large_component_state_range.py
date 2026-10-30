"""Many index-two state directions need sparse issuance AND original joint correction."""

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear.index_two import certified_null_directions
from numerics_lab.linear.storage_border import StorageRows
from numerics_lab.nonlinear.finite_storage import finite_storage_actual_rate_initial_pair


@pytest.fixture
def bounded_dense(monkeypatch):
    for cls in (sp.csr_matrix, sp.csc_matrix):
        original = cls.toarray

        def guarded(matrix, *args, _original=original, **kwargs):
            assert min(matrix.shape) <= 512, 'global wide dense conversion'
            return _original(matrix, *args, **kwargs)

        monkeypatch.setattr(cls, 'toarray', guarded)
    original_eye, original_zeros = np.eye, np.zeros

    def eye(n, m=None, *args, **kwargs):
        assert min(n, n if m is None else m) <= 512, 'global dense null identity'
        return original_eye(n, m, *args, **kwargs)

    def zeros(shape, *args, **kwargs):
        if isinstance(shape, tuple) and len(shape) == 2:
            assert min(shape) <= 512, 'global dense state/range allocation'
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(np, 'eye', eye)
    monkeypatch.setattr(np, 'zeros', zeros)


@pytest.mark.parametrize('count', [3, 513])
@pytest.mark.parametrize('mixed', ['none', 'pair', 'chain'])
@pytest.mark.usefixtures('bounded_dense')
def test_complete_index_two_pair_with_actual_cross_component_mixed_columns(count, mixed):
    # At t=0: F=[z+xdot, x-s(t)+t*K*z], Q=[x,0]. The actual mixed G columns are K.
    identity = sp.eye(count, format='csr')
    mass = sp.block_diag((identity, sp.csr_matrix((count, count))), format='csr')
    jacobian = sp.bmat([[None, identity], [identity, None]], format='csr')
    stored = np.arange(count)
    algebraic = sp.hstack((sp.csr_matrix((count, count)), identity), format='csr')
    storage = StorageRows(stored, algebraic)  # Exact analytic row-space owner of this declared M.
    if mixed == 'chain':
        k = sp.diags([np.full(count - 1, 0.25)], [1], shape=(count, count), format='csr')
    elif mixed == 'pair':
        k = sp.csr_matrix(([0.25, 0.125], ([0, 1], [1, 0])), shape=(count, count))
    else:
        k = sp.csr_matrix((count, count))
    source = np.full(count, 0.25)
    source_rate = np.ones(count)
    initial = np.concatenate((source, np.zeros(count)))

    def partial(state, _rate):
        return np.concatenate((np.zeros(count), -source_rate + k @ state[count:]))

    def state_columns(_state, _rate, directions):
        assert sp.issparse(directions)
        return sp.vstack((sp.csr_matrix((count, directions.shape[1])), k @ directions[count:]), format='csr')

    def solve():
        return finite_storage_actual_rate_initial_pair(
            initial,
            residual_at=lambda state, rate: jacobian @ state + mass @ rate - np.r_[np.zeros(count), source],
            jacobian_at=lambda _state, _rate: jacobian,
            mass_at=lambda _state, _rate: mass,
            storage_at=lambda state: mass @ state,
            partial_time_at=partial,
            storage_rows_at=lambda _state, _rate: storage,
            stored_columns_at=lambda *_args: stored,
            rate_constraint_jacobian_at=lambda _state, _rate: jacobian,
            rate_constraint_state_columns_at=state_columns,
            sparse_state_indices=stored,
        )

    if count > 512 and mixed == 'chain':
        # Discovery components are singletons, but the TRUE mixed joint system is irreducible.
        with pytest.raises(NotImplementedError, match='Schur component'):
            solve()
        return
    pair = solve()
    expected_rate = source_rate.copy()
    if mixed != 'none':
        width = count if mixed == 'chain' else 2
        expected_rate[:width] = np.linalg.solve(np.eye(width) - k[:width, :width].toarray(), source_rate[:width])
    np.testing.assert_allclose(pair.state[count:], -expected_rate, rtol=1e-14, atol=0)
    np.testing.assert_allclose(pair.rate[:count], expected_rate, rtol=1e-14, atol=0)
    np.testing.assert_array_equal(mass @ pair.state, mass @ initial)
    np.testing.assert_allclose(jacobian @ pair.state + mass @ pair.rate - np.r_[np.zeros(count), source], 0, atol=1e-14)
    np.testing.assert_allclose(algebraic @ (jacobian @ pair.rate + partial(pair.state, pair.rate)), 0, atol=1e-14)
    assert pair.rate_is_unique[:count].all() and not pair.rate_is_unique[count:].any()


@pytest.mark.usefixtures('bounded_dense')
def test_sparse_global_null_basis_and_small_component_are_not_pruned():
    count = 1025
    basis = certified_null_directions(sp.eye(count, format='csr'), sp.csr_matrix((count, count)), sparse_output=True)
    assert sp.isspmatrix_csr(basis) and basis.shape == (count, count) and basis.nnz == count
    right = sp.csr_matrix([[1.0, 1e16, 0.0], [0.0, 1e-16, 1.0]])
    tiny = certified_null_directions(sp.eye(2, format='csr'), right, sparse_output=True)
    assert tiny.nnz == 3
    np.testing.assert_allclose(tiny.toarray()[:, 0] / tiny[0, 0], [1.0, -1e-16, 1e-32], rtol=1e-15, atol=0)


def test_connectivity_does_not_overflow_256_shared_rows():
    right = sp.csr_matrix(np.tile([1.0, -1.0], (256, 1)))
    basis = certified_null_directions(sp.eye(256, format='csr'), right)
    assert basis.shape == (2, 1)
    np.testing.assert_allclose(right @ basis, 0, atol=0)


@pytest.mark.usefixtures('bounded_dense')
def test_large_irreducible_rank_work_still_refuses():
    right = sp.csr_matrix(np.ones((1, 513)))
    with pytest.raises(NotImplementedError, match='component|bounded'):
        certified_null_directions(sp.eye(1, format='csr'), right, sparse_output=True)
