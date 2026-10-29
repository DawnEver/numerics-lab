"""Bounded native-only factor/cache stress with independent numerical and object oracles."""

import gc
from dataclasses import dataclass

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear import native_direct

pytestmark = pytest.mark.skipif(not native_direct.native_available(), reason='native direct backend unavailable')


@dataclass(frozen=True)
class Probe:
    iteration: int
    label: str


def _matrix(pattern, iteration):
    dense = np.diag(np.array([4.0, 5.0, 6.0, 7.0]))
    row, col = divmod(pattern, 4)
    col = (col + row + 1) % 4
    dense[row, col] = 0.2
    dense[col, row] = 0.2 if iteration % 3 == 0 else -0.1
    if iteration % 3 == 2:
        dense[0, 0] = -4.0
    matrix = sp.csr_matrix(dense)
    # Supply genuinely short-lived strided values through the normal public seam.
    storage = np.empty(matrix.nnz * 2)
    storage[::2] = matrix.data
    matrix.data = storage[::2]
    return matrix, dense


def test_native_factor_reuse_eviction_and_strided_inputs_preserve_numerics_and_python_objects():
    before = sum(native_direct.DIRECT_SOLVES.values())
    for iteration in range(400):
        matrix, dense = _matrix(iteration % 5, iteration)
        rhs_storage = np.arange(8.0, dtype=float)
        rhs_storage[::2] += 1.0
        rhs = rhs_storage[::2]
        held = native_direct.native_factor(matrix)
        assert held is not None
        changed = matrix.copy()
        changed.data *= 1.1
        other = native_direct.native_factor(changed)
        assert other is not None
        np.testing.assert_allclose(other.solve(rhs), np.linalg.solve(dense * 1.1, rhs), rtol=1e-12, atol=1e-13)
        # Stale generation and later cache evictions must not overwrite the held values.
        for pattern in range(5):
            temporary, _ = _matrix(pattern, iteration + 1)
            assert native_direct.native_factor(temporary) is not None
        del matrix, changed, temporary, other, rhs_storage
        if iteration % 25 == 0:
            gc.collect()
        np.testing.assert_allclose(held.solve(rhs), np.linalg.solve(dense, rhs), rtol=1e-12, atol=1e-13)
        coo = sp.coo_matrix(([1.0, 2.0], ([0, 1], [1, 0])), shape=(2, 2))
        np.testing.assert_array_equal(coo.toarray(), [[0.0, 1.0], [2.0, 0.0]])
        probe = Probe(iteration, 'intact')
        assert (probe.iteration, probe.label) == (iteration, 'intact')
    assert sum(native_direct.DIRECT_SOLVES.values()) >= before + 800
