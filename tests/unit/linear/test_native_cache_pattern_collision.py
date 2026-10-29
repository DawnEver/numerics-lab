"""A cache hash match is not proof of the full sparse pattern."""

import zlib

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear import native_direct


@pytest.mark.skipif(not native_direct.native_available(), reason='native direct backend unavailable')
def test_a_planted_cache_hash_collision_cannot_reuse_a_different_full_pattern(monkeypatch):
    first = np.diag([4.0, 5.0, 6.0, 7.0])
    first[0, 1] = first[1, 0] = 0.3
    second = np.diag([4.0, 5.0, 6.0, 7.0])
    second[2, 3] = second[3, 2] = 0.3
    held = native_direct.native_factor(sp.csr_matrix(first))
    matrix = sp.csr_matrix(second)
    key = (
        matrix.shape[0],
        matrix.nnz,
        zlib.crc32(np.asarray(matrix.indptr, dtype=np.int64).tobytes()),
        zlib.crc32(np.asarray(matrix.indices, dtype=np.int64).tobytes()),
    )
    # Deterministically simulate the same lookup outcome as a CRC collision;
    # no C-extension monkeypatching or probabilistic hash search is involved.
    monkeypatch.setitem(native_direct._cache, key, held._entry)
    result = native_direct.native_factor(matrix)
    exact_key = (
        matrix.shape[0],
        np.asarray(matrix.indptr, dtype=np.int64).tobytes(),
        np.asarray(matrix.indices, dtype=np.int64).tobytes(),
    )
    assert native_direct._cache[exact_key] is result._entry
    rhs = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(result.solve(rhs), np.linalg.solve(second, rhs), rtol=1e-12, atol=1e-13)
