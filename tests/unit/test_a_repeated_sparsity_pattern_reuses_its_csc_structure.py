"""A march's jacobian keeps one sparsity pattern step after step; its CSC structure is built once.

`_default_solve` hands SuperLU a CSC matrix. Converting each step's CSR jacobian rebuilds the same
column structure every solve; the cache keeps that structure (and the CSR->CSC data permutation)
for as long as the pattern is unchanged, and the cached solve is BIT-IDENTICAL to the uncached one.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from numerics_lab.linear import solve_linear_system
from numerics_lab.time_integrators import _forms
from numerics_lab.time_integrators._forms import _default_solve


def _pattern(n: int, seed: int) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    a = sp.random(n, n, density=0.08, random_state=rng, format='csr') + sp.eye(n, format='csr') * 4.0
    a.sort_indices()
    return sp.csr_matrix(a)


def _same_pattern_new_values(a: sp.csr_matrix, seed: int) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    data = a.data * rng.uniform(0.5, 1.5, a.data.size)
    return sp.csr_matrix((data, a.indices.copy(), a.indptr.copy()), shape=a.shape)


def _uncached(matrix, rhs):
    return np.asarray(solve_linear_system(sp.csc_matrix(matrix), np.asarray(rhs, dtype=float)))


def test_the_cached_solve_is_bit_identical_to_the_uncached_one_over_a_march() -> None:
    _forms._CSC_STRUCTURE.clear()
    base = _pattern(200, 0)
    rng = np.random.default_rng(1)
    for step in range(25):
        matrix = _same_pattern_new_values(base, 100 + step)
        rhs = rng.standard_normal(base.shape[0])
        cached = _default_solve(matrix, rhs)
        assert cached.tobytes() == _uncached(matrix, rhs).tobytes(), step


def test_the_structure_is_built_once_for_one_pattern() -> None:
    _forms._CSC_STRUCTURE.clear()
    base = _pattern(120, 2)
    for step in range(5):
        _default_solve(_same_pattern_new_values(base, step), np.ones(base.shape[0]))
    assert _forms._CSC_STRUCTURE.builds == 1


def test_a_changed_pattern_rebuilds_and_stays_bit_identical() -> None:
    """Planted control: a NEW pattern must not be served the stale structure."""
    _forms._CSC_STRUCTURE.clear()
    first, second = _pattern(150, 3), _pattern(150, 4)
    assert not np.array_equal(first.indices, second.indices)
    rhs = np.arange(150, dtype=float)
    _default_solve(first, rhs)
    cached = _default_solve(second, rhs)
    assert _forms._CSC_STRUCTURE.builds == 2
    assert cached.tobytes() == _uncached(second, rhs).tobytes()


def test_the_planted_stale_structure_is_caught_by_the_bit_identity_check() -> None:
    """Planted control on the TEST: serve a deliberately wrong permutation and the comparison reds."""
    _forms._CSC_STRUCTURE.clear()
    base = _pattern(100, 5)
    rhs = np.ones(100)
    _default_solve(base, rhs)
    _forms._CSC_STRUCTURE.permutation = _forms._CSC_STRUCTURE.permutation[::-1].copy()
    matrix = _same_pattern_new_values(base, 6)
    planted = _default_solve(matrix, rhs)
    assert planted.tobytes() != _uncached(matrix, rhs).tobytes()
    _forms._CSC_STRUCTURE.clear()
