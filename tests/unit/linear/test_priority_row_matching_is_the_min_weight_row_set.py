"""The priority row matching selects exactly the rows of the row-order min-weight full matching."""

import time

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from numerics_lab.linear.initial_rate import _priority_row_matching


def _min_weight_rows(pattern):
    entries = pattern.tocoo()
    weights = sp.csr_matrix(
        (1.0 + entries.row / (pattern.shape[0] + 1), (entries.row, entries.col)), shape=pattern.shape
    )
    try:
        _columns, rows = min_weight_full_bipartite_matching(weights.T)
    except ValueError:
        return None
    return None if np.any(rows < 0) else rows


@pytest.mark.parametrize('seed', range(40))
def test_the_selected_rows_equal_the_weighted_matching_rows(seed):
    rng = np.random.default_rng(seed)
    for _ in range(50):
        columns = int(rng.integers(1, 30))
        rows = columns + int(rng.integers(0, 30))
        pattern = sp.random(rows, columns, density=rng.uniform(0.03, 0.5), random_state=rng, format='csr')
        pattern.data[:] = 1.0
        matched, oracle = _priority_row_matching(pattern), _min_weight_rows(pattern)
        assert (matched is None) == (oracle is None)
        if matched is not None:
            assert sorted(matched.tolist()) == sorted(oracle.tolist())
            assert np.all(pattern[matched, np.arange(columns)].A1 != 0)


def test_a_row_short_pattern_has_no_column_perfect_matching():
    assert _priority_row_matching(sp.csr_matrix(np.ones((2, 3)))) is None


def test_a_field_size_pattern_matches_in_well_under_a_second():
    pattern = sp.random(3641, 1943, density=237792 / (3641 * 1943), random_state=1, format='csr')
    started = time.perf_counter()
    matched = _priority_row_matching(pattern)
    assert time.perf_counter() - started < 1.0
    assert sorted(matched.tolist()) == sorted(_min_weight_rows(pattern).tolist())
