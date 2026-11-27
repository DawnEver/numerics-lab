"""``numerics_lab.structural`` against SciPy's csgraph, and the golden the Rust port judges itself by.

The reference is ``scipy.sparse.csgraph`` itself: ``maximum_bipartite_matching`` gives the structural rank of each
frozen pattern, ``connected_components(connection='strong')`` the strong-component partition of each frozen digraph.
``crates/numerics/tests/golden_structural.rs`` reads ``goldens/structural.jsonl``; this file asserts live SciPy still
produces it and checks the Python seam against SciPy on the same graphs. Matchings are compared by SIZE (a maximum
matching is not unique), components as a PARTITION (labels are arbitrary up to renaming).

Regenerate deliberately: ``python -c "from tests.unit.test_structural_graph_port_golden import write; write()"``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, maximum_bipartite_matching
from tests.port_goldens import GOLDENS, load_golden, register, write_golden

from numerics_lab import structural

_STRUCTURAL = GOLDENS / 'structural.jsonl'


def _patterns() -> list[tuple[int, int, list[int], list[int]]]:
    rng = np.random.default_rng(20261126)
    out = []
    for n_rows, n_cols, density in ((6, 6, 0.3), (12, 9, 0.2), (9, 12, 0.15), (40, 40, 0.06), (60, 50, 0.04), (30, 30, 0.02)):
        mask = rng.random((n_rows, n_cols)) < density
        r, c = np.nonzero(mask)
        out.append((n_rows, n_cols, r.tolist(), c.tolist()))
    return out


def _digraphs() -> list[tuple[int, list[int], list[int]]]:
    rng = np.random.default_rng(20261127)
    out = []
    for n, density in ((8, 0.2), (25, 0.08), (60, 0.03), (100, 0.012)):
        mask = rng.random((n, n)) < density
        s, t = np.nonzero(mask)
        out.append((n, s.tolist(), t.tolist()))
    return out


def _rank(n_rows: int, n_cols: int, rows: list[int], cols: list[int]) -> int:
    a = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_rows, n_cols))
    return int((maximum_bipartite_matching(a, perm_type='column') >= 0).sum())


def _scc(n: int, sources: list[int], targets: list[int]) -> list[int]:
    a = csr_matrix((np.ones(len(sources)), (sources, targets)), shape=(n, n))
    return connected_components(a, directed=True, connection='strong')[1].tolist()


def _build_structural() -> dict:
    return {
        'source': 'scipy.sparse.csgraph maximum_bipartite_matching / connected_components(strong)',
        'matching': [
            {'n_rows': nr, 'n_cols': nc, 'rows': r, 'cols': c, 'rank': _rank(nr, nc, r, c)}
            for nr, nc, r, c in _patterns()
        ],
        'scc': [{'n': n, 'sources': s, 'targets': t, 'labels': _scc(n, s, t)} for n, s, t in _digraphs()],
    }


register(_STRUCTURAL, _build_structural)


def write() -> None:
    write_golden(_STRUCTURAL, _build_structural())


def _same_partition(a: list[int] | np.ndarray, b: list[int] | np.ndarray) -> bool:
    pairs = set(zip(np.asarray(a).tolist(), np.asarray(b).tolist(), strict=True))
    return len(pairs) == len({p[0] for p in pairs}) == len({p[1] for p in pairs})


def _ix(values: list[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def test_the_live_scipy_still_produces_the_golden() -> None:
    assert load_golden(_STRUCTURAL) == _build_structural()


def test_the_golden_holds_a_rank_deficient_pattern_and_a_nontrivial_component() -> None:
    golden = load_golden(_STRUCTURAL)
    assert any(m['rank'] < min(m['n_rows'], m['n_cols']) for m in golden['matching'])
    assert any(len(set(s['labels'])) < s['n'] for s in golden['scc'])


@pytest.mark.parametrize(('n_rows', 'n_cols', 'rows', 'cols'), _patterns())
def test_the_native_matching_is_maximum_and_valid(n_rows: int, n_cols: int, rows: list[int], cols: list[int]) -> None:
    col_of_row = structural.maximum_matching(n_rows, n_cols, _ix(rows), _ix(cols))
    matched = [(r, c) for r, c in enumerate(col_of_row.tolist()) if c >= 0]
    assert len(matched) == _rank(n_rows, n_cols, rows, cols)
    assert len({c for _, c in matched}) == len(matched)
    assert set(matched) <= set(zip(rows, cols, strict=True))


@pytest.mark.parametrize(('n', 'sources', 'targets'), _digraphs())
def test_the_native_components_are_scipys_partition_and_dependency_first(
    n: int, sources: list[int], targets: list[int]
) -> None:
    count, labels = structural.strongly_connected_components(n, _ix(sources), _ix(targets))
    assert count == len(set(labels.tolist()))
    assert _same_partition(labels, _scc(n, sources, targets))
    assert all(labels[t] <= labels[s] for s, t in zip(sources, targets, strict=True))


def test_the_partition_comparison_sees_a_merged_component() -> None:
    assert _same_partition([0, 1, 1, 2], [5, 3, 3, 9])
    assert not _same_partition([0, 1, 1, 2], [5, 3, 3, 3])


def test_the_decompositions_are_block_lower_triangular() -> None:
    for n_rows, n_cols, rows, cols in _patterns():
        ro, co, rb, cb = structural.dulmage_mendelsohn(n_rows, n_cols, _ix(rows), _ix(cols))
        assert sorted(ro.tolist()) == list(range(n_rows))
        assert sorted(co.tolist()) == list(range(n_cols))
        row_block = np.empty(n_rows, dtype=np.int64)
        col_block = np.empty(n_cols, dtype=np.int64)
        for k in range(len(rb) - 1):
            row_block[ro[rb[k] : rb[k + 1]]] = k
            col_block[co[cb[k] : cb[k + 1]]] = k
        assert all(col_block[c] <= row_block[r] for r, c in zip(rows, cols, strict=True))
        square = sum(min(rb[k + 1] - rb[k], cb[k + 1] - cb[k]) for k in range(1, len(rb) - 2))
        assert square + min(rb[1], cb[1]) + min(rb[-1] - rb[-2], cb[-1] - cb[-2]) == _rank(n_rows, n_cols, rows, cols)


def test_blt_refuses_a_singular_pattern_by_its_rank() -> None:
    with pytest.raises(ValueError, match='covers 1 of 2'):
        structural.block_triangular(2, _ix([0, 1]), _ix([0, 0]))
    ro, co, bounds = structural.block_triangular(3, _ix([0, 1, 1, 2, 2]), _ix([0, 0, 1, 1, 2]))
    assert bounds.tolist() == [0, 1, 2, 3]
    assert ro.tolist() == [0, 1, 2]


def test_tearing_and_pantelides_answer_through_the_seam() -> None:
    tears, order = structural.tear(3, _ix([0, 1, 2]), _ix([1, 2, 0]))
    assert len(tears) == 1
    assert sorted([*tears.tolist(), *order.tolist()]) == [0, 1, 2]
    # The planar pendulum: index 3 (see crates/numerics/tests/structural_textbook.rs).
    rows = [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4]
    cols = [5, 2, 6, 3, 7, 4, 0, 8, 4, 1, 0, 1]
    er, _, eq_der, var_der, assign = structural.pantelides(5, 9, _ix(rows), _ix(cols), _ix([5, 6, 7, 8, -1, -1, -1, -1, -1]))
    assert eq_der[:5].tolist().count(-1) == 2
    assert int(er.max()) + 1 == 9
    assert len(var_der) == len(assign)
    with pytest.raises(ValueError, match='negative'):
        structural.maximum_matching(2, 2, _ix([-1]), _ix([0]))
    with pytest.raises(ValueError, match='structurally singular'):
        structural.pantelides(2, 2, _ix([0, 1]), _ix([0, 0]), _ix([1, -1]))
