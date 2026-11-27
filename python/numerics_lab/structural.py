"""Structural graph algorithms on sparse bipartite and directed graphs -- domain-free, native (``crates/numerics/src/structural/``).

A pattern is index triplets (``rows``, ``cols`` as ``int64``); a digraph is edge lists (``sources``, ``targets``).
"None" in a returned map is ``-1``. What a row, a column or a node means is the caller's.

* :func:`maximum_matching` -- Hopcroft-Karp; the column of each row, as ``scipy.sparse.csgraph.maximum_bipartite_matching``.
* :func:`dulmage_mendelsohn` -- over-determined, square (fine blocks) and under-determined parts, block lower triangular.
* :func:`block_triangular` -- the BLT form of a structurally nonsingular square pattern.
* :func:`strongly_connected_components` -- Tarjan; labels dependency-first.
* :func:`tear` -- tearing nodes that make a dependency digraph acyclic, and the evaluation order of the rest.
* :func:`pantelides` -- Pantelides structural index reduction of a DAE pattern.
"""

from __future__ import annotations

from numerics_lab._native import (
    block_triangular,
    dulmage_mendelsohn,
    maximum_matching,
    pantelides,
    strongly_connected_components,
    tear,
)

__all__ = [
    'block_triangular',
    'dulmage_mendelsohn',
    'maximum_matching',
    'pantelides',
    'strongly_connected_components',
    'tear',
]
