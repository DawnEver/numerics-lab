"""The entangled closure is a shallow FOREST of tiny cycles, so it is not a dense inverse.

``resolve_constraint_chains`` priced its ``np.eye(n)`` on case_a under ``--set mesh.max_aspect=4``,
where "of 3892 hanging nodes, 4 are chained" -- a 4x4 dense solve, free at any price. That
declaration is a statement about ONE mesh, and case_p's ``fea,quad`` leg refutes it:
measured 2026-09-06 on the real leg, 13,575 of its 32,343 constraints are entangled (42%), the
closure matrix ``I - W_cc`` carries 25,496 nonzeros -- **1.878 per row, density 1.384e-4** -- and
its condensation is 13,526 strongly connected components of which 41 are non-trivial, the largest
holding 4 nodes. 90 nodes of 13,575 are in a cycle; every other one is a tree edge.

The dense form allocated ``n x n + n x m`` = 2.07 GiB per Newton iteration and cost the leg its
500 s budget (``numpy.linalg.solve``: 399.0 s tottime against 30.1 s in the actual physics).

So the two properties asserted here are:

* **agreement** -- the resolved weights match the dense ``(I - W_cc)^-1 W_cf`` oracle. The
  tolerance is DIMENSIONLESS: a constraint weight is a ratio of shape functions, so ``1e-12``
  here carries no unit and survives any units split (``.claude/rules/rem/porting.md``).
* **cost** -- peak allocation grows with the NONZEROS, not with ``n^2``. Pinned as the RATIO
  between two operating points, never a wall-clock number from one box.

Both carry a FLOOR: an agreement test between two resolvers that each did nothing is vacuous, so
each asserts the planted closure size, the presence of a multi-hop expansion, and that a node
inside a non-trivial cycle came out resolved.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from numerics_lab.linear.constrained.master_slave import resolve_constraint_chains

_FREE_BASE = 1_000_000  # free-target ids live above every constrained id


def _planted_closure(n_nodes, *, n_cycles=41, seed=20260906):
    """A closure with case_p's measured SHAPE: ~1.9 masters/row, depth <= 4, tiny cycles.

    Returns ``(constraints, cycle_nodes, closure)`` where ``closure`` is computed HERE, from the
    plant, so the resolver's own closure can be reconciled against a total it did not produce.
    """
    rng = np.random.default_rng(seed)
    depth = np.zeros(n_nodes, dtype=int)
    declared: dict[int, list[tuple[int, float]]] = {}
    for i in range(n_nodes):
        if i < 8 or i % 4 == 0:
            depth[i] = 0
            masters = [(_FREE_BASE + 2 * i, 0.5), (_FREE_BASE + 2 * i + 1, 0.5)]
        else:
            cand = int(rng.integers(max(0, i - 64), i))
            while depth[cand] >= 4:
                cand = int(rng.integers(max(0, i - 64), i))
            depth[i] = depth[cand] + 1
            masters = [(cand, 0.5), (_FREE_BASE + 2 * i, 0.5)]
        declared[i] = masters
    # Mutual pairs: A <-> B, each also mastering a free target, so ``I - W_cc`` stays regular.
    cycle_nodes: list[int] = []
    step = max(2, n_nodes // (n_cycles + 1))
    for k in range(n_cycles):
        a = 8 + k * step
        b = a + 1
        if b >= n_nodes:
            break
        declared[a] = [(b, 0.5), (_FREE_BASE + 2 * a, 0.5)]
        declared[b] = [(a, 0.5), (_FREE_BASE + 2 * b, 0.5)]
        cycle_nodes.extend((a, b))
    closure = {h for h, ms in declared.items() if any(m in declared for m, _w in ms)}
    frontier = list(closure)
    while frontier:
        for m, _w in declared[frontier.pop()]:
            if m in declared and m not in closure:
                closure.add(m)
                frontier.append(m)
    return [(h, declared[h]) for h in range(n_nodes)], cycle_nodes, closure


def _dense_oracle(constraints):
    """The pre-2026-09-06 algorithm, verbatim, as the reference answer (never as the shipped path)."""
    declared = {int(h): [(int(m), float(w)) for m, w in masters] for h, masters in constraints}
    entangled = {h for h, ms in declared.items() if any(m in declared for m, _w in ms)}
    frontier = list(entangled)
    while frontier:
        h = frontier.pop()
        for m, _w in declared[h]:
            if m in declared and m not in entangled:
                entangled.add(m)
                frontier.append(m)
    order = sorted(entangled)
    pos = {h: i for i, h in enumerate(order)}
    free_targets = sorted({m for h in order for m, _w in declared[h] if m not in declared})
    fpos = {m: j for j, m in enumerate(free_targets)}
    a_mat = np.eye(len(order))
    b_mat = np.zeros((len(order), len(free_targets)))
    for h in order:
        for m, w in declared[h]:
            if m in pos:
                a_mat[pos[h], pos[m]] -= w
            else:
                b_mat[pos[h], fpos[m]] += w
    x_mat = np.linalg.solve(a_mat, b_mat)
    return {h: {free_targets[j]: float(x_mat[pos[h], j]) for j in np.flatnonzero(x_mat[pos[h]])} for h in order}, order


def test_the_sparse_resolver_agrees_with_the_dense_inverse_on_a_closure_of_real_scale():
    """Same weights as ``(I - W_cc)^-1 W_cf``, to 1e-12 DIMENSIONLESS -- a weight carries no unit."""
    constraints, cycle_nodes, closure = _planted_closure(1500)
    expected, order = _dense_oracle(constraints)

    got = {int(h): dict(ms) for h, ms in resolve_constraint_chains(constraints)}

    # FLOOR: the scan reached the whole planted closure, and it is not a set of one-hop rows.
    assert set(order) == closure, 'the resolved closure must reconcile with the one the plant grew'
    assert len(closure) > 1200, f'only {len(closure)} of 1500 planted nodes entangled -- too weak a fixture'
    assert max(len(v) for v in expected.values()) >= 3, 'a closure whose rows are all one-hop proves nothing'
    assert cycle_nodes and all(n in got for n in cycle_nodes), 'nodes inside a mutual pair must come out resolved'
    assert set(got) >= set(expected), 'every entangled node must appear in the result'

    # Compared over the UNION of targets, because the dense oracle FILLS: a global LU leaves
    # roundoff where the algebra has an exact zero (measured on the real case_p closure: 255 of
    # 13,575 rows carry dense-only entries, largest 3.2e-17, smallest 1e-86, and the sparse side
    # invents none). Requiring identical support would pin a LAPACK artefact, so what is asserted
    # is the direction that matters -- the sparse form never invents a target, and no shared
    # target moves.
    for h, exp_row in expected.items():
        assert not (set(got[h]) - set(exp_row)), f'node {h} invented a free target the algebra has no path to'
        for target in set(exp_row) | set(got[h]):
            assert got[h].get(target, 0.0) == pytest.approx(exp_row.get(target, 0.0), abs=1e-12), (
                f'node {h} target {target}'
            )


def test_peak_allocation_follows_the_nonzeros_not_n_squared():
    """The ratio between two operating points, not a number from one box.

    Doubling ``n`` doubles the nonzeros, so a sparse resolver's peak grows ~2x; the dense
    ``np.eye(n)`` alone grows 4x, and its cost PER NODE is ``8n`` bytes -- unbounded in ``n``,
    96 KiB per node already at ``n = 12000`` (1.15 GiB for the identity alone). So the bar is a
    per-node CEILING of 2 KiB -- generous for the Python dicts and tuples the closure is made of,
    and unreachable for anything holding an ``n x n`` array -- plus the ratio, 3x for a doubling.
    Measured 2026-09-06: 11.5 MiB at ``n = 12000``, against 4859 MiB for the dense form.
    """
    peaks = {}
    for n_nodes in (6000, 12000):
        constraints, _cycles, _closure = _planted_closure(n_nodes)
        tracemalloc.start()
        try:
            out = resolve_constraint_chains(constraints)
            peaks[n_nodes] = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        # FLOOR: a resolver that returned the input untouched would allocate nothing and pass.
        resolved = {int(h): dict(ms) for h, ms in out}
        assert len(resolved) == n_nodes
        assert sum(len(v) for v in resolved.values()) > 2 * n_nodes, 'the rows were never expanded'
        assert max(len(v) for v in resolved.values()) >= 3, 'no multi-hop row was actually substituted'

    assert peaks[12000] < 2048 * 12000, (
        f'peak {peaks[12000] / 2**20:.1f} MiB at n = 12000 is more than 2 KiB per node -- '
        f'that is an n x n allocation, not a sparse one (the dense form needs 8n = 96 KiB/node here)'
    )
    assert peaks[12000] / peaks[6000] < 3.0, (
        f'peak grew {peaks[12000] / peaks[6000]:.2f}x when n doubled: superlinear in n, so it is not the nonzeros'
    )
