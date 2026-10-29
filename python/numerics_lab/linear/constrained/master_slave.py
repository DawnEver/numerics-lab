"""Resolve a master-slave constraint list so no master is itself constrained.

Physics-free constraint algebra, shared by the FEA fold
(``hamilton/discretizations/fem/_solve.py``) and the MEC reduction
(``hamilton/formulations/magnetic_circuit/_reduction.py``). It lived in the FEA module and MEC had a
SECOND implementation of the same idea -- "chains resolve in list order" -- carrying the same
order-dependence. One scheme, one copy.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

__all__ = ['resolve_constraint_chains']


def _closure(declared):
    """Every node entangled in a chain or cycle, plus everything such a node depends on."""
    entangled = {h for h, ms in declared.items() if any(m in declared for m, _w in ms)}
    frontier = list(entangled)
    while frontier:
        for m, _w in declared[frontier.pop()]:
            if m in declared and m not in entangled:
                entangled.add(m)
                frontier.append(m)
    return entangled


def _blocks_in_dependency_order(order, pos, declared):
    """Condense the closure into strongly connected components, dependencies FIRST.

    Each block is a list of node ids. A block of size one with no self-reference is a plain
    substitution; anything larger is the genuine cycle case and gets its own small dense solve.
    """
    rows: list[int] = []
    cols: list[int] = []
    for h in order:
        for m, _w in declared[h]:
            if m in pos:
                rows.append(pos[h])
                cols.append(pos[m])
    n = len(order)
    graph = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    _n_comp, label = connected_components(graph, directed=True, connection='strong')

    members: dict[int, list[int]] = {}
    for idx, node in enumerate(order):
        members.setdefault(int(label[idx]), []).append(node)
    # Kahn over the condensation, on the edge "the component of a master must resolve first".
    comp_deps: dict[int, set[int]] = {}
    for i, j in zip(rows, cols, strict=True):
        if label[i] != label[j]:
            comp_deps.setdefault(int(label[i]), set()).add(int(label[j]))
    dependents: dict[int, list[int]] = {}
    for comp, deps in comp_deps.items():
        for dep in deps:
            dependents.setdefault(dep, []).append(comp)
    remaining = {comp: len(deps) for comp, deps in comp_deps.items()}
    ready = [comp for comp in members if remaining.get(comp, 0) == 0]
    ordered: list[list[int]] = []
    while ready:
        comp = ready.pop()
        ordered.append(members[comp])
        for dependent in dependents.get(comp, ()):
            remaining[dependent] -= 1
            if remaining[dependent] == 0:
                ready.append(dependent)
    return ordered


def _solve_block(block, declared, resolved):
    """Resolve one strongly connected block against already-resolved blocks and free targets."""
    inner = {h: i for i, h in enumerate(block)}
    rhs: list[dict[int, float]] = []
    for h in block:
        row: dict[int, float] = {}
        for m, w in declared[h]:
            if m in inner:
                continue
            expansion = resolved.get(m)
            if expansion is None:
                if m in declared:  # constrained but outside the solved closure cannot happen
                    msg = f'constraint {h} references constrained node {m} outside the solved closure'
                    raise ValueError(msg)
                row[m] = row.get(m, 0.0) + w
            else:
                for target, weight in expansion.items():
                    row[target] = row.get(target, 0.0) + w * weight
        rhs.append(row)
    if len(block) == 1 and not any(m in inner for m, _w in declared[block[0]]):
        # The overwhelming majority -- a tree edge. One expansion, and NO linear solve at all.
        return {block[0]: {t: w for t, w in rhs[0].items() if w != 0.0}}

    size = len(block)
    a_mat = np.eye(size)
    for h in block:
        for m, w in declared[h]:
            if m in inner:
                a_mat[inner[h], inner[m]] -= w
    targets = sorted({t for row in rhs for t in row})
    tpos = {t: j for j, t in enumerate(targets)}
    b_mat = np.zeros((size, len(targets)))
    for i, row in enumerate(rhs):
        for target, weight in row.items():
            b_mat[i, tpos[target]] = weight
    try:
        x_mat = np.linalg.solve(a_mat, b_mat)
    except np.linalg.LinAlgError as exc:
        loop = ' -> '.join(str(h) for h in sorted(block))
        msg = (
            f'hanging constraints form a SINGULAR system over nodes ({loop}): the relations are '
            f'mutually dependent with no solution in terms of free DOFs, so no fold can satisfy '
            f'them. This is not a chain (chains resolve); it is a contradictory constraint set '
            f'and the mesh that produced it is wrong.'
        )
        raise ValueError(msg) from exc
    return {h: {targets[j]: float(x_mat[inner[h], j]) for j in np.flatnonzero(x_mat[inner[h]])} for h in block}


def resolve_constraint_chains(constraints):
    """Rewrite every constraint in terms of UNCONSTRAINED targets only, chains and cycles alike.

    ``constraints`` is ``[(slave, [(master, weight), ...]), ...]``. A master that is itself a
    slave is substituted out; the result names free targets only, so whatever consumes it can
    fold in any order.

    WHY THIS EXISTS. Both solvers eliminated one constraint at a time in DECLARATION ORDER --
    the FEA fold replacing the slave's row with an identity, MEC reading the current ``p_lil``
    rows and calling it "chains resolve in list order". Neither resolves: they resolve if the
    list happens to be topologically sorted, and nothing produces it that way. Measured on a 5x5
    unit-quad grid carrying ``A = 0.3x - 0.7y`` (exactly representable, every planted constraint
    exactly satisfied, so a correct fold cannot move the solution): independent constraints left
    ``max|A - exact|`` at 4e-16, while the chain ``12 <- {11, 13}`` with ``11 <- {6, 16}`` gave
    5.500e-01 declared one way and 2.983e-07 the other. Two different wrong answers for one
    problem is what identifies the elimination rather than the physics.

    A CYCLE IS NOT AUTOMATICALLY CONTRADICTORY, and refusing one was this function's own earlier
    mistake. For a mutual pair ``P = w_a A + w_b B`` and ``A = u_p P + u_c C``, substitution
    gives ``P (1 - w_a u_p) = w_a u_c C + w_b B`` -- a unique solution whenever
    ``w_a u_p != 1``. So the general form is a small linear solve, not a topological sort:
    with ``A_c = W_cc A_c + W_cf A_f``, the resolved rows are
    ``A_c = (I - W_cc)^-1 W_cf A_f``. Only a SINGULAR ``I - W_cc`` is genuinely unsolvable, and
    that alone raises.

    THE CLOSURE IS A SHALLOW FOREST OF TINY CYCLES, WHICH IS WHY THE INVERSE IS NOT DENSE. This
    function used to build ``np.eye(n)`` over the WHOLE closure and call ``numpy.linalg.solve``,
    a decision priced on case_a under ``--set mesh.max_aspect=4`` -- "of 3892 hanging nodes, 4 are
    chained, forming two mutual pairs (2590 <-> 8105, 7744 <-> 7803)", a 4x4 solve at any price.
    That claim was about ONE MESH OF ONE CASE, and it did not even hold for the rest of its own
    case. Measured 2026-09-06 on the shipped legs:

    * ``case_p --select fea,quad``: 13,575 of 32,343 constraints entangled (42.0%),
      ``I - W_cc`` carrying 25,496 nonzeros -- 1.878 per row, density 1.384e-4 -- condensing into
      13,526 strongly connected components of which 41 are non-trivial, the largest holding 4
      nodes. 90 of 13,575 nodes lie on a cycle: 0.66%.
    * ``case_a --select fea,quad`` (the DEFAULT mesh, not ``--set mesh.max_aspect=4``): 5,238 of
      13,711 entangled (38.2%), and **zero** non-trivial components -- a pure forest.

    So a large closure is the NORM and a cycle is the exception, which is the reverse of what the
    old comment implied. The dense form allocated 2.07 GiB per Newton iteration and spent 399.0 s
    of case_p's 500 s budget inside ``numpy.linalg.solve``, against 30.1 s in the actual physics.

    So the closure is condensed into strongly connected components and substituted in dependency
    order. A singleton block with no self-reference -- 13,485 of case_p's 13,575, and all 5,238 of
    case_a's -- is one expansion of already-resolved rows and NO linear solve; the dense
    ``(I - W_cc)^-1`` survives only INSIDE a genuine cycle, where it is the right instrument and
    the block is small. There is no size threshold and no second entry point: the block structure
    IS the criterion, and it degenerates to the old dense call exactly when the closure really is
    one entangled lump.

    Both forms are the same algebra. Measured on case_p's real 13,575-node closure: 1.33 s and
    22 MiB against 31.9 s and 4.84 GiB, agreeing to **3.0e-16** over the union of targets --
    DIMENSIONLESS, since a constraint weight is a ratio of shape functions, so no units split can
    move that bar. The two do not agree on SUPPORT, and the sparse side is the clean one: a global
    dense LU fills, leaving roundoff (largest 3.2e-17, down to 1e-86) on 255 of the 13,575 rows
    where the algebra has an exact zero. The sparse form invents no target at all, which is what
    ``tests/unit/test_the_constraint_closure_is_sparse_not_a_dense_inverse.py`` asserts
    rather than identical support -- pinning that would pin a LAPACK artefact.

    Only the nodes actually entangled are solved; every other row is returned untouched, and a
    constraint set with no chains at all is returned AS THE SAME OBJECT, so meshes that never had
    the defect stay bit-for-bit.
    """
    declared = {int(h): [(int(m), float(w)) for m, w in masters] for h, masters in constraints}
    if not any(m in declared for masters in declared.values() for m, _w in masters):
        return constraints

    order = sorted(_closure(declared))
    pos = {h: i for i, h in enumerate(order)}
    resolved: dict[int, dict[int, float]] = {}
    for block in _blocks_in_dependency_order(order, pos, declared):
        resolved.update(_solve_block(block, declared, resolved))
    # SORTED BY TARGET ID, because that is the order the dense form emitted (it read a row of
    # `x_mat` against a globally sorted `free_targets`) and `_mec_port_cases`'s golden pins it.
    # Both folds are order-insensitive, so this is a contract with the GOLDEN, not with the algebra.
    return [(h, sorted(resolved[h].items())) if h in resolved else (h, masters) for h, masters in declared.items()]
