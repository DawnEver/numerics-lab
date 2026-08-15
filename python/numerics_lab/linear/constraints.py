"""Resolve a master-slave constraint list so no master is itself constrained.

Physics-free constraint algebra, shared by the FEA fold
(``hamilton/discretizations/fem/_solve.py``) and the MEC reduction
(``hamilton/formulations/magnetic_circuit/_sparse.py``). It lived in the FEA module and MEC had a
SECOND implementation of the same idea -- "chains resolve in list order" -- carrying the same
order-dependence. One scheme, one copy.
"""

from __future__ import annotations

import numpy as np

__all__ = ['resolve_constraint_chains']


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
    that alone raises. ipm_uv_cummins under ``--max-aspect 4`` carries exactly this shape: of
    3892 hanging nodes, 4 are chained and those 4 form two mutual pairs
    (2590 <-> 8105, 7744 <-> 7803).

    Only the nodes actually entangled are solved; every other row is returned untouched, and a
    constraint set with no chains at all is returned AS THE SAME OBJECT, so meshes that never had
    the defect stay bit-for-bit.
    """
    declared = {int(h): [(int(m), float(w)) for m, w in masters] for h, masters in constraints}
    entangled = {h for h, ms in declared.items() if any(m in declared for m, _w in ms)}
    if not entangled:
        return constraints

    # Grow to the closure: a node the entangled set depends on must be solved with it.
    frontier = list(entangled)
    while frontier:
        h = frontier.pop()
        for m, _w in declared[h]:
            if m in declared and m not in entangled:
                entangled.add(m)
                frontier.append(m)

    order = sorted(entangled)
    pos = {h: i for i, h in enumerate(order)}
    n = len(order)
    free_targets = sorted({m for h in order for m, _w in declared[h] if m not in declared})
    fpos = {m: j for j, m in enumerate(free_targets)}
    a_mat = np.eye(n)
    b_mat = np.zeros((n, len(free_targets)))
    for h in order:
        i = pos[h]
        for m, w in declared[h]:
            if m in pos:
                a_mat[i, pos[m]] -= w
            elif m in declared:  # constrained but outside the closure cannot happen
                msg = f'constraint {h} references constrained node {m} outside the solved closure'
                raise ValueError(msg)
            else:
                b_mat[i, fpos[m]] += w
    try:
        x_mat = np.linalg.solve(a_mat, b_mat)
    except np.linalg.LinAlgError as exc:
        loop = ' -> '.join(str(h) for h in order)
        msg = (
            f'hanging constraints form a SINGULAR system over nodes ({loop}): the relations are '
            f'mutually dependent with no solution in terms of free DOFs, so no fold can satisfy '
            f'them. This is not a chain (chains resolve); it is a contradictory constraint set '
            f'and the mesh that produced it is wrong.'
        )
        raise ValueError(msg) from exc

    resolved = {h: [(free_targets[j], float(x_mat[pos[h], j])) for j in np.flatnonzero(x_mat[pos[h]])] for h in order}
    return [(h, resolved.get(h, ms)) for h, ms in declared.items()]
