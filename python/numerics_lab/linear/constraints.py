"""Resolve a master-slave constraint list so no master is itself constrained.

Physics-free constraint algebra, shared by the FEA fold
(``hamilton/discretizations/fem/_solve.py``) and the MEC reduction
(``hamilton/formulations/magnetic_circuit/_sparse.py``). It lived in the FEA module and MEC had a
SECOND implementation of the same idea -- "chains resolve in list order" -- carrying the same
order-dependence. One scheme, one copy.
"""

from __future__ import annotations

__all__ = ['resolve_constraint_chains']


def resolve_constraint_chains(constraints):
    """Substitute a master that is ITSELF hanging until every constraint names free DOFs only.

    ``refine_quads_2to1`` performs no 2:1 balancing, so a refined region routinely produces a
    hanging node whose master lies on a finer edge and is hanging in turn. The fold eliminates
    one constraint at a time and replaces the slave's equation with an identity row, so a chain
    folded in declaration order lands on a row that is already eliminated: no error, no warning,
    and an answer that CHANGES WITH THE DECLARATION ORDER.

    Measured on a 5x5 unit-quad grid carrying ``A = 0.3x - 0.7y`` -- exactly representable, and
    every planted constraint exactly satisfied by it, so a correct fold cannot move the solution:
    independent constraints left ``max|A - exact|`` at 4e-16, while the chain
    ``12 <- {11, 13}`` with ``11 <- {6, 16}`` gave 5.500e-01 declared one way and 2.983e-07 the
    other. Two different wrong answers for one problem is what identifies the elimination rather
    than the physics. On ipm_uv_cummins it is why two mesh operators with different mechanisms
    and nearly double the hanging count between them (870 vs 1570) produce the same field error
    to four figures and the same zero torque.

    Resolution is exact substitution: ``A[h] = sum_m w_m A[m]`` with ``A[m] = sum_k u_k A[k]``
    becomes ``A[h] = sum_k (w_m u_k) A[k]``, weights multiplying and repeated targets summing.
    A CYCLE has no such resolution and raises rather than picking an arbitrary entry point
    (invariant.md: an unsupported combo raises, never silently wrong).

    The resolved masters are what the caller folds AND what it recovers from. Recovering from the
    DECLARED masters would reintroduce the ordering problem in the recovery step, since a declared
    master's own value is only available after ITS recovery; the resolved form references free
    DOFs whose values the solve already gives, and it is an exact rewrite of the declared
    constraint, so the declared relation holds after recovery either way.
    """
    declared = {int(h): [(int(m), float(w)) for m, w in masters] for h, masters in constraints}
    if not any(m in declared for masters in declared.values() for m, _w in masters):
        return constraints  # no chains: bit-for-bit the previous behaviour
    memo: dict[int, list[tuple[int, float]]] = {}

    def resolve(h: int, path: tuple[int, ...]) -> list[tuple[int, float]]:
        if h in memo:
            return memo[h]
        if h in path:
            cycle = ' -> '.join(str(n) for n in (*path[path.index(h) :], h))
            msg = (
                f'hanging constraints form a cycle ({cycle}): a node is transitively its own master, '
                f'so the relation has no resolution in terms of free DOFs and cannot be folded'
            )
            raise ValueError(msg)
        out: dict[int, float] = {}
        for m, w in declared[h]:
            if m in declared:
                for k, u in resolve(m, (*path, h)):
                    out[k] = out.get(k, 0.0) + w * u
            else:
                out[m] = out.get(m, 0.0) + w
        memo[h] = [(k, v) for k, v in out.items() if v != 0.0]
        return memo[h]

    return [(h, resolve(h, ())) for h in declared]
