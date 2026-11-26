"""Smoothed-aggregation algebraic multigrid for real SPD nodal operators.

This is not JMAG's 3D A-phi electromagnetic-decoupling algorithm. Its supported
surface is an actual Galerkin V-cycle, selectable by the existing FEA/MEC seam.
No mesh coordinates or optional package are needed; coarse spaces come from
strong matrix couplings and a constant tentative near-null vector.
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator

__all__ = ['COARSENINGS', 'aggregates', 'amg_preconditioner', 'strong_connections']


#: The selectable coarsenings. ``pairwise``: greedy strongest-neighbour pairs (each level shrinks ~2x),
#: prolongation smoothed on the full operator. ``smoothed-aggregation``: Vanek-Mandel-Brezina root-node
#: aggregates over the strong graph, prolongation smoothed on the FILTERED operator (weak couplings lumped
#: into the diagonal) so the coarse stencils stay bounded. Ruge-Stueben (C/F splitting) is not implemented.
COARSENINGS = ('pairwise', 'smoothed-aggregation')


def strong_connections(matrix, theta: float):
    """Off-diagonal couplings with ``|a_ij| >= theta sqrt(a_ii a_jj)``; ``theta = 0`` keeps every nonzero one."""
    coo = sp.coo_matrix(matrix)
    diagonal = np.abs(matrix.diagonal())
    keep = (coo.row != coo.col) & (coo.data != 0)
    keep &= np.abs(coo.data) >= theta * np.sqrt(diagonal[coo.row] * diagonal[coo.col])
    return sp.csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=matrix.shape)


def _tentative(labels, count: int):
    size = labels.size
    return sp.csr_matrix((np.ones(size), (np.arange(size), labels)), shape=(size, count))


def _pairwise_aggregates(strong):
    """Greedy strongest-neighbour pairs over the strong graph; unmatched rows form singleton aggregates."""
    size = strong.shape[0]
    labels = np.full(size, -1, dtype=int)
    count = 0
    for row in range(size):
        if labels[row] >= 0:
            continue
        labels[row] = count
        start, end = strong.indptr[row : row + 2]
        columns = strong.indices[start:end]
        values = strong.data[start:end]
        available = labels[columns] < 0
        if np.any(available):
            neighbour = columns[available][np.argmax(np.abs(values[available]))]
            labels[neighbour] = count
        count += 1
    return _tentative(labels, count)


def _root_node_aggregates(strong):
    """Standard aggregation: a free node whose strong neighbours are all free roots an aggregate of them.

    Phase 2 attaches each leftover node to the aggregate of its strongest aggregated neighbour; phase 3
    roots new aggregates at whatever is still free (with its free strong neighbours). A node with no strong
    neighbour is a singleton.
    """
    size = strong.shape[0]
    labels = np.full(size, -1, dtype=int)
    indptr, indices, data = strong.indptr, strong.indices, np.abs(strong.data)
    count = 0
    for row in range(size):
        columns = indices[indptr[row] : indptr[row + 1]]
        if labels[row] < 0 and columns.size and np.all(labels[columns] < 0):
            labels[row] = count
            labels[columns] = count
            count += 1
    attached = labels.copy()
    for row in np.flatnonzero(labels < 0):
        columns = indices[indptr[row] : indptr[row + 1]]
        values = data[indptr[row] : indptr[row + 1]]
        aggregated = labels[columns] >= 0
        if np.any(aggregated):
            attached[row] = labels[columns[aggregated][np.argmax(values[aggregated])]]
    labels = attached
    for row in range(size):
        if labels[row] >= 0:
            continue
        columns = indices[indptr[row] : indptr[row + 1]]
        labels[row] = count
        labels[columns[labels[columns] < 0]] = count
        count += 1
    return _tentative(labels, count)


def aggregates(matrix, *, coarsening: str = 'pairwise', theta: float = 0.0):
    """The tentative prolongation (one column per aggregate) of one level."""
    strong = strong_connections(matrix, theta)
    return _pairwise_aggregates(strong) if coarsening == 'pairwise' else _root_node_aggregates(strong)


def _prolongation(matrix, coarsening: str, theta: float):
    """The smoothed prolongation of one level, or ``None`` when the coarsening cannot shrink it."""
    tentative = aggregates(matrix, coarsening=coarsening, theta=theta)
    if tentative.shape[1] == matrix.shape[0]:
        return None
    if coarsening == 'pairwise':
        smoothed, weight = matrix, 2.0 / 3.0
    else:
        # The FILTERED operator: weak couplings move onto the diagonal, so P's sparsity follows the strong graph.
        strong = strong_connections(matrix, theta)
        # ``rowsum(A - S)`` is ``a_ii`` plus the row's weak couplings.
        smoothed = (strong + sp.diags(np.asarray((matrix - strong).sum(axis=1)).ravel())).tocsr()
        weight = 4.0 / 3.0
    diagonal = smoothed.diagonal()
    upper = float(np.max(np.asarray(abs(smoothed).sum(axis=1)).ravel() / diagonal))
    # Strict damping (2/3 on the full operator) keeps I-B A nonsingular even at the spectral bound, so
    # smoothing cannot erase a tentative coarse column; SA's 4/3 over a Gershgorin bound stays below 2/rho.
    inverse = weight / (upper * diagonal)
    return (tentative - sp.diags(inverse) @ smoothed @ tentative).tocsr()


def amg_preconditioner(matrix, *, coarsening: str = 'pairwise', theta: float = 0.0) -> LinearOperator:
    """Symmetric pre/post Jacobi V-cycle with smoothed Galerkin coarse spaces.

    ``coarsening`` is one of :data:`COARSENINGS`; ``theta`` is the strength-of-connection threshold: only
    couplings with ``|a_ij| >= theta sqrt(a_ii a_jj)`` may join an aggregate (``0`` keeps them all). An
    anisotropic operator needs ``theta > 0`` so its weak direction is not aggregated (numerics-lab #33).

    Assumes SPD, as CG does; refuses non-real, nonsymmetric and nonpositive
    diagonals. Coarsest Cholesky additionally refuses indefinite/singular coarse
    operators. The weighted smoother uses a Gershgorin spectral upper bound,
    not a guessed omega that might destabilize a strongly coupled operator.
    """
    if coarsening not in COARSENINGS:
        msg = f'unknown AMG coarsening {coarsening!r}; known: {COARSENINGS}'
        raise ValueError(msg)
    if not 0.0 <= theta < 1.0:
        msg = f'AMG strength threshold theta must lie in [0, 1), got {theta!r}'
        raise ValueError(msg)
    matrix = sp.csr_matrix(matrix, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if (
        matrix.shape[0] != matrix.shape[1]
        or not matrix.shape[0]
        or np.iscomplexobj(matrix.data)
        or not np.all(np.isfinite(matrix.data))
    ):
        msg = 'AMG requires a finite real nonempty square SPD operator'
        raise ValueError(msg)
    difference = (matrix - matrix.T).tocoo()
    if difference.nnz:
        scale = np.maximum(
            np.abs(np.asarray(matrix[difference.row, difference.col]).ravel()),
            np.abs(np.asarray(matrix[difference.col, difference.row]).ravel()),
        )
        if np.any(np.abs(difference.data) > 64 * np.finfo(float).eps * scale):
            msg = 'AMG requires a symmetric SPD operator'
            raise ValueError(msg)
        matrix = (0.5 * matrix + 0.5 * matrix.T).tocsr()
    levels = []
    operators = []
    while True:
        operators.append(matrix)
        diagonal = matrix.diagonal()
        if np.any(diagonal <= 0):
            msg = 'AMG requires positive diagonals on every level'
            raise ValueError(msg)
        if matrix.shape[0] <= 32:
            break
        prolongation = _prolongation(matrix, coarsening, theta)
        if prolongation is None:
            break  # no strong coupling left to aggregate: the coarsest solve below
        upper = float(np.max(np.asarray(abs(matrix).sum(axis=1)).ravel() / diagonal))
        inverse = (2.0 / 3.0) / (upper * diagonal)  # the Jacobi smoother's damping
        levels.append((matrix, inverse, prolongation))
        matrix = (prolongation.T @ matrix @ prolongation).tocsr()
    if matrix.shape[0] > 32:

        def coarse_solve(rhs):
            return rhs / diagonal

    else:
        try:
            factor = cho_factor(matrix.toarray())
        except np.linalg.LinAlgError as exc:
            msg = 'AMG coarse operator is not positive definite'
            raise ValueError(msg) from exc

        def coarse_solve(rhs):
            return cho_solve(factor, rhs)

    def cycle(level, rhs):
        if level == len(levels):
            return coarse_solve(rhs)
        operator, inverse, prolongation = levels[level]
        estimate = inverse * rhs
        residual = rhs - operator @ estimate
        estimate += prolongation @ cycle(level + 1, prolongation.T @ residual)
        return estimate + inverse * (rhs - operator @ estimate)

    result = LinearOperator(
        operators[0].shape,
        matvec=lambda rhs: cycle(0, np.asarray(rhs).ravel()),
        rmatvec=lambda rhs: cycle(0, np.asarray(rhs).ravel()),
        dtype=float,
    )
    result.level_sizes = tuple(operator.shape[0] for operator in operators)
    return result
