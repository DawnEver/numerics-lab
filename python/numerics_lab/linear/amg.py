"""Aggregation algebraic multigrid for real SPD nodal operators.

This is not JMAG's 3D A-phi electromagnetic-decoupling algorithm. Its supported
surface is an actual Galerkin V-cycle, selectable by the existing FEA/MEC seam.
No mesh coordinates or optional package are needed; coarse spaces come from
strong matrix couplings and a constant tentative near-null vector.

The setup and the cycle are Rust (``crates/numerics/src/linear/amg.rs``, which also states the
#33 mechanism); this module validates the operator and marshals it.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

from numerics_lab import _native

__all__ = ['COARSENINGS', 'SMOOTHINGS', 'aggregates', 'amg_preconditioner', 'strong_connections']


#: The selectable coarsenings. ``pairwise``: greedy strongest-neighbor pairs (each level shrinks ~2x).
#: ``smoothed-aggregation``: Vanek-Mandel-Brezina root-node aggregates over the strong graph -- the default.
#: Ruge-Stueben (C/F splitting) is not implemented.
COARSENINGS = ('pairwise', 'smoothed-aggregation')

#: How the tentative prolongation is smoothed. ``none``: plain aggregation. ``filtered``: one Jacobi step on
#: the operator with weak couplings lumped into the diagonal, so ``P`` follows the strong graph. ``full``: one
#: Jacobi step on the full operator -- on a 2x pairwise hierarchy it widens every coarse stencil, operator
#: complexity grows with N and setup goes ~N^2 (numerics-lab #33); it stays selectable.
SMOOTHINGS = ('none', 'filtered', 'full')

#: The smoothing that keeps each coarsening's hierarchy sparse.
_DEFAULT_SMOOTHING = {'pairwise': 'none', 'smoothed-aggregation': 'filtered'}


def _canonical(matrix):
    matrix = sp.csr_matrix(matrix)
    matrix.sum_duplicates()
    matrix.sort_indices()
    return (
        matrix.shape[0],
        matrix.indptr.astype(np.int64),
        matrix.indices.astype(np.int64),
        np.ascontiguousarray(matrix.data, dtype=float),
    )


def strong_connections(matrix, theta: float):
    """Off-diagonal couplings with ``|a_ij| >= theta sqrt(a_ii a_jj)``; ``theta = 0`` keeps every nonzero one."""
    n = matrix.shape[0]
    indptr, indices, values = _native.amg_strong_connections(*_canonical(matrix), theta)
    return sp.csr_matrix((values, indices, indptr), shape=(n, n))


def aggregates(matrix, *, coarsening: str = 'pairwise', theta: float = 0.0):
    """The tentative prolongation (one column per aggregate) of one level."""
    labels, count = _native.amg_aggregates(*_canonical(matrix), coarsening, theta)
    return sp.csr_matrix((np.ones(labels.size), (np.arange(labels.size), labels)), shape=(labels.size, count))


def amg_preconditioner(
    matrix, *, coarsening: str = 'smoothed-aggregation', theta: float = 0.08, smoothing: str | None = None
) -> LinearOperator:
    """Symmetric pre/post Jacobi V-cycle with Galerkin coarse spaces.

    ``coarsening`` is one of :data:`COARSENINGS`; ``theta`` is the strength-of-connection threshold: only
    couplings with ``|a_ij| >= theta sqrt(a_ii a_jj)`` may join an aggregate (``0`` keeps them all). An
    anisotropic operator needs ``theta > 0`` so its weak direction is not aggregated (numerics-lab #33).
    ``smoothing`` is one of :data:`SMOOTHINGS`; ``None`` picks the coarsening's sparse one. The
    returned operator carries ``level_sizes`` and ``operator_complexity`` (``sum(nnz_l) / nnz_0``).

    Assumes SPD, as CG does; refuses non-real, nonsymmetric and nonpositive
    diagonals. Coarsest Cholesky additionally refuses indefinite/singular coarse
    operators. The weighted smoother uses a Gershgorin spectral upper bound,
    not a guessed omega that might destabilize a strongly coupled operator.
    """
    if coarsening not in COARSENINGS:
        msg = f'unknown AMG coarsening {coarsening!r}; known: {COARSENINGS}'
        raise ValueError(msg)
    if smoothing is None:
        smoothing = _DEFAULT_SMOOTHING[coarsening]
    if smoothing not in SMOOTHINGS:
        msg = f'unknown AMG smoothing {smoothing!r}; known: {SMOOTHINGS}'
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
    hierarchy = _native.AmgHierarchy(*_canonical(matrix), coarsening, theta, smoothing)

    def cycle(rhs):
        return hierarchy.apply(np.ascontiguousarray(np.asarray(rhs, dtype=float).ravel()))

    result = LinearOperator(matrix.shape, matvec=cycle, rmatvec=cycle, dtype=float)
    result.level_sizes = tuple(hierarchy.level_sizes)
    # sum(nnz_l) / nnz_0: the hierarchy's storage and per-cycle work relative to the fine operator.
    result.operator_complexity = hierarchy.operator_complexity
    return result
