"""Zero-fill incomplete Cholesky, preserving the lower triangle's sparsity."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, splu

from numerics_lab.linear.constrained.fixed_values import apply_dirichlet_csr
from numerics_lab.linear.constrained.null_space_pinning import constraint_rows

__all__ = ['IC0Operator', 'ic0_preconditioner', 'iccg_system']


def iccg_system(matrix, rhs, x0):
    """Exactly lift identity equations; never average nonsymmetric physical entries.

    Row-only boundary/gauge pins obscure an otherwise SPD free operator. Their
    values are known algebraically, so eliminate columns and preserve them exactly
    in the initial guess even when the inner absolute tolerance is loose.
    """
    csr = sp.csr_matrix(matrix, copy=True)
    csr.sum_duplicates()
    csr.eliminate_zeros()
    pinned = np.flatnonzero(constraint_rows(csr) & (csr.diagonal() == 1.0))
    if not pinned.size:
        return matrix, rhs, x0
    matrix, lifted = apply_dirichlet_csr(csr, rhs, pinned, np.asarray(rhs)[pinned])
    seed = np.zeros_like(lifted) if x0 is None else np.array(x0, copy=True)
    seed[pinned] = np.asarray(rhs)[pinned]
    return matrix, lifted, seed


def ic0_preconditioner(matrix) -> LinearOperator:
    """Return (L L.T)^-1; refuse non-real, nonsymmetric or nonpositive pivots.

    No diagonal shifting or ILU substitution: those are different algorithms.
    """
    a = sp.csr_matrix(matrix, copy=True)
    a.sum_duplicates()
    a.eliminate_zeros()
    if a.shape[0] != a.shape[1] or np.iscomplexobj(a.data) or not np.all(np.isfinite(a.data)):
        msg = 'IC0 requires a finite real square matrix'
        raise ValueError(msg)
    difference = (a - a.T).tocoo()
    if difference.nnz:
        # Pairwise relative tolerance is invariant to units; no absolute "1" floor.
        scale = np.maximum(
            np.abs(np.asarray(a[difference.row, difference.col]).ravel()),
            np.abs(np.asarray(a[difference.col, difference.row]).ravel()),
        )
        if np.any(np.abs(difference.data) > 64 * np.finfo(float).eps * scale):
            msg = 'IC0 requires a symmetric matrix (apart from relative roundoff)'
            raise ValueError(msg)
        a = (0.5 * (a + a.T)).tocsr()  # preconditioner only; the solved operator is unchanged
    lower = sp.tril(a, format='csr')
    rows = []
    for i in range(a.shape[0]):
        row = dict(
            zip(
                lower.indices[lower.indptr[i] : lower.indptr[i + 1]],
                lower.data[lower.indptr[i] : lower.indptr[i + 1]],
                strict=True,
            )
        )
        for j in sorted(k for k in row if k < i):
            row[j] = (row[j] - sum(row[k] * rows[j].get(k, 0.0) for k in row if k < j)) / rows[j][j]
        pivot = row.get(i, 0.0) - sum(value * value for j, value in row.items() if j < i)
        if not np.isfinite(pivot) or pivot <= 0:
            msg = f'IC0 nonpositive pivot at row {i}'
            raise ValueError(msg)
        row[i] = np.sqrt(pivot)
        rows.append(row)
    indices, values, pointers = [], [], [0]
    for row in rows:
        for j in sorted(row):
            indices.append(j)
            values.append(row[j])
        pointers.append(len(indices))
    factor = sp.csr_matrix((values, indices, pointers), shape=a.shape)
    return IC0Operator(factor)


class IC0Operator(LinearOperator):
    """``(L L.T)^-1`` applied by two PRE-FACTORED triangular sweeps.

    ``spsolve_triangular`` re-validates, re-sorts and re-multiplies its operand on every call --
    ~89 matvecs of work for two sweeps (ruling 30, `tests/unit/test_the_ic0_apply_is_factored_once.py`).
    SuperLU on ``L`` itself, natural column order and diagonal pivots, IS ``L`` (no fill, no
    permutation -- checked, never assumed); its plain solve sweeps ``L`` and its transposed solve
    sweeps ``L.T``, both compiled, once set up.
    """

    def __init__(self, lower) -> None:
        super().__init__(dtype=float, shape=lower.shape)
        self.lower = lower
        self._sweeps = splu(lower.tocsc(), permc_spec='NATURAL', diag_pivot_thresh=0.0, options={'SymmetricMode': True})
        identity = np.arange(lower.shape[0])
        if not (np.array_equal(self._sweeps.perm_r, identity) and np.array_equal(self._sweeps.perm_c, identity)):
            msg = 'IC0 sweeps: SuperLU permuted the triangular factor'
            raise ValueError(msg)

    def _matvec(self, vector):
        flat = np.asarray(vector, dtype=float).ravel()
        return self._sweeps.solve(self._sweeps.solve(flat), trans='T')

    def _rmatvec(self, vector):
        return self._matvec(vector)
