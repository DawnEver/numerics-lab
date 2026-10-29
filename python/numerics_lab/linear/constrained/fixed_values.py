"""Dirichlet pinning on COO triplets -- pure algebra, no physics, no mesh.

Moved out of ``hamilton.discretizations.fem._solve`` (V2 step 2b,
``.claude/memory/2026/07/25/plan-axis-decoupling.md``). The function never referenced an element,
a material or a chart: it zeroes the rows of pinned indices, appends unit diagonal entries, and
pins the corresponding rhs entries. That makes it a ``gauss`` citizen by the dependency arrow --
and the reason it MATTERS which package it lives in is that ``studies/harmonic`` needed it and
could only reach it by importing a PRIVATE symbol out of the FEA discretization
(a downstream FEM module's ``_apply_dirichlet``). That import was
one of the two remaining welds pinning the time-harmonic regime to the FEA discretization; the
operation itself was never FEA-specific, so the honest fix is to move it, not to inject it.

Dtype-agnostic by construction: the pinned value is written into ``rhs`` as given (a complex
phasor stays complex) and the appended diagonal is real ``1.0``, which promotes correctly against
a complex value array.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, diags

__all__ = ['apply_dirichlet_csr', 'apply_dirichlet_triplets', 'lift_dirichlet_rhs']


def apply_dirichlet_csr(
    matrix: csr_matrix,
    rhs: NDArray[np.number],
    pinned: NDArray[np.intp],
    values: NDArray[np.number] | None = None,
) -> tuple[csr_matrix, NDArray[np.number]]:
    """Apply symmetric Dirichlet elimination to a CSR system.

    Non-zero prescribed values are lifted into ``rhs`` as ``rhs - matrix @ g`` before their
    columns are removed. Pinned rows and columns then become identity rows, preserving symmetry
    for solvers that require it. This operation depends on no mesh, basis, or physics.
    ``rhs`` may contain multiple columns. A vector of prescribed values applies to
    every column; a matrix supplies each column's own boundary values.
    """
    if pinned.size == 0:
        return matrix, rhs
    lifted = lift_dirichlet_rhs(matrix, rhs, pinned, values)

    # D A D + (I - D), D = diag(free mask): pinned rows and columns cleared, unit diagonal. Free
    # entries are multiplied by exactly 1.0. Sparse products, not a per-dof LIL edit (ruling 30).
    free = np.ones(matrix.shape[0], dtype=float)
    free[pinned] = 0.0
    keep = diags(free, format='csr')
    constrained = (keep @ csr_matrix(matrix) @ keep + diags(1.0 - free, format='csr')).tocsr()
    constrained.eliminate_zeros()
    constrained.sort_indices()
    return constrained, lifted


def lift_dirichlet_rhs(
    matrix: csr_matrix,
    rhs: NDArray[np.number],
    pinned: NDArray[np.intp],
    values: NDArray[np.number] | None = None,
) -> NDArray[np.number]:
    """Lift each RHS using its declared boundary values without rebuilding the operator."""
    dtype = np.result_type(
        matrix.dtype, rhs.dtype, np.float64, np.asarray(values).dtype if values is not None else float
    )
    prescribed = np.zeros(rhs.shape, dtype=dtype)
    if values is not None:
        boundary_values = np.asarray(values)
        if rhs.ndim == 2 and boundary_values.ndim == 1:
            boundary_values = boundary_values[:, None]
        prescribed[pinned] = boundary_values
    lifted = rhs - matrix @ prescribed

    lifted = lifted.copy()
    lifted[pinned] = prescribed[pinned]
    return lifted


def apply_dirichlet_triplets(row, col, val, rhs, bound_dirichlet, bias):
    """Pin ``rhs[i] = value + bias`` for every index ``i`` in ``bound_dirichlet``.

    Args:
        row: COO row indices.
        col: COO column indices.
        val: COO values (real or complex).
        rhs: dense right-hand side, **modified in place** at the pinned indices.
        bound_dirichlet: ``{value: index_ids}`` -- the prescribed value keys the group of indices
            it is imposed on (the shape the FEA/harmonic boundary maps already use).
        bias: added to every prescribed value (the FEA sliding-band A-offset).

    Returns:
        The ``(row, col, val)`` triplets, extended with one unit-diagonal entry per pinned index.
        Existing entries in a pinned row are zeroed rather than removed, so the returned arrays
        stay valid COO input.

    """
    for value, _index_ids in bound_dirichlet.items():
        index_ids = np.unique(_index_ids)
        if len(index_ids) < 1:
            continue
        rhs[index_ids] = value + bias

        val[np.where(np.isin(row, index_ids))] = 0
        row = np.concatenate((row, index_ids))
        col = np.concatenate((col, index_ids))
        val = np.concatenate((val, np.ones(len(index_ids))))

    return row, col, val
