"""AMG on the 2-D Poisson family at growing N: CG converges and the hierarchy stays sparse (numerics-lab #33).

Operator complexity is ``sum(nnz(A_l)) / nnz(A_0)``. The cause of the issue's ~N^2 setup: ``pairwise``
halves each level while smoothing the prolongation on the FULL operator, so every Galerkin product widens
the stencil and the coarse operators fill -- its complexity grows with N (the planted control below).
Smoothed aggregation over the strong graph (the default) shrinks ~3-4x per level with bounded complexity.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import cg

from numerics_lab.linear.amg import amg_preconditioner

_SIZES = (32, 64, 128)
_COMPLEXITY_BAR = 2.0  # dimensionless sum(nnz_l) / nnz_0


def _poisson(n: int):
    line = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n))
    identity = sp.identity(n)
    return (sp.kron(identity, line) + sp.kron(line, identity)).tocsr()


def _cg_iterations(matrix, preconditioner) -> int:
    rhs = np.ones(matrix.shape[0])
    count = [0]

    def counted(_x) -> None:
        count[0] += 1

    solution, info = cg(matrix, rhs, M=preconditioner, rtol=1e-8, maxiter=300, callback=counted)
    assert info == 0
    assert np.linalg.norm(rhs - matrix @ solution) / np.linalg.norm(rhs) == pytest.approx(0.0, abs=1e-8)
    return count[0]


@pytest.mark.parametrize('n', _SIZES)
def test_default_amg_converges_with_bounded_operator_complexity(n) -> None:
    matrix = _poisson(n)
    preconditioner = amg_preconditioner(matrix)
    assert _cg_iterations(matrix, preconditioner) <= 40
    assert preconditioner.operator_complexity <= _COMPLEXITY_BAR


def test_planted_control_pairwise_complexity_grows_with_n() -> None:
    complexities = [amg_preconditioner(_poisson(n), coarsening='pairwise').operator_complexity for n in _SIZES]
    assert all(later > earlier for earlier, later in pairwise(complexities))
    assert complexities[-1] > _COMPLEXITY_BAR


def test_a_single_level_hierarchy_has_unit_complexity() -> None:
    assert amg_preconditioner(_poisson(4)).operator_complexity == pytest.approx(1.0, abs=1e-12)
