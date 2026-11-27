"""AMG on the 2-D Poisson family at growing N: CG converges and the hierarchy stays sparse (numerics-lab #33).

Operator complexity is ``sum(nnz(A_l)) / nnz(A_0)``. The cause of the issue's ~N^2 setup: ``pairwise``
halved each level while smoothing the prolongation on the FULL operator, so every Galerkin product widened
the stencil and the coarse operators filled -- complexity grows with N (``smoothing='full'``, the planted
control below). Both coarsenings now default to a sparse smoothing: filtered for smoothed aggregation, none
(plain aggregation) for pairwise.
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


# CG iteration bars: plain pairwise aggregation measured 30 / 43 at 64^2 / 128^2 (2026-11-27) -- its V-cycle
# trades iterations for sparsity; smoothed aggregation stays grid-independent.
@pytest.mark.parametrize(('coarsening', 'bar'), [('smoothed-aggregation', 40), ('pairwise', 50)])
@pytest.mark.parametrize('n', _SIZES)
def test_default_amg_converges_with_bounded_operator_complexity(n, coarsening, bar) -> None:
    matrix = _poisson(n)
    preconditioner = amg_preconditioner(matrix, coarsening=coarsening)
    assert _cg_iterations(matrix, preconditioner) <= bar
    assert preconditioner.operator_complexity <= _COMPLEXITY_BAR


def test_planted_control_full_smoothing_complexity_grows_with_n() -> None:
    complexities = [
        amg_preconditioner(_poisson(n), coarsening='pairwise', smoothing='full').operator_complexity for n in _SIZES
    ]
    assert all(later > earlier for earlier, later in pairwise(complexities))
    assert complexities[-1] > _COMPLEXITY_BAR


def test_a_single_level_hierarchy_has_unit_complexity() -> None:
    assert amg_preconditioner(_poisson(4)).operator_complexity == pytest.approx(1.0, abs=1e-12)
