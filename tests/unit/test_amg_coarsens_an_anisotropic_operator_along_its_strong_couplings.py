"""AMG coarsens an anisotropic operator along its STRONG couplings (numerics-lab #33).

The pairwise coarsening paired every row with its strongest neighbor over every nonzero coupling, so each
level shrank ~2x and the smoothed prolongation filled the coarse operators (the issue's 3-D litz conduction:
13 levels, setup ~N^2). A strength threshold ``theta`` keeps only ``|a_ij| >= theta sqrt(a_ii a_jj)``, and
``coarsening='smoothed-aggregation'`` builds root-node aggregates over that graph with a prolongation
smoothed on the filtered operator.

Measured here on the 2-D anisotropic Poisson (eps = 1e-3), CG to rtol 1e-8: pairwise CONVERGES (14
iterations at 64^2, 16 at 128^2), so it is recorded as passing, not xfailed; what it does not do is shrink
-- 8 levels at 64^2, 10 at 128^2, setup 0.18 s -> 2.9 s for 4x the unknowns. Smoothed aggregation: 17 / 19
iterations, setup 0.04 s -> 0.13 s, first level shrinking ~2.9x. Those pairwise figures are its prolongation
smoothed on the FULL operator, now ``smoothing='full'``; pairwise defaults to plain aggregation (bounded
complexity, numerics-lab #33), which converges in 38 iterations at 64^2 -- a plain-aggregation V-cycle
gives up iterations for sparsity.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import cg

from numerics_lab.linear.amg import aggregates, amg_preconditioner, strong_connections

_EPS = 1e-3


def _anisotropic(n: int, eps: float = _EPS):
    """``-u_xx - eps u_yy`` on an ``n x n`` grid; index ``i + n j`` (x fastest)."""
    line = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n))
    identity = sp.identity(n)
    return (sp.kron(identity, line) + eps * sp.kron(line, identity)).tocsr()


def _cg_iterations(matrix, preconditioner) -> tuple[int, float]:
    rhs = np.ones(matrix.shape[0])
    count = [0]

    def counted(_x) -> None:
        count[0] += 1

    solution, info = cg(matrix, rhs, M=preconditioner, rtol=1e-8, maxiter=300, callback=counted)
    assert info == 0
    return count[0], float(np.linalg.norm(rhs - matrix @ solution) / np.linalg.norm(rhs))


def test_smoothed_aggregation_converges_on_the_anisotropic_operator() -> None:
    matrix = _anisotropic(64)
    preconditioner = amg_preconditioner(matrix, coarsening='smoothed-aggregation', theta=0.08)
    iterations, relative = _cg_iterations(matrix, preconditioner)
    assert iterations <= 25
    assert relative == pytest.approx(0.0, abs=1e-8)
    assert preconditioner.level_sizes[1] <= matrix.shape[0] / 2.5


@pytest.mark.parametrize(('smoothing', 'bar'), [('full', 25), ('none', 45)])
def test_pairwise_converges_but_only_halves_each_level(smoothing, bar) -> None:
    matrix = _anisotropic(64)
    preconditioner = amg_preconditioner(matrix, coarsening='pairwise', theta=0.0, smoothing=smoothing)
    iterations, relative = _cg_iterations(matrix, preconditioner)
    assert iterations <= bar
    assert relative == pytest.approx(0.0, abs=1e-8)
    assert preconditioner.level_sizes[:4] == (4096, 2048, 1024, 512)


def test_smoothed_aggregation_without_a_threshold_degrades() -> None:
    # theta = 0 aggregates ACROSS the weak direction: the coarse space misses the smooth-along-x error.
    matrix = _anisotropic(64)
    preconditioner = amg_preconditioner(matrix, coarsening='smoothed-aggregation', theta=0.0)
    iterations, _relative = _cg_iterations(matrix, preconditioner)
    assert iterations > 100


def test_planted_control_the_threshold_drops_the_weak_couplings() -> None:
    matrix = _anisotropic(8)
    strong = strong_connections(matrix, 0.08).tocoo()
    assert np.all(np.abs(strong.row - strong.col) == 1)  # only the x neighbors survive
    assert strong_connections(matrix, 0.0).nnz == matrix.nnz - matrix.shape[0]  # theta = 0 keeps every one


@pytest.mark.parametrize('coarsening', ['pairwise', 'smoothed-aggregation'])
def test_planted_control_no_aggregate_spans_the_weak_direction(coarsening) -> None:
    # The same coarsening at theta = 0 DOES span grid rows: the check can say no.
    assert _spans_grid_rows(aggregates(_anisotropic(15), coarsening=coarsening, theta=0.0), 15)
    assert not _spans_grid_rows(aggregates(_anisotropic(15), coarsening=coarsening, theta=0.08), 15)


def _spans_grid_rows(tentative, n: int) -> bool:
    """Whether some aggregate holds nodes of two grid rows (``y`` neighbors, the weak direction)."""
    entries = tentative.tocoo()
    grid_rows: dict[int, set[int]] = {}
    for node, aggregate in zip(entries.row, entries.col, strict=True):
        grid_rows.setdefault(int(aggregate), set()).add(int(node) // n)
    return any(len(rows) > 1 for rows in grid_rows.values())


@pytest.mark.parametrize(
    ('options', 'match'),
    [({'coarsening': 'ruge-stueben'}, 'coarsening'), ({'theta': 1.0}, 'theta'), ({'smoothing': 'twice'}, 'smoothing')],
)
def test_an_unknown_coarsening_or_threshold_is_refused(options, match) -> None:
    with pytest.raises(ValueError, match=match):
        amg_preconditioner(_anisotropic(8), **options)
