"""AMG on the #33 operator class: 3-D anisotropic conduction with a Robin film, at growing N (numerics-lab #33).

The issue's operator (litz transformer, n = 72k) is 3-D conduction with 0.41 W/(m K) across vs 138 W/(m K)
along the strands and a film coefficient on the surface. Here: the 7-point finite-volume operator on the
unit cube, the same conductivities, h = 10 W/(m^2 K) on every face, ``n^3`` cells.

Measured before the fix (2026-11-27, Python AMG on origin/integrate/main 9bb0edc), CG to rtol 1e-8:
- default smoothed aggregation: complexity 1.74 / 2.13 / 2.14 / 1.98, 13 / 16 / 17 / 16 iterations at
  n = 8 / 12 / 16 / 20 -- #33 does not reproduce on the default;
- ``pairwise`` (prolongation smoothed on the FULL operator): complexity 10.0 / 19.0 / 29.6 / 43.1, setup
  0.02 s -> 2.2 s for 16x the unknowns -- the ~N^2 of #33. Its stencil widens on every 2x level.
- pyamg 5 SA (reference): theta 0 -> 49 / 55 / 61 / 67 / 72 iterations at n = 8 .. 28 (it aggregates across
  the weak direction); theta 0.08 with its unfiltered smoothing -> complexity 4.2 / 9.5 / 9.8 / 12.2 / 23.9.

After (Rust AMG): smoothed aggregation identical (same levels, complexity and iterations: the port is
equivalent), 16 iterations at n = 28; pairwise (plain aggregation by default) complexity 1.83 / 1.97 / 1.92 /
2.02 / 1.98, 12 / 18 / 18 / 25 / 29 iterations at n = 8 / 12 / 16 / 20 / 28, setup 0.02 s at n = 28.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import cg

from numerics_lab.linear.amg import amg_preconditioner

_SIZES = (8, 12, 16)
_COMPLEXITY_BAR = 2.5  # dimensionless sum(nnz_l) / nnz_0
_K_ACROSS = 0.41  # W/(m K)
_K_ALONG = 138.0  # W/(m K)
_FILM = 10.0  # W/(m^2 K)


def _conduction(n: int):
    """Finite-volume ``-div(k grad T)`` on ``n^3`` cells of the unit cube, Robin film on every face; W/K."""
    width = 1.0 / n
    line = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n)).tolil()
    line[0, 0] = line[-1, -1] = 1.0
    line = line.tocsr()
    eye = sp.identity(n)
    stiffness = width * (
        _K_ALONG * sp.kron(sp.kron(eye, eye), line)
        + _K_ACROSS * sp.kron(sp.kron(eye, line), eye)
        + _K_ACROSS * sp.kron(sp.kron(line, eye), eye)
    )
    face = np.zeros(n)
    face[[0, -1]] = 1.0
    faces = face[None, None, :] + face[None, :, None] + face[:, None, None]
    return (stiffness + sp.diags(_FILM * width * width * faces.ravel())).tocsr()


def _cg_iterations(matrix, preconditioner) -> int:
    rhs = np.ones(matrix.shape[0])  # W
    count = [0]

    def counted(_x) -> None:
        count[0] += 1

    solution, info = cg(matrix, rhs, M=preconditioner, rtol=1e-8, maxiter=300, callback=counted)
    assert info == 0
    assert np.linalg.norm(rhs - matrix @ solution) / np.linalg.norm(rhs) == pytest.approx(0.0, abs=1e-8)
    return count[0]


@pytest.mark.parametrize('coarsening', ['smoothed-aggregation', 'pairwise'])
def test_complexity_stays_bounded_and_iterations_grid_independent(coarsening) -> None:
    iterations = []
    for n in _SIZES:
        matrix = _conduction(n)
        preconditioner = amg_preconditioner(matrix, coarsening=coarsening)
        assert preconditioner.operator_complexity <= _COMPLEXITY_BAR
        iterations.append(_cg_iterations(matrix, preconditioner))
    assert max(iterations) <= 30
    assert iterations[-1] - iterations[0] <= 10  # bounded as N grows 8x


def test_planted_control_full_operator_smoothing_fills_the_pairwise_hierarchy() -> None:
    # The #33 mechanism, kept selectable: the bar above can say no.
    complexities = [
        amg_preconditioner(_conduction(n), coarsening='pairwise', smoothing='full').operator_complexity for n in _SIZES
    ]
    assert complexities[0] < complexities[1] < complexities[2]
    assert complexities[-1] > 4 * _COMPLEXITY_BAR


def test_setup_grows_no_faster_than_linearly_times_a_margin() -> None:
    def seconds(n: int) -> float:
        matrix = _conduction(n)
        start = time.perf_counter()
        amg_preconditioner(matrix, coarsening='pairwise')
        return time.perf_counter() - start

    small, large = min(seconds(8) for _ in range(3)), min(seconds(16) for _ in range(3))
    assert large <= 8 * 4 * max(small, 1e-3)  # 8x the unknowns; a ~N^2 setup is 64x
