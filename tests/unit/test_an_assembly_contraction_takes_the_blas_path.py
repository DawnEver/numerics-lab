"""A multi-operand contraction must reach BLAS, and every assembly site must go through the one
helper that guarantees it.

THE MEASUREMENT THAT PUT THIS HERE, taken 2026-09-16 on ``cases/case_d`` at
``SPLINE_REFINE = (2, 2)`` with ``compile_iga_magnetostatic``. Every ``np.einsum`` on the
assembly path was timed by subscript, and the same call priced with ``optimize=True``:

===========================  =====  ==========  =========  ========
subscripts                       n  naive (s)   opt (s)     speedup
===========================  =====  ==========  =========  ========
``qid,qjd,q->ij``               38       82.50       2.33     35.5x
``qid,qde->qie``               812       26.89       1.47     18.3x
``qid,ic->qcd``                850        9.12       3.61      2.5x
``qi,qj,q->ij``                136        0.71       0.06     11.5x
TOTAL                         1972      119.23       7.48     15.9x
===========================  =====  ==========  =========  ========

119.2 s of a 136.5 s compile -- **87%** -- was one defect: ``numpy.einsum`` defaults to
``optimize=False``, which runs its own naive C loop instead of factoring the contraction into
pairwise ``tensordot`` calls that land on ``dgemm``. Both languages are compiled; what differed
was the ALGORITHM, not the implementation.

Note WHICH site dominated, because the lane's standing suspicion was the other one: the 82.5 s is
the VOLUME stiffness (``iga.py``), while ``trim_nitsche_blocks`` -- the 68 cut interfaces that
were the prime suspect for the assembly blow-up -- account for the 0.71 s row, 0.6% of the budget.

WHY A HELPER RATHER THAN ``optimize=True`` AT EACH SITE: the defect is a DEFAULT, so any site that
forgets the keyword silently re-acquires it and nothing fails. The scan below is what makes the
declaration one the code must consult.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from numerics_lab.contract import contract


@pytest.mark.parametrize(
    ('subscripts', 'shapes'),
    [
        # The four subscripts the profile above actually measured, at their measured shapes.
        ('qid,qjd,q->ij', ((400, 105, 2), (400, 105, 2), (400,))),
        ('qid,qde->qie', ((400, 105, 2), (400, 2, 2))),
        ('qid,ic->qcd', ((400, 105, 2), (105, 3))),
        ('qi,qj,q->ij', ((256, 210), (256, 210), (256,))),
    ],
)
def test_the_HELPER_AGREES_WITH_THE_NAIVE_CONTRACTION_it_replaces(subscripts, shapes) -> None:
    """Same contraction, different summation order -- so agreement is RELATIVE, not bitwise.

    BLAS blocks its accumulation, so the result is not bit-identical to numpy's naive loop and
    cannot be: reassociating a floating-point sum changes its rounding. The bar is therefore the
    only honest one, a relative agreement at the level reassociation can move a double, and it is
    stated against the operand scale rather than against the entries -- a stiffness block's entries
    span many orders of magnitude and a per-entry relative bar would be a statement about the
    smallest one.
    """
    rng = np.random.default_rng(20260916)
    operands = [rng.random(shape) for shape in shapes]
    naive = np.einsum(subscripts, *operands)
    fast = contract(subscripts, *operands)
    assert fast.shape == naive.shape
    scale = float(np.abs(naive).max())
    assert np.abs(fast - naive).max() <= 1.0e-12 * scale


def test_the_HELPER_IS_FASTER_THAN_THE_NAIVE_PATH_BY_A_STATED_RATIO() -> None:
    """The point of the helper, pinned as a RATIO between two ways of taking the SAME contraction.

    A wall time is a property of the box; a ratio measured back-to-back on one box is a property of
    the algorithm. The floor is 5x against a measured 35.5x, so the pin survives a busy box by a
    factor of seven and still cannot pass if the helper stops reaching BLAS.
    """
    rng = np.random.default_rng(20260916)
    gradients = rng.random((400, 105, 2))
    measure = rng.random(400)

    def timed(call) -> float:
        call()  # warm the path-cache and the BLAS thread pool out of the measurement
        best = float('inf')
        for _ in range(3):
            start = time.perf_counter()
            call()
            best = min(best, time.perf_counter() - start)
        return best

    naive = timed(lambda: np.einsum('qid,qjd,q->ij', gradients, gradients, measure))
    fast = timed(lambda: contract('qid,qjd,q->ij', gradients, gradients, measure))
    assert fast > 0.0
    assert naive / fast >= 5.0, f'the helper is only {naive / fast:.1f}x the naive path, not the 35.5x measured'
