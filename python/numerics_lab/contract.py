"""The one contraction an assembly takes -- a multi-operand einsum routed onto BLAS.

WHY THIS MODULE EXISTS, AND IT IS A MEASUREMENT RATHER THAN A PREFERENCE. ``numpy.einsum``
defaults to ``optimize=False``, which evaluates a multi-operand expression with its own naive
nested C loop over every index at once. ``optimize=True`` instead factors the expression into a
sequence of PAIRWISE contractions, each of which numpy can express as a reshape plus a
``tensordot`` -- i.e. as ``dgemm``. Same arithmetic, same asymptotic operation count; what changes
is that the inner loop becomes a blocked, vectorised, cache-aware BLAS kernel instead of a
scalar one.

Measured 2026-09-16 on ``cases/case_d`` at ``SPLINE_REFINE = (2, 2)``, timing every
``np.einsum`` on ``compile_iga_magnetostatic``'s path by subscript and pricing the same call
with ``optimize=True``:

===========================  =====  ==========  =========  ========
subscripts                       n  naive (s)   opt (s)     speedup
===========================  =====  ==========  =========  ========
``qid,qjd,q->ij``               38       82.50       2.33     35.5x
``qid,qde->qie``               812       26.89       1.47     18.3x
``qid,ic->qcd``                850        9.12       3.61      2.5x
``qi,qj,q->ij``                136        0.71       0.06     11.5x
TOTAL                         1972      119.23       7.48     15.9x
===========================  =====  ==========  =========  ========

That 119.2 s was 87% of a 136.5 s compile. Note which row dominates: the 82.5 s is the VOLUME
stiffness, not the 68 cut interfaces that were the standing suspect for the isogeometric leg's
assembly cost -- those are the 0.71 s row, 0.6% of the budget.

THIS IS NOT BIT-IDENTICAL TO THE NAIVE PATH AND CANNOT BE. Reassociating a floating-point sum
changes its rounding, so the results agree to relative 1e-12 rather than exactly; BLAS's blocked
accumulation is if anything the more accurate of the two.
``tests/unit/test_an_assembly_contraction_takes_the_blas_path.py`` pins both halves -- the
agreement and the ratio -- and scans the assembly modules so that a site cannot silently fall back
onto the default.

WHY A NAMED HELPER RATHER THAN THE KEYWORD AT EACH SITE: the defect is a DEFAULT. A site that
omits the keyword re-acquires it, returns the right answer, and fails nothing -- which is this
repo's dominant defect shape, a declaration that lies. One spelling gives the scan something to
consult.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ['contract']


def contract(subscripts: str, *operands: NDArray[np.float64]) -> NDArray[np.float64]:
    """Contract ``operands`` according to ``subscripts``, via the pairwise BLAS path.

    A drop-in for :func:`numpy.einsum` on the assembly path, differing only in that the
    contraction ORDER is chosen rather than left at numpy's naive default.

    Args:
        subscripts: the einsum subscript string, e.g. ``'qid,qjd,q->ij'``.
        *operands: the arrays to contract. TWO OR MORE -- a single-operand einsum is a transpose
            and a trace, has no pairwise ordering to choose, and gains nothing here.

    Returns:
        The contracted array.

    Raises:
        ValueError: on fewer than two operands, because there is then no contraction order to
            optimise and the caller means :func:`numpy.einsum` rather than this.

    """
    if len(operands) < 2:
        msg = (
            f'contract() was given {len(operands)} operand(s) for {subscripts!r}; it exists to choose '
            'a PAIRWISE contraction order, which needs at least two. A one-operand einsum is a '
            'transpose or a trace -- call numpy.einsum directly for that.'
        )
        raise ValueError(msg)
    return np.einsum(subscripts, *operands, optimize=True)
