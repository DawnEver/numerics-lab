"""Quadrature over a spline patch: where the points are, and what is summed at them.

The counterpart of `motronics.gauss.spline_basis`, which owns the BASIS. Evaluation points are an
ARGUMENT to a basis, never a mode of it -- Greville points, Gauss points, plot sampling and
read-back at another discretization's sample points are four different point sets over one basis
-- so choosing them is this module's job and evaluating there is that one's.

It holds the tensor-product Gauss rule (:func:`patch_quadrature`, :func:`gauss_legendre_1d`) and
the two reductions an assembly takes over it: :func:`scatter_to_patch`, which lifts a local
column block back to patch-wide dofs, and :func:`gradient_gram_triplets`, which forms the
gradient Gram matrix in batches bounded by :data:`GRAM_BATCH_POINTS`.
"""

from __future__ import annotations

import itertools
from functools import lru_cache
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from motronics.contracts.ir.spline_space_ir import Patch
from motronics.gauss.contract import contract
from motronics.gauss.spline_basis import BasisValues, SplineBasisError

__all__ = [
    'EXTRA_QUADRATURE_POINTS',
    'GRAM_BATCH_POINTS',
    'QuadratureRule',
    'diffusion_blocks',
    'gauss_legendre_1d',
    'gradient_gram_triplets',
    'patch_quadrature',
    'scatter_to_patch',
]


class QuadratureRule(NamedTuple):
    """Parametric quadrature points and their weights, in the patch's parameter domain.

    The weights carry the PARAMETRIC measure only. Multiply by
    :attr:`BasisValues.det_jacobian` to integrate over physical space -- kept separate because the
    determinant depends on the geometry map and the rule does not, so a caller that wants a
    parametric integral (a knot-span measure, a refinement indicator) is not forced through the
    geometry.
    """

    points: NDArray[np.float64]
    weights: NDArray[np.float64]


#: Extra quadrature points beyond the polynomial rule. The area element of a rational patch is NOT
#: a polynomial, so no fixed rule is exact for it; four extra points per direction drives the
#: quarter-annulus area to ~1e-10 relative (measured) and costs nothing at these sizes. It is a
#: CHOICE, recorded as one, not an exactness claim.
EXTRA_QUADRATURE_POINTS = 4


def gauss_legendre_1d(n: int, lo: float, hi: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``n``-point Gauss-Legendre points and weights mapped onto ``[lo, hi]``.

    Exact for polynomials up to degree ``2n - 1``; the test asserts the BOUNDARY case, since a rule
    checked only on a low-degree integrand cannot tell a correct rule from an over-generous one.
    """
    x, w = _reference_gauss(int(n))
    half = 0.5 * (hi - lo)
    return lo + half * (x + 1.0), half * w


@lru_cache(maxsize=64)
def _reference_gauss(n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The nodes and weights on ``[-1, 1]``, which depend on ``n`` and on NOTHING ELSE.

    ``leggauss`` solves an eigenproblem, and the span it is mapped onto is a scaling applied after.
    Assembling one motor asked for the same three or four values of ``n`` **8660 times**: 2.455 s of
    a 9.101 s whole-motor assembly, 27% of it, spent recomputing identical arrays. The cache is on
    the reference rule rather than on the mapped one because the interval is different almost every
    call and would make every entry a miss.

    The arrays are marked read-only: a cached array a caller mutates in place is a defect that
    shows up somewhere else entirely.
    """
    x, w = np.polynomial.legendre.leggauss(n)
    x.flags.writeable = False
    w.flags.writeable = False
    return x, w


#: How many quadrature points one :func:`gradient_gram_triplets` batch may hold.
#:
#: A CEILING on the one intermediate still proportional to the point count, not a tuning
#: preference: the per-point outer product is ``n_support**2`` doubles per point, so this caps it
#: at roughly ``GRAM_BATCH_POINTS * n_support**2 * 8`` bytes -- 75 MB at degree 2 -- however many
#: points a trimmed rule turns out to carry. Batches split on CELL boundaries, so the value cannot
#: change the answer, only the footprint.
GRAM_BATCH_POINTS = 262144


def scatter_to_patch(values: BasisValues, weighted: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sum ``weighted`` -- ``(n_points, n_support)`` -- into the patch-wide ``(n_basis,)`` vector.

    The load-vector half of locality. A dense assembly wrote ``(rows * measure).sum(axis=0)`` and
    got a patch-wide vector for free because the array was already patch-wide; on the support the
    same statement is a scatter-add, and ``bincount`` is the vectorised spelling of one.
    """
    return np.bincount(
        values.columns.ravel(), weights=np.asarray(weighted, dtype=float).ravel(), minlength=values.n_basis
    )


def gradient_gram_triplets(
    values: BasisValues, dof_ids: NDArray[np.intp], measure: NDArray[np.float64]
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64]]:
    """COO triplets of ``INT measure * grad N_i . grad N_j`` over one patch, built on the SUPPORT.

    THE CELL IS THE UNIT THAT EMITS ENTRIES, not the point and not the patch. Points sharing a knot
    cell share a support, so this emits ``n_cells * n_support**2`` entries -- against the
    ``n_basis**2`` a dense patch block holds (11025 for 6 numbers per point on spm_slotted's
    carriers at refine (2, 2), and growing with the square of the refinement) and the
    ``n_points * n_support**2`` a per-point scatter would emit. The cell is identified by the FIRST
    column of the support, which is a bijection with the span indices that produced it.

    ``measure`` carries whatever the caller integrates against -- a reluctivity, a conductivity, a
    bare area element. Nothing here knows which, which is why it lives in ``gauss``.

    The sum over quadrature points is REASSOCIATED into per-cell partial sums. That is a last-bit
    change and not a modelling one, the same class as the BLAS reassociation ``gauss.contract``
    already accepts.
    """
    if values.gradients is None:
        msg = 'a gradient Gram block was requested but the basis was evaluated at derivative order 0'
        raise SplineBasisError(msg)
    key = values.columns[:, 0]
    order = np.argsort(key, kind='stable')
    key_sorted = key[order]
    starts = np.flatnonzero(np.concatenate([[True], key_sorted[1:] != key_sorted[:-1]]))
    gradients = values.gradients[order]
    weights = np.asarray(measure, dtype=float)[order]

    # BATCHED OVER CELLS, because the per-point outer product is the one array here still
    # proportional to the POINT COUNT: ``(n_points, n_support, n_support)`` is 36 doubles per point
    # at degree 2. A batch boundary is a CELL boundary, so every cell's ``reduceat`` is still taken
    # whole and the result is the same to the last bit -- this bounds the footprint, not the answer,
    # and the six recorded spm_slotted values came back byte-for-byte identical with and without it.
    #
    # WHAT IT IS NOT, MEASURED. It was introduced on the guess that this intermediate dominated the
    # compile's peak at ``SPLINE_REFINE = (4, 4)``, and that guess is REFUTED: the (4, 4) compile
    # peaks at 34.37 GB with the ceiling and 34.37 GB without it, unchanged, while the wall fell
    # 378.9 s -> 311.3 s. So this is a real ceiling on a real array that is not the binding one --
    # the (4, 4) peak lives in the trimmed rule's own point accumulation, not here. Kept because a
    # bound proportional to nothing is what the rest of this module now provides, and quoted with
    # the refutation so nobody reads it as the fix for that peak.
    bounds = np.append(starts, key_sorted.size)
    blocks_by_batch = []
    batch_start = 0
    while batch_start < starts.size:
        batch_end = int(np.searchsorted(bounds, bounds[batch_start] + GRAM_BATCH_POINTS, side='right')) - 1
        batch_end = max(batch_end, batch_start + 1)
        lo, hi = int(bounds[batch_start]), int(bounds[batch_end])
        outer = contract('qld,qmd,q->qlm', gradients[lo:hi], gradients[lo:hi], weights[lo:hi])
        blocks_by_batch.append(np.add.reduceat(outer, starts[batch_start:batch_end] - lo, axis=0))
        batch_start = batch_end
    blocks = np.concatenate(blocks_by_batch, axis=0)
    support = np.asarray(dof_ids, dtype=np.intp)[values.columns[order][starts]]
    width = support.shape[1]
    return (
        np.repeat(support, width, axis=1).ravel(),
        np.tile(support, (1, width)).ravel(),
        blocks.ravel(),
    )


def patch_quadrature(patch: Patch, *, extra_points: int = 0) -> QuadratureRule:
    """A tensor Gauss rule over every non-empty knot span of ``patch``.

    ``degree + 1 + extra_points`` points per direction per span: exact for the polynomial part of
    the integrand, and ``extra_points`` is the honest knob for the rational part, whose integrand is
    NOT a polynomial and for which no fixed rule is exact. The default of zero is the usual
    engineering choice and is a choice, not an exactness claim.
    """
    per_direction = []
    for knot, degree in zip(patch.knots, patch.degrees, strict=True):
        breaks = np.unique(knot)
        points, weights = [], []
        for lo, hi in itertools.pairwise(breaks):
            x, w = gauss_legendre_1d(degree + 1 + extra_points, float(lo), float(hi))
            points.append(x)
            weights.append(w)
        per_direction.append((np.concatenate(points), np.concatenate(weights)))

    grids = np.meshgrid(*[p for p, _ in per_direction], indexing='ij')
    weight_grids = np.meshgrid(*[w for _, w in per_direction], indexing='ij')
    points = np.stack([g.ravel(order='F') for g in grids], axis=1)
    weights = np.prod(np.stack([g.ravel(order='F') for g in weight_grids], axis=1), axis=1)
    return QuadratureRule(points=points, weights=weights)


def diffusion_blocks(
    gradients: NDArray[np.float64],
    conductivity: NDArray[np.float64],
    measure: NDArray[np.float64],
) -> NDArray[np.float64]:
    """``INT k grad(N_i) . grad(N_j)`` per cell, for ANY space that hands over its quadrature.

    The one spelling of the diffusion operator (IGA plan, phase 3), physics-free. A P1 simplex is a cell with one
    quadrature point -- its constant gradient, weighted by its measure -- and a spline cell carries
    its own rule; the operator reads neither the basis nor the element shape.

    Args:
        gradients: ``(n_cell, n_q, n_basis, dim)`` PHYSICAL basis gradients at each point.
        conductivity: ``(n_cell, dim)`` diagonal conductivity per cell.
        measure: ``(n_cell, n_q)`` quadrature weight times ``|det J|``.

    Returns:
        ``(n_cell, n_basis, n_basis)``.

    """
    return contract('eqid,eqjd,ed,eq->eij', gradients, gradients, conductivity, measure)
