"""``PatchBasis.evaluate`` allocates on the SUPPORT, not on the patch.

WHY THIS FILE EXISTS. A B-spline basis is LOCAL: at any parameter only ``(p+1)`` functions per
direction are nonzero, so ``(p+1)**dim`` in total -- 9 at degree 2. ``evaluate`` used to return
``(n_points, n_basis)`` DENSE over the WHOLE patch, and a patch at ``SPLINE_REFINE = (3, 3)``
carries 333 control points, so **97% of every array it returned was a structural zero**. The zeros
are exact (Cox-de Boor writes ``0.0`` outside the support and every later step multiplies or
divides that through unchanged), so they carried no information at all.

MEASURED, ``cases/field/pm/spm_slotted`` at ``SPLINE_REFINE = (2, 2)``, the spline leg through
``run_case_comparison`` -- the profile that named this defect before anything was changed:

    peak RSS                                       2.87 GB
    dense bytes returned, summed over 19613 calls  4.328 GB
    the same information at (p+1)**dim locality    0.385 GB   -> 11.2x
    largest SINGLE call    234200 points x 105 basis functions, one array of  0.55 GB

and 1.759 GB of that peak is not transient at all: ``SplineNonlinearSystem`` RETAINS
``values.values`` and ``values.gradients`` for every patch for the whole Picard loop, so the dense
zeros stay resident until the solve ends. That is 61% of the peak and it is what makes the cost
superlinear in ``refine`` -- refining grows the point count AND the patch the zeros are counted
over. ``SPLINE_REFINE = (3, 3)`` peaked at 22.1 GB and ``(4, 4)`` was measured still climbing at
30.7 GB.

WHAT IS ASSERTED HERE. The locality is a SHAPE claim with a FLOOR (a patch where the dense and the
local widths coincide could not fail it), and the equivalence is against an INDEPENDENT dense
Cox-de Boor written in this file rather than against a table this module produced -- so a scatter
that puts a basis function in the wrong column fails, which is the defect class this repo has paid
for twice: a cut cell given the wrong quadrature is a wrong field with no exception anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from motronics.contracts.ir.spline_space_ir import arc_patch_quarter_annulus
from motronics.euclid.mesh.spline_refine import refine_patch
from motronics.gauss.spline_basis import PatchBasis

#: How many times wider the DENSE width must be than the local one before the locality claim says
#: anything. Without it a bilinear patch -- 4 control points, a support of 4 -- passes every
#: assertion below while proving nothing, which is this repo's vacuous-scan shape.
MINIMUM_DENSITY_RATIO = 8.0


def _refined_annulus(refine: tuple[int, int] = (3, 3)):
    """A quarter annulus refined until its basis is far wider than any point's support."""
    return refine_patch(arc_patch_quarter_annulus(r_inner=0.02, r_outer=0.05), refine)


def _dense_cox_de_boor(knot: np.ndarray, degree: int, x: np.ndarray) -> np.ndarray:
    """An INDEPENDENT dense 1D B-spline evaluation, written from the recursion, as the oracle."""
    n_basis = len(knot) - degree - 1
    out = np.zeros((x.size, n_basis))
    for point_index, value in enumerate(np.asarray(x, dtype=float)):
        for i in range(n_basis):
            out[point_index, i] = _one_basis(knot, degree, i, float(value))
    return out


def _one_basis(knot: np.ndarray, degree: int, i: int, x: float) -> float:
    if degree == 0:
        inside = knot[i] <= x < knot[i + 1]
        if x == knot[-1] and knot[i] < knot[i + 1] and knot[i + 1] == knot[-1]:
            inside = True
        return 1.0 if inside else 0.0
    left_den = knot[i + degree] - knot[i]
    right_den = knot[i + degree + 1] - knot[i + 1]
    total = 0.0
    if left_den > 0:
        total += (x - knot[i]) / left_den * _one_basis(knot, degree - 1, i, x)
    if right_den > 0:
        total += (knot[i + degree + 1] - x) / right_den * _one_basis(knot, degree - 1, i + 1, x)
    return total


def _dense_tensor_values(patch, points: np.ndarray) -> np.ndarray:
    """The RATIONAL patch basis, dense over the whole patch, from the oracle above."""
    per_direction = [
        _dense_cox_de_boor(knot, degree, points[:, d])
        for d, (knot, degree) in enumerate(zip(patch.knots, patch.degrees, strict=True))
    ]
    values = per_direction[0]
    for nxt in per_direction[1:]:
        values = (values[:, None, :] * nxt[:, :, None]).reshape(values.shape[0], -1)
    weighted = values * patch.weights[None, :]
    return weighted / weighted.sum(axis=1, keepdims=True)


@pytest.fixture(scope='module')
def refined():
    return _refined_annulus()


@pytest.fixture(scope='module')
def sample_points(refined) -> np.ndarray:
    lo = np.array([float(k[0]) for k in refined.knots])
    hi = np.array([float(k[-1]) for k in refined.knots])
    rng = np.random.default_rng(20260916)
    interior = lo + (hi - lo) * rng.random((97, len(lo)))
    # The two domain ENDS are included on purpose: the end knot is the one parameter where the span
    # search has to be clamped, and a scatter that is off by one there is invisible in the interior.
    return np.vstack([interior, lo[None, :], hi[None, :]])


def test_the_fixture_is_dense_enough_for_the_locality_claim_to_mean_something(refined) -> None:
    """THE FLOOR. A patch whose support IS its basis cannot fail the two tests below."""
    basis = PatchBasis(refined)
    support = int(np.prod([p + 1 for p in refined.degrees]))
    ratio = basis.n_basis / support
    assert ratio > MINIMUM_DENSITY_RATIO, (
        f'the fixture carries {basis.n_basis} basis functions against a support of {support} '
        f'({ratio:.1f}x); below {MINIMUM_DENSITY_RATIO}x this file asserts nothing'
    )


def test_evaluate_allocates_on_the_support_rather_than_on_the_patch(refined, sample_points) -> None:
    """The arrays are ``(n_points, (p+1)**dim)`` wide, and every column is named by ``columns``.

    This is the whole fix in one assertion: the width is a property of the DEGREE, so it does not
    move when the patch is refined, which is what makes refinement affordable.
    """
    basis = PatchBasis(refined)
    support = int(np.prod([p + 1 for p in refined.degrees]))
    n_points = sample_points.shape[0]

    out = basis.evaluate(sample_points, derivative_order=2)
    assert out.values.shape == (n_points, support)
    assert out.columns.shape == (n_points, support)
    assert out.gradients.shape == (n_points, support, basis.parametric_dim)
    assert out.hessians.shape == (n_points, support, basis.parametric_dim, basis.parametric_dim)
    assert out.columns.min() >= 0
    assert out.columns.max() < basis.n_basis
    # Each point's support is a SET: a repeated column would double-count a basis function into
    # the stiffness matrix, and every row sum below would still be 1.0.
    for row in out.columns:
        assert len(set(row.tolist())) == support


def test_the_local_rows_scatter_back_to_the_dense_basis_the_oracle_computes(refined, sample_points) -> None:
    """EQUIVALENCE against an independent dense Cox-de Boor, not against a recorded table.

    A scatter into the wrong column keeps the partition of unity intact -- the same nine numbers
    are still there and still sum to one -- so a row-sum check cannot catch it and this one can.
    """
    basis = PatchBasis(refined)
    out = basis.evaluate(sample_points, derivative_order=0)
    expected = _dense_tensor_values(refined, sample_points)

    dense = out.dense_values()
    assert dense.shape == (sample_points.shape[0], basis.n_basis)
    assert np.allclose(dense, expected, rtol=0.0, atol=1e-12)
    # and the local rows themselves carry exactly the nonzeros of that dense array
    assert np.allclose(np.abs(dense).sum(axis=1), np.abs(out.values).sum(axis=1), rtol=1e-12, atol=0.0)
    assert np.allclose(out.values.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)


def test_the_geometry_map_is_reproduced_from_the_local_rows(refined, sample_points) -> None:
    """The rational basis times the control net is the PATCH -- the end-to-end statement.

    ``Patch.evaluate`` is an independent route to the same point, so this catches a locality bug
    that is self-consistent inside the basis (wrong columns AND a matching control-point gather).
    """
    basis = PatchBasis(refined)
    out = basis.evaluate(sample_points, derivative_order=0)
    gathered = refined.control_points[out.columns]  # (n_points, support, 3)
    mapped = np.einsum('ql,qlc->qc', out.values, gathered)
    assert np.allclose(mapped, refined.evaluate(sample_points), rtol=0.0, atol=1e-9)
