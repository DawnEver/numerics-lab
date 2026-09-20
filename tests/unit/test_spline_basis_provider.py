"""The physics-free spline basis/quadrature provider (IGA plan, deliverable 2).

Four interface constraints are asserted here rather than described, because each was chosen to
keep a KNOWN consumer reachable and an interface that quietly drops one cannot be told from one
that keeps it:

1. derivative order is a PARAMETER (collocation needs second derivatives);
2. evaluation points are an ARGUMENT, decoupled from the quadrature rule;
3. geometry degree and field degree are INDEPENDENT (the repo already ships a superparametric
   kernel, so welding them would contradict a kernel in the tree);
4. the unit of delivery grows into a COMPLEX, so the space carries which form-degree it is.

The numerical claims are checked against properties that hold for ANY correct implementation --
partition of unity, the derivative of a reproduced polynomial, the analytic area of a curved cell --
never against a table of numbers this module itself produced.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from motronics.contracts.ir.spline_space_ir import Patch, arc_patch_quarter_annulus
from motronics.euclid.mesh.producers.patch import annular_sector_patches
from motronics.euclid.mesh.spline_refine import refine_space
from motronics.gauss.quadrature import (
    gauss_legendre_1d,
    patch_quadrature,
)
from motronics.gauss.spline_basis import (
    FormDegree,
    PatchBasis,
    SplineBasisError,
)


def _bilinear_unit_square() -> Patch:
    return Patch.build(
        degrees=(1, 1),
        knots=(np.array([0.0, 0.0, 1.0, 1.0]), np.array([0.0, 0.0, 1.0, 1.0])),
        control_points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
    )


def _biquadratic_unit_square() -> Patch:
    """A p=q=2 patch on [0,1]^2 with a uniformly spaced control net -- geometry is still the square."""
    xs = np.linspace(0.0, 1.0, 3)
    control = np.array([[x, y, 0.0] for y in xs for x in xs])
    knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    return Patch.build(degrees=(2, 2), knots=(knots, knots), control_points=control)


# --- constraint 1: derivative order is a parameter -------------------------------------


def test_the_provider_returns_only_values_when_asked_for_order_zero() -> None:
    basis = PatchBasis(_bilinear_unit_square())
    out = basis.evaluate(np.array([[0.5, 0.5]]), derivative_order=0)
    assert out.values.shape == (1, 4)
    assert out.gradients is None
    assert out.hessians is None


def test_the_provider_returns_second_derivatives_when_asked_for_order_two() -> None:
    """Isogeometric COLLOCATION needs this; a hard-coded (N, dN) pair would lock it out."""
    basis = PatchBasis(_biquadratic_unit_square())
    out = basis.evaluate(np.array([[0.3, 0.7]]), derivative_order=2)
    assert out.values.shape == (1, 9)
    assert out.gradients.shape == (1, 9, 2)
    assert out.hessians.shape == (1, 9, 2, 2)


def test_an_unsupported_derivative_order_refuses_rather_than_returning_zeros() -> None:
    basis = PatchBasis(_bilinear_unit_square())
    with pytest.raises(SplineBasisError, match='derivative order'):
        basis.evaluate(np.array([[0.5, 0.5]]), derivative_order=3)


# --- constraint 2: points are an argument, decoupled from the quadrature rule -----------


def test_greville_points_are_a_point_set_the_same_call_accepts() -> None:
    """The collocation point set is produced by the provider and consumed by the same entry."""
    basis = PatchBasis(_biquadratic_unit_square())
    points = basis.greville_points()
    assert points.shape == (9, 2)
    out = basis.evaluate(points, derivative_order=2)
    assert out.values.shape == (9, 9)


def test_the_quadrature_rule_is_a_separate_call_from_evaluation() -> None:
    patch = _bilinear_unit_square()
    rule = patch_quadrature(patch)
    assert rule.points.shape[1] == 2
    assert rule.points.shape[0] == rule.weights.shape[0]
    out = PatchBasis(patch).evaluate(rule.points, derivative_order=1)
    assert out.values.shape[0] == rule.points.shape[0]


def test_a_gauss_legendre_rule_integrates_a_polynomial_it_is_exact_for() -> None:
    """N points are exact to degree 2n-1; check the boundary case, not an interior one."""
    for n in (1, 2, 3, 4):
        x, w = gauss_legendre_1d(n, 0.0, 1.0)
        degree = 2 * n - 1
        assert np.isclose(float(w @ x**degree), 1.0 / (degree + 1), atol=1e-13)


# --- constraint 3: geometry degree and field degree are independent ---------------------


def test_a_field_space_may_be_a_different_degree_from_the_geometry_patch() -> None:
    """Sub-/super-parametric. The repo already ships a Q8-geometry, Q4-field kernel."""
    geometry = arc_patch_quarter_annulus(r_inner=1.0, r_outer=2.0)
    basis = PatchBasis(geometry, field_degrees=(1, 1))
    assert basis.geometry_degrees == (2, 1)
    assert basis.field_degrees == (1, 1)
    out = basis.evaluate(np.array([[0.5, 0.5]]), derivative_order=1)
    # The rows are on the SUPPORT and `n_basis` is the patch width; on this single-span field
    # space the two coincide, so both are pinned -- one of them alone would stop saying which.
    assert out.values.shape == (1, 4), 'the FIELD space sets the support width, not the geometry'
    assert out.n_basis == 4, 'the FIELD space sets the basis count, not the geometry'


def test_the_field_space_defaults_to_the_geometry_space_when_unset() -> None:
    basis = PatchBasis(_biquadratic_unit_square())
    assert basis.field_degrees == basis.geometry_degrees


def test_a_field_degree_below_one_is_refused() -> None:
    with pytest.raises(SplineBasisError, match='degree'):
        PatchBasis(_bilinear_unit_square(), field_degrees=(0, 1))


# --- constraint 4: the delivery unit carries its form degree ----------------------------


def test_a_scalar_space_declares_itself_a_zero_form() -> None:
    """H1 is the 0-form of the de Rham complex; the space says so instead of a caller assuming it."""
    basis = PatchBasis(_bilinear_unit_square())
    assert basis.form_degree is FormDegree.ZERO


def test_a_form_degree_this_wave_does_not_build_refuses_by_name() -> None:
    """A refusal that names the missing thing, not a silent fallback to the scalar space."""
    with pytest.raises(SplineBasisError, match='ONE'):
        PatchBasis(_bilinear_unit_square(), form_degree=FormDegree.ONE)


# --- the numerical properties any correct implementation has ----------------------------


def test_the_basis_is_a_partition_of_unity_everywhere_it_is_evaluated() -> None:
    basis = PatchBasis(_biquadratic_unit_square())
    rng = np.random.default_rng(20260913)
    points = rng.random((64, 2))
    out = basis.evaluate(points, derivative_order=1)
    assert np.allclose(out.values.sum(axis=1), 1.0, atol=1e-12)


def test_the_gradients_of_a_partition_of_unity_sum_to_zero() -> None:
    basis = PatchBasis(_biquadratic_unit_square())
    rng = np.random.default_rng(6)
    out = basis.evaluate(rng.random((32, 2)), derivative_order=1)
    assert np.allclose(out.gradients.sum(axis=1), 0.0, atol=1e-11)


def test_the_physical_gradient_reproduces_the_derivative_of_a_linear_field() -> None:
    """``grad`` is in PHYSICAL coordinates: a field equal to x has gradient (1, 0) everywhere."""
    patch = _biquadratic_unit_square()
    basis = PatchBasis(patch)
    coefficients = patch.control_points[:, 0]
    rng = np.random.default_rng(11)
    out = basis.evaluate(rng.random((24, 2)), derivative_order=1)
    grad = np.einsum('qi,qid->qd', np.tile(coefficients, (24, 1)), out.gradients)
    assert np.allclose(grad[:, 0], 1.0, atol=1e-10)
    assert np.allclose(grad[:, 1], 0.0, atol=1e-10)


def test_the_jacobian_determinant_of_the_unit_square_is_one() -> None:
    basis = PatchBasis(_bilinear_unit_square())
    out = basis.evaluate(np.array([[0.25, 0.75], [0.5, 0.5]]), derivative_order=1)
    assert np.allclose(out.det_jacobian, 1.0, atol=1e-12)


def _annulus_area(extra_points: int) -> float:
    """Integrate the quarter annulus with ``extra_points`` beyond the polynomial rule."""
    patch = arc_patch_quarter_annulus(r_inner=1.0, r_outer=2.0)
    rule = patch_quadrature(patch, extra_points=extra_points)
    out = PatchBasis(patch).evaluate(rule.points, derivative_order=1)
    # |det|: the determinant is SIGNED, and this patch's (arc, radial) ordering is left-handed.
    return float((rule.weights * np.abs(out.det_jacobian)).sum())


_EXACT_QUARTER_ANNULUS_AREA = np.pi / 4.0 * (2.0**2 - 1.0**2)


def test_a_rational_patch_is_exact_in_GEOMETRY_but_not_in_QUADRATURE() -> None:
    """The distinction that a careless exactness claim erases.

    The arc is exact to machine precision (asserted in the contract tests). The AREA ELEMENT of a
    rational patch is not a polynomial, so a degree+1 Gauss rule -- exact for the polynomial part --
    does NOT integrate it exactly. Measured 2026-09-13: 9.23e-05 relative at the default rule.
    Claiming machine precision here would be an exactness claim the numbers refute.
    """
    error = abs(_annulus_area(0) - _EXACT_QUARTER_ANNULUS_AREA) / _EXACT_QUARTER_ANNULUS_AREA
    assert 1e-5 < error < 1e-3, f'default-rule relative error moved: {error:.3e}'


def test_refining_only_the_RULE_reaches_machine_precision() -> None:
    """Because the geometry never moved, convergence is the rule's alone -- no remeshing.

    That is the property worth having: a chorded mesh buys accuracy by adding elements and changing
    the domain it integrates over; here the domain is fixed and exact at every step.
    """
    error = abs(_annulus_area(8) - _EXACT_QUARTER_ANNULUS_AREA) / _EXACT_QUARTER_ANNULUS_AREA
    assert error < 1e-14, f'{error:.3e}'


def test_the_error_falls_monotonically_as_the_rule_is_refined() -> None:
    errors = [abs(_annulus_area(n) - _EXACT_QUARTER_ANNULUS_AREA) for n in (0, 1, 2, 4)]
    assert all(b < a for a, b in itertools.pairwise(errors)), errors


def test_the_crudest_chord_is_thousands_of_times_worse_than_the_default_rule() -> None:
    """The comparison that makes the numbers above a claim rather than a coincidence."""
    chord = 0.5 * np.sin(np.pi / 2.0) * (2.0**2 - 1.0**2)
    chord_error = abs(chord - _EXACT_QUARTER_ANNULUS_AREA)
    patch_error = abs(_annulus_area(0) - _EXACT_QUARTER_ANNULUS_AREA)
    assert patch_error * 1000 < chord_error, f'patch {patch_error:.3e} vs chord {chord_error:.3e}'


def test_the_quadrature_point_count_follows_the_degree() -> None:
    """p+1 points per direction per non-empty knot span -- exact for the polynomial part."""
    assert patch_quadrature(_bilinear_unit_square()).points.shape[0] == 2 * 2
    assert patch_quadrature(_biquadratic_unit_square()).points.shape[0] == 3 * 3


def test_a_two_span_patch_gets_a_rule_per_span() -> None:
    knots = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
    xs = np.linspace(0.0, 1.0, 3)
    control = np.array([[x, y, 0.0] for y in (0.0, 1.0) for x in xs])
    patch = Patch.build(
        degrees=(1, 1),
        knots=(knots, np.array([0.0, 0.0, 1.0, 1.0])),
        control_points=control,
    )
    assert patch_quadrature(patch).points.shape[0] == (2 * 2) * 2


def test_the_local_to_global_map_is_the_identity_on_a_single_patch_space() -> None:
    basis = PatchBasis(_biquadratic_unit_square())
    assert basis.dof_ids().tolist() == list(range(9))


def test_an_offset_shifts_every_dof_id_by_the_same_amount() -> None:
    basis = PatchBasis(_biquadratic_unit_square(), dof_offset=100)
    assert basis.dof_ids().tolist() == list(range(100, 109))


# --- the SECOND derivative, and the defect that shipped in it ---------------------------------


def test_the_laplacian_of_a_COORDINATE_is_zero_on_a_curved_rational_patch() -> None:
    """The closed form that caught a real defect, and the sharpest one available here.

    Feeding a patch its own control points' x-coordinates as field coefficients reproduces
    ``u = x`` EXACTLY -- that is the definition of the geometry map, not an approximation of it.
    ``x`` is harmonic, so ``laplacian(u)`` must be zero, and on a rational annular patch that is a
    demanding check because the map is nowhere affine.

    MEASURED 2026-09-14: the corrected pullback reaches **2.3e-13**, against **18.2** from the one
    that shipped until then. It dropped the
    geometry map's own second-derivative term from the chain rule and rationalised the second
    derivative by a bare weight multiply instead of the quotient rule. Its docstring named the
    first approximation and argued it was safe because "the one consumer that needs second
    derivatives -- collocation -- is not built in this wave, so the approximation has no caller
    yet and must not acquire one silently". It acquired one -- the a-posteriori error indicator --
    and the indicator ranked the error BACKWARDS (Spearman -0.46) as a direct result.
    """
    patch = refine_space(annular_sector_patches(radii=(0.02, 0.05), group_ids=(0,), span_deg=90.0), (2, 2)).patches[0]
    grid = np.stack(
        np.meshgrid(np.linspace(0.05, 0.95, 9), np.linspace(0.05, 0.95, 9), indexing='ij'), axis=-1
    ).reshape(-1, 2)
    values = PatchBasis(patch).evaluate(grid, derivative_order=2)

    for component in (0, 1):
        # GATHERED onto each point's support, which is the stronger statement: the rows are local,
        # so a basis function scattered to the wrong column gives the wrong coefficient here and
        # the coordinate stops being reproduced. A dense accessor would hide that.
        coefficients = patch.control_points[values.columns, component]
        laplacian = np.einsum('qldd,ql->q', values.hessians, coefficients)
        assert np.abs(laplacian).max() < 1e-10, component


def test_the_gradient_of_a_COORDINATE_is_the_unit_vector_along_it() -> None:
    """The first-derivative twin of the check above -- it was already right, and stays right."""
    patch = refine_space(annular_sector_patches(radii=(0.02, 0.05), group_ids=(0,), span_deg=90.0), (2, 2)).patches[0]
    grid = np.stack(
        np.meshgrid(np.linspace(0.05, 0.95, 9), np.linspace(0.05, 0.95, 9), indexing='ij'), axis=-1
    ).reshape(-1, 2)
    values = PatchBasis(patch).evaluate(grid, derivative_order=2)

    for component in (0, 1):
        gradient = np.einsum('qld,ql->qd', values.gradients, patch.control_points[values.columns, component])
        expected = np.zeros(2)
        expected[component] = 1.0
        assert np.abs(gradient - expected).max() < 1e-12, component


def test_the_hessian_is_SYMMETRIC_in_its_two_derivative_axes() -> None:
    """Mixed partials commute; an asymmetry is an index slip in the pullback."""
    patch = refine_space(annular_sector_patches(radii=(0.02, 0.05), group_ids=(0,), span_deg=90.0), (1, 1)).patches[0]
    grid = np.array([[0.3, 0.4], [0.7, 0.2], [0.5, 0.9]])
    hessians = PatchBasis(patch).evaluate(grid, derivative_order=2).hessians
    assert np.abs(hessians - np.swapaxes(hessians, 2, 3)).max() < 1e-9 * np.abs(hessians).max()
