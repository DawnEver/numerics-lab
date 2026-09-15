"""Spline basis + quadrature over a patch -- the physics-free evaluation seam (IGA deliverable 2).

THE PIECE THAT MAKES ISOGEOMETRIC A CAPABILITY RATHER THAN ONE BACKEND'S FEATURE. Every spatial
backend in this repo inlines its own element formulas, so a spline basis added inside one of them
would have to be added inside each of the others too. An integrand is
``(constitutive operator) x (basis values and derivatives at quadrature points)``; the first factor
is already separated per physics, and this module is the second factor, owned once.

It lives in ``gauss`` because ``gauss`` is the physics-free numerical runtime, and nothing here
knows what is being solved: it maps a patch and a set of parametric points to values, physical
derivatives and a Jacobian determinant.

WHAT IT DOES NOT DO, and this is load-bearing: it does not touch any existing kernel. The triangle,
quad and polygon assemblies keep their own formulas untouched, because their goldens are pinned
bit-for-bit and an interface-uniformity rewrite would put that at risk for no numerical gain. This
is an ADDITION beside them.

Four interface constraints, each forced by a consumer that would otherwise be locked out
-----------------------------------------------------------------------------------------

1. **Derivative order is a PARAMETER.** Collocation evaluates SECOND derivatives at Greville
   points; a Galerkin assembly wants first derivatives at Gauss points. A fixed ``(N, dN)`` return
   serves one and excludes the other permanently.
2. **Evaluation points are an ARGUMENT, decoupled from the quadrature rule.** Greville points,
   Gauss points, plot sampling and read-back at another discretization's sample points are four
   different point sets over one basis, so :func:`patch_quadrature` is a separate call rather than
   a mode of :meth:`PatchBasis.evaluate`.
3. **Geometry degree and field degree are INDEPENDENT.** The repo already ships a superparametric
   kernel -- an eight-node geometry map carrying an unchanged bilinear field -- so welding the two
   degrees together here would contradict a kernel in the tree. ``field_degrees`` therefore
   defaults to the geometry's and may differ.
4. **The delivery unit carries its FORM DEGREE.** A curl-conforming space is not a scalar space
   with extra components; it is the 1-form of a de Rham complex and comes with the incidence
   relations to its neighbours. This wave builds the 0-form only, and
   :class:`FormDegree` is how the space SAYS so -- a caller asking for a form this module does not
   build gets a refusal naming it, never a silent scalar substitute.

Cost note, so nobody reads the loop below as a claim about speed: the basis evaluation is the dense
Cox-de Boor recursion over the whole knot vector, not a per-span local-support window. That is
correct and wasteful, and it is the shape a Bezier-extraction rewrite replaces once a real assembly
is measuring. Nothing here should be quoted as the cost of isogeometric assembly.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from motronics.contracts.errors import MotronicsError
from motronics.contracts.ir.spline_space_ir import Patch

__all__ = [
    'EXTRA_QUADRATURE_POINTS',
    'BasisValues',
    'FormDegree',
    'PatchBasis',
    'QuadratureRule',
    'SplineBasisError',
    'gauss_legendre_1d',
    'patch_quadrature',
]

#: The highest derivative order this wave evaluates. Two is not a placeholder: it is what
#: collocation needs, and the recursion below produces exactly that.
MAX_DERIVATIVE_ORDER = 2

#: How far below the LARGEST Jacobian determinant on a rule a determinant may fall before the
#: geometry map is called singular. Relative, so it means the same thing in metres and
#: millimetres -- an absolute floor on a determinant is a tolerance with a hidden unit.
_SINGULAR_JACOBIAN_RATIO = 1e-12


class SplineBasisError(
    MotronicsError,
    ValueError,
    error_code='MTS-SPLINE-007',
    user_action=(
        'The geometry map is singular or the request is malformed. A singular map means the patch folds or '
        'collapses, so fix the producer input that built it rather than the quadrature; the message names the '
        'point and the determinant it measured. '
    ),
):
    """A basis or quadrature request this provider refuses rather than approximating."""


class FormDegree(Enum):
    """Which de Rham form a spline space represents.

    Declared rather than assumed so the scalar space cannot be silently substituted for one of the
    others. ``ZERO`` is the H1-conforming scalar space (a magnetic vector potential in plane, a
    temperature, a displacement component); ``ONE`` is curl-conforming, ``TWO`` div-conforming and
    ``THREE`` the fully discontinuous end of the complex.

    Only ``ZERO`` is built in this wave. The others exist as NAMES so a request for one refuses by
    name -- a space that answered a curl-conforming request with a scalar basis would be the
    declaration-that-lies shape, and an enum with one member could not refuse at all.
    """

    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3


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


@dataclass(frozen=True)
class BasisValues:
    """What an evaluation returns: values, and derivatives to the order that was asked for.

    Attributes:
        values: ``(n_points, n_basis)`` basis values.
        gradients: ``(n_points, n_basis, dim)`` PHYSICAL first derivatives, or ``None`` when order
            0 was requested. Physical, not parametric: a caller assembling a stiffness wants
            ``d/dx``, and doing the chain rule at every call site is how two call sites end up
            disagreeing about it.
        hessians: ``(n_points, n_basis, dim, dim)`` physical second derivatives, or ``None`` below
            order 2.
        det_jacobian: ``(n_points,)`` determinant of the geometry map's Jacobian, the factor that
            turns a parametric weight into a physical one.
        jacobian: ``(n_points, 3, dim)`` the geometry Jacobian itself, for a caller that needs the
            map rather than only its determinant (a normal, a pull-back).

    """

    values: NDArray[np.float64]
    gradients: NDArray[np.float64] | None
    hessians: NDArray[np.float64] | None
    det_jacobian: NDArray[np.float64]
    jacobian: NDArray[np.float64]


class PatchBasis:
    """The evaluation seam for ONE patch: a geometry map plus a (possibly different) field space.

    Args:
        patch: the geometry patch. Its degrees and knots define the MAP.
        field_degrees: the field space's degrees, defaulting to the geometry's. Differing from the
            geometry is the sub-/super-parametric case and is deliberate, not an error.
        form_degree: which de Rham form the field space is. Only :attr:`FormDegree.ZERO` is built.
        dof_offset: added to every local dof id, so a multipatch assembly can number patches
            consecutively without this class knowing about the space that owns it.

    Raises:
        SplineBasisError: on a field degree below 1, or a form degree this wave does not build.

    """

    def __init__(
        self,
        patch: Patch,
        *,
        field_degrees: tuple[int, ...] | None = None,
        form_degree: FormDegree = FormDegree.ZERO,
        dof_offset: int = 0,
    ) -> None:
        if form_degree is not FormDegree.ZERO:
            msg = (
                f'form degree {form_degree.name} is not built in this wave -- only ZERO (the '
                f'H1-conforming scalar space) is. A curl- or div-conforming space is a different '
                f'basis with its own incidence relations, not this one with extra components.'
            )
            raise SplineBasisError(msg)
        self.patch = patch
        self.form_degree = form_degree
        self.dof_offset = int(dof_offset)
        # Both are properties of the MAP at the points of the last evaluation, cached together
        # because a hessian pulled back through a stale inverse Jacobian is silently wrong.
        self._inverse_jacobian: NDArray[np.float64] | None = None
        self._geometry_curvature: NDArray[np.float64] | None = None
        self.geometry_degrees = tuple(patch.degrees)
        if field_degrees is None:
            self.field_degrees = self.geometry_degrees
            self._field_knots = tuple(patch.knots)
        else:
            self.field_degrees = tuple(int(p) for p in field_degrees)
            if len(self.field_degrees) != len(self.geometry_degrees):
                msg = (
                    f'field_degrees has {len(self.field_degrees)} directions but the patch has '
                    f'{len(self.geometry_degrees)}'
                )
                raise SplineBasisError(msg)
            for direction, degree in enumerate(self.field_degrees):
                if degree < 1:
                    msg = f'field degree {degree} in direction {direction} is below 1'
                    raise SplineBasisError(msg)
            self._field_knots = tuple(
                _open_knot_vector(knot, source_degree, degree)
                for knot, source_degree, degree in zip(patch.knots, patch.degrees, self.field_degrees, strict=True)
            )

    @property
    def parametric_dim(self) -> int:
        """How many parametric directions the patch has."""
        return self.patch.parametric_dim

    @property
    def n_basis(self) -> int:
        """How many FIELD basis functions the patch carries."""
        return int(np.prod([len(k) - p - 1 for k, p in zip(self._field_knots, self.field_degrees, strict=True)]))

    def dof_ids(self) -> NDArray[np.intp]:
        """The global dof ids of this patch's field basis, offset by ``dof_offset``."""
        return np.arange(self.dof_offset, self.dof_offset + self.n_basis, dtype=np.intp)

    def greville_points(self) -> NDArray[np.float64]:
        """The Greville abscissae of the FIELD space -- the collocation point set.

        One point per basis function, the average of the ``degree`` interior knots that support it.
        Produced here rather than by a caller so the collocation consumer and the basis cannot
        disagree about which points they are.
        """
        per_direction = [
            _greville_1d(knot, degree) for knot, degree in zip(self._field_knots, self.field_degrees, strict=True)
        ]
        grids = np.meshgrid(*per_direction, indexing='ij')
        # The FIRST direction varies fastest, matching the control-point ordering.
        return np.stack([g.ravel(order='F') for g in grids], axis=1)

    def evaluate(self, points: NDArray[np.float64], *, derivative_order: int = 1) -> BasisValues:
        """Evaluate the field basis and the geometry map at parametric ``points``.

        Args:
            points: ``(n_points, parametric_dim)`` parameters.
            derivative_order: 0, 1 or 2. Higher orders RAISE rather than returning zeros --
                a zero array is indistinguishable from a genuinely vanishing derivative.

        Raises:
            SplineBasisError: on an unsupported derivative order, a point array of the wrong
                width, or a singular geometry map at a requested point.

        """
        if not 0 <= derivative_order <= MAX_DERIVATIVE_ORDER:
            msg = (
                f'derivative order {derivative_order} is not supported; this provider evaluates '
                f'0..{MAX_DERIVATIVE_ORDER}'
            )
            raise SplineBasisError(msg)
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != self.parametric_dim:
            msg = f'points has shape {points.shape}; expected (n, {self.parametric_dim})'
            raise SplineBasisError(msg)
        if points.shape[0] == 0:
            # The shape check above accepts (0, dim), and everything downstream then either
            # reshapes a zero-size array or reduces over an empty axis -- the first produces a bare
            # numpy ValueError and the second makes the geometry guards pass VACUOUSLY.
            msg = 'no evaluation points were given, so there is nothing to evaluate the basis at'
            raise SplineBasisError(msg)
        # The DOMAIN check, matching Patch.evaluate's. Without it an out-of-domain parameter makes
        # _cox_de_boor return an all-zero row, _rationalise_values divides 0/0, and the NaN flows
        # into the stiffness matrix -- past the singular-map guard, which np.abs(nan) < tol does not
        # trip. Every consumer then assembles NaN with no error raised anywhere.
        for axis, knot in enumerate(self.patch.knots):
            lo, hi = float(knot[0]), float(knot[-1])
            outside = (points[:, axis] < lo) | (points[:, axis] > hi)
            if np.any(outside):
                bad = float(points[outside, axis][0])
                msg = (
                    f'parameter {bad} on axis {axis} is outside the patch domain [{lo}, {hi}]; '
                    'evaluating there would return a zero basis row and assemble NaN.'
                )
                raise SplineBasisError(msg)

        geom = _tensor_basis(self.patch.knots, self.geometry_degrees, points, order=max(derivative_order, 1))
        # Degrees decide it: the field knot vectors are DERIVED from the geometry's breakpoints,
        # so equal degrees means the same space. Comparing the knot arrays themselves would be an
        # elementwise numpy comparison in a boolean context, which raises rather than answering.
        field = (
            geom
            if self.field_degrees == self.geometry_degrees
            else _tensor_basis(self._field_knots, self.field_degrees, points, order=derivative_order)
        )

        jac, det = self._geometry_jacobian(geom, points.shape[0])
        values = _rationalise_values(field.values, self._field_weights(field.values.shape[1]))
        gradients = hessians = None
        if derivative_order >= 1:
            gradients = self._physical_gradients(field, jac)
        if derivative_order >= MAX_DERIVATIVE_ORDER:
            hessians = self._physical_hessians(field, jac)
        return BasisValues(values=values, gradients=gradients, hessians=hessians, det_jacobian=det, jacobian=jac)

    # --- internals -------------------------------------------------------------------

    def _field_weights(self, n_basis: int) -> NDArray[np.float64]:
        """The weights the FIELD space carries: the patch's own when the spaces coincide.

        A field space of a different degree has no weights of its own -- it is a polynomial space
        on the same parameter domain -- so it is non-rational by construction. Saying that here is
        why the sub-parametric case does not silently inherit a rational denominator that belongs
        to the geometry.
        """
        if n_basis == self.patch.n_control_points and self.field_degrees == self.geometry_degrees:
            return self.patch.weights
        return np.ones(n_basis)

    def _geometry_jacobian(self, geom: _TensorBasis, n_points: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``dX/dq`` and its determinant, from the RATIONAL geometry map."""
        weights = self.patch.weights
        weighted = geom.values * weights[None, :]
        denominator = weighted.sum(axis=1)
        rational = weighted / denominator[:, None]

        d_weighted = geom.gradients * weights[None, :, None]
        d_denominator = d_weighted.sum(axis=1)
        d_rational = (d_weighted - rational[:, :, None] * d_denominator[:, None, :]) / denominator[:, None, None]

        jac = np.einsum('qid,ic->qcd', d_rational, self.patch.control_points)
        square = jac[:, : self.parametric_dim, :]
        det = np.linalg.det(square)
        # RELATIVE, because a determinant carries units: dX/dq on a 50 mm patch in metres is ~1e-3
        # and in millimetres ~1e3, so a bare 1e-14 floor means a different thing in each and is a
        # tolerance nobody can state. Scaling by the largest determinant on the rule makes the test
        # the same statement in any unit. `not np.all(... > ...)` rather than `np.any(... < ...)`
        # so a NaN -- which compares False both ways -- FAILS the guard instead of passing it.
        # det is never empty here: evaluate() refuses an empty point array before reaching this.
        scale = float(np.max(np.abs(det)))
        if not np.all(np.abs(det) > _SINGULAR_JACOBIAN_RATIO * scale):
            msg = (
                'the geometry map is singular (or non-finite) at a requested point: the Jacobian '
                f'determinant falls to {float(np.nanmin(np.abs(det))) if det.size else 0.0} against '
                f'a largest of {scale} on this rule.'
            )
            raise SplineBasisError(msg)
        # SIGN CONSISTENCY. |det| alone is blind to a FOLDED patch: a legal reflex control net gives
        # determinants of mixed sign, the parametrization inverts over part of the patch, and the
        # signed area then under-reports the covered one (measured 0.200 against a true 0.331) with
        # no exception anywhere. A patch that turns itself inside out is not a patch.
        if float(det.min()) * float(det.max()) <= 0.0:
            msg = (
                'the geometry map FOLDS: the Jacobian determinant changes sign across the patch '
                f'(from {float(det.min())} to {float(det.max())}), so the parametrization is '
                'inverted over part of it and any integral over it is wrong.'
            )
            raise SplineBasisError(msg)
        self._inverse_jacobian = np.linalg.inv(square)
        self._geometry_curvature = (
            np.einsum('qide,ic->qcde', _rational_hessians(geom, weights), self.patch.control_points)
            if geom.hessians is not None
            else None
        )
        return jac, det

    def _physical_gradients(self, field: _TensorBasis, jac: NDArray[np.float64]) -> NDArray[np.float64]:
        """Chain-rule the parametric derivatives onto physical coordinates."""
        del jac
        rational = _rationalise_gradients(field, self._field_weights(field.values.shape[1]))
        return np.einsum('qid,qde->qie', rational, self._inverse_jacobian)

    def _physical_hessians(self, field: _TensorBasis, jac: NDArray[np.float64]) -> NDArray[np.float64]:
        """Second derivatives on PHYSICAL coordinates, with the map's own curvature included.

        ``H_phys = J^-T (H_param - grad . H_F) J^-1``, where ``H_F`` is the second derivative of the
        geometry map. The chain rule for a second derivative has two terms and the map-curvature one
        is not optional off an affine map.

        THIS WAS WRONG UNTIL 2026-09-14 AND THE DOCSTRING SAID SO. The previous version dropped the
        ``grad . H_F`` term and rationalised by a bare weight multiply, and it stated that the
        approximation was safe because "the one consumer that needs second derivatives --
        collocation -- is not built in this wave, so the approximation has no caller yet and must
        not acquire one silently". It acquired one: the a-posteriori error indicator. On the annular
        patches this package builds, the dropped term IS the entire curvature of the polar map, and
        with a degree-(2, 1) patch the radial parametric second derivative vanishes identically --
        so the "Laplacian" reduced to roughly ``A_tt / r^2`` and measured where the solution curves
        in theta rather than where the error was.

        MEASURED CONSEQUENCE, against a four-times-refined reference on a 1000x contrast annulus:
        the indicator built on the old pullback ranked the error at Spearman **-0.46** (top-quartile
        overlap 0/12). With this pullback it ranks at **+0.92**, and +0.999 once the indicator's own
        length scale is also corrected. The estimator's global norm also stops plateauing and falls
        as O(h) -- the plateau WAS this dropped term saturating.
        """
        del jac
        if field.hessians is None:
            msg = 'second derivatives were requested but the field basis did not produce them'
            raise SplineBasisError(msg)
        weights = self._field_weights(field.values.shape[1])
        parametric = _rational_hessians(field, weights)
        gradients = self._physical_gradients(field, None)
        curvature = self._geometry_hessians()
        inv = self._inverse_jacobian
        dim = self.parametric_dim
        corrected = parametric - np.einsum('qic,qcde->qide', gradients, curvature[:, :dim])
        return np.einsum('qide,qdf,qeg->qifg', corrected, inv, inv)

    def _geometry_hessians(self) -> NDArray[np.float64]:
        """``d2X/dq dq`` of the RATIONAL geometry map -- the term the chain rule needs.

        Cached per evaluation beside the inverse Jacobian, because both are properties of the map at
        the same points and recomputing one without the other is how they come to disagree.
        """
        if self._geometry_curvature is None:
            msg = 'the geometry hessian was requested before the map was evaluated at these points'
            raise SplineBasisError(msg)
        return self._geometry_curvature


class _TensorBasis(NamedTuple):
    values: NDArray[np.float64]
    gradients: NDArray[np.float64] | None
    hessians: NDArray[np.float64] | None


def _tensor_basis(
    knots: tuple[NDArray[np.float64], ...],
    degrees: tuple[int, ...],
    points: NDArray[np.float64],
    *,
    order: int,
) -> _TensorBasis:
    """Tensor-product basis values and parametric derivatives, first direction varying fastest."""
    per_direction = [
        _bspline_1d(knot, degree, points[:, d], order=order)
        for d, (knot, degree) in enumerate(zip(knots, degrees, strict=True))
    ]
    dim = len(degrees)
    values = per_direction[0][0]
    for d in range(1, dim):
        values = _outer_fastest_first(values, per_direction[d][0])

    gradients = None
    if order >= 1:
        columns = []
        for direction in range(dim):
            factors = [per_direction[d][1 if d == direction else 0] for d in range(dim)]
            block = factors[0]
            for d in range(1, dim):
                block = _outer_fastest_first(block, factors[d])
            columns.append(block)
        gradients = np.stack(columns, axis=2)

    hessians = None
    if order >= MAX_DERIVATIVE_ORDER:
        blocks = np.empty((points.shape[0], values.shape[1], dim, dim))
        for a in range(dim):
            for b in range(dim):
                factors = []
                for d in range(dim):
                    order_d = (1 if d == a else 0) + (1 if d == b else 0)
                    factors.append(per_direction[d][order_d])
                block = factors[0]
                for d in range(1, dim):
                    block = _outer_fastest_first(block, factors[d])
                blocks[:, :, a, b] = block
        hessians = blocks
    return _TensorBasis(values=values, gradients=gradients, hessians=hessians)


def _outer_fastest_first(accumulated: NDArray[np.float64], nxt: NDArray[np.float64]) -> NDArray[np.float64]:
    """Tensor two per-direction bases so the ALREADY-ACCUMULATED index varies fastest."""
    return (accumulated[:, None, :] * nxt[:, :, None]).reshape(accumulated.shape[0], -1)


def _bspline_1d(
    knot: NDArray[np.float64], degree: int, values: NDArray[np.float64], *, order: int
) -> list[NDArray[np.float64]]:
    """``[N, dN, d2N]`` up to ``order`` for one knot vector, by the Cox-de Boor recursion."""
    out = [_cox_de_boor(knot, degree, values)]
    out.extend(_bspline_derivative(knot, degree, values, k) for k in range(1, order + 1))
    return out


def _cox_de_boor(knot: NDArray[np.float64], degree: int, values: NDArray[np.float64]) -> NDArray[np.float64]:
    """All B-spline basis functions of ``degree`` on ``knot`` at ``values``, dense."""
    n_basis = len(knot) - degree - 1
    values = np.asarray(values, dtype=float)
    current = np.zeros((values.size, len(knot) - 1))
    for i in range(len(knot) - 1):
        inside = (values >= knot[i]) & (values < knot[i + 1])
        if knot[i + 1] == knot[-1] and knot[i] < knot[i + 1]:
            inside |= values == knot[-1]
        current[inside, i] = 1.0
    for p in range(1, degree + 1):
        width = len(knot) - p - 1
        nxt = np.zeros((values.size, width))
        for i in range(width):
            left_den = knot[i + p] - knot[i]
            right_den = knot[i + p + 1] - knot[i + 1]
            if left_den > 0:
                nxt[:, i] += (values - knot[i]) / left_den * current[:, i]
            if right_den > 0:
                nxt[:, i] += (knot[i + p + 1] - values) / right_den * current[:, i + 1]
        current = nxt
    return current[:, :n_basis]


def _bspline_derivative(
    knot: NDArray[np.float64], degree: int, values: NDArray[np.float64], order: int
) -> NDArray[np.float64]:
    """The ``order``-th derivative of every basis function, by the standard degree-drop identity.

    ``dN_{i,p} = p * (N_{i,p-1} / (t_{i+p} - t_i) - N_{i+1,p-1} / (t_{i+p+1} - t_{i+1}))``, applied
    ``order`` times. A degree that drops below zero gives an identically zero derivative, which is
    the true answer rather than a guard.
    """
    n_basis = len(knot) - degree - 1
    if order > degree:
        return np.zeros((np.asarray(values).size, n_basis))
    lower = _bspline_derivative(knot, degree - 1, values, order - 1) if order > 1 else None
    if order == 1:
        lower = _cox_de_boor(knot, degree - 1, values)
    out = np.zeros((np.asarray(values).size, n_basis))
    for i in range(n_basis):
        left_den = knot[i + degree] - knot[i]
        right_den = knot[i + degree + 1] - knot[i + 1]
        if left_den > 0:
            out[:, i] += degree * lower[:, i] / left_den
        if right_den > 0:
            out[:, i] -= degree * lower[:, i + 1] / right_den
    return out


def _rationalise_values(values: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """``R_i = w_i N_i / sum_j w_j N_j`` -- the identity when every weight is one."""
    weighted = values * weights[None, :]
    return weighted / weighted.sum(axis=1, keepdims=True)


def _rationalise_gradients(field: _TensorBasis, weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Quotient-rule the rational basis's PARAMETRIC first derivatives."""
    if field.gradients is None:
        msg = 'gradients were requested but the field basis did not produce them'
        raise SplineBasisError(msg)
    weighted = field.values * weights[None, :]
    denominator = weighted.sum(axis=1)
    rational = weighted / denominator[:, None]
    d_weighted = field.gradients * weights[None, :, None]
    d_denominator = d_weighted.sum(axis=1)
    return (d_weighted - rational[:, :, None] * d_denominator[:, None, :]) / denominator[:, None, None]


def _rational_hessians(field: _TensorBasis, weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Quotient-rule the rational basis's PARAMETRIC second derivatives.

    ``R,de = (w N,de - R,d W,e - R,e W,d - R W,de) / W``. All three correction terms are needed; a
    bare ``w * N,de`` is the first derivative's formula applied twice and is only right when every
    weight is equal, which is exactly the case an annular patch is not.
    """
    if field.hessians is None or field.gradients is None:
        msg = 'second derivatives were requested but the field basis did not produce them'
        raise SplineBasisError(msg)
    weighted = field.values * weights[None, :]
    denominator = weighted.sum(axis=1)
    rational = weighted / denominator[:, None]

    d_weighted = field.gradients * weights[None, :, None]
    d_denominator = d_weighted.sum(axis=1)
    d_rational = (d_weighted - rational[:, :, None] * d_denominator[:, None, :]) / denominator[:, None, None]

    dd_weighted = field.hessians * weights[None, :, None, None]
    dd_denominator = dd_weighted.sum(axis=1)
    correction = (
        d_rational[:, :, :, None] * d_denominator[:, None, None, :]
        + d_rational[:, :, None, :] * d_denominator[:, None, :, None]
        + rational[:, :, None, None] * dd_denominator[:, None, :, :]
    )
    return (dd_weighted - correction) / denominator[:, None, None, None]


def _open_knot_vector(source: NDArray[np.float64], source_degree: int, degree: int) -> NDArray[np.float64]:
    """An open knot vector of ``degree`` over the same interior breakpoints as ``source``.

    Used only when a caller asks for a field degree that differs from the geometry's: the parameter
    domain and its spans are the geometry's, the multiplicity at the ends is the new degree's.

    INTERIOR MULTIPLICITY IS PRESERVED, scaled to the new degree, and that is the whole subtlety.
    ``np.unique`` collapsed repeats, so geometry knots ``[0,0,0,0.5,0.5,1,1,1]`` (degree 2 -- a
    deliberate C^0 crease, which is exactly how a corner or a material line is modelled) became
    field knots ``[0,0,0,0.5,1,1,1]``, i.e. C^1. Asking for a different field degree therefore
    SMOOTHED the field space across the crease, and a smooth field across a material line is a
    plausible wrong answer rather than a failure.

    A knot of multiplicity ``m`` in a degree-``p`` vector carries continuity ``C^(p-m)``; holding
    that continuity at the new degree needs multiplicity ``m + (degree - source_degree)``, clamped
    so it never exceeds the new degree (which would split the patch).
    """
    lo, hi = float(source[0]), float(source[-1])
    inner = source[(source > lo) & (source < hi)]
    pieces = [np.full(degree + 1, lo)]
    if inner.size:
        values, counts = np.unique(inner, return_counts=True)
        shift = degree - source_degree
        for value, count in zip(values, counts, strict=True):
            pieces.append(np.full(min(max(int(count) + shift, 1), degree), float(value)))
    pieces.append(np.full(degree + 1, hi))
    return np.concatenate(pieces)


def _greville_1d(knot: NDArray[np.float64], degree: int) -> NDArray[np.float64]:
    """Greville abscissae: the moving average of ``degree`` interior knots per basis function."""
    n_basis = len(knot) - degree - 1
    return np.array([knot[i + 1 : i + degree + 1].mean() for i in range(n_basis)])


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
