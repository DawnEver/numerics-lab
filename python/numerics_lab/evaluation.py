"""Unified ``FieldEvaluator`` / ``AssemblyEquation`` protocol pair (plan-multi-layer-decoupling P3).

P1 extracted the per-branch (MEC, ``_ReluctanceSystem._eval_nu_branch``) and per-element (FEA,
``eval_nu_element``) nu lookups into named methods; P2 wired an opt-in seam so a caller-supplied
evaluator could replace either lookup, but the seam was ad hoc -- MEC's ``field_evaluator`` was
duck-typed on an ``evaluate_branch(b_abs, system)`` method, FEA's on ``ElementFieldEvaluator``'s
``evaluate_element(...)`` (a per-scalar-element signature the P4 ``PerEdgeFromElement`` evaluator
did not even implement, forcing a second ``getattr(..., 'evaluate_edges', None)`` duck-type at the
call site). This module is where those seams get a shared home: one ``FieldEvaluator`` Protocol
(``granularity`` names what the evaluator is keyed on: ``'branch'``, ``'element'``, ``'edge'``) and
one ``AssemblyEquation`` Protocol (the narrowed :class:`~motronics.gauss.nonlinear.NonlinearSystem`
seam an equation implements once it has a field-dependent :class:`CoefficientState` in hand).

Two things this module deliberately does NOT do, because the codebase's own granularities are not
uniform (see the plan's evaluator table -- MEC's branch evaluators consume a per-branch ``|B|``
array, FEA's element evaluators a single per-element scalar, the P4 edge evaluator a per-element
``(3, 2)`` edge-vector block): it does not force one calling CONVENTION on every evaluator (that
would mean rewriting the T3/Q4/VEM per-element kernels' hot loops purely for interface uniformity,
risking the bit-identity every golden here depends on -- AGENTS.md's "implement, don't exclude"
cuts against a refactor whose only benefit is symmetry), and it does not replace ``tangent()``,
which stays on the equation itself (MEC and FEA compute their Newton differential term directly
from the material's reluctivity SLOPE, never through an evaluator -- P2 never routed a tangent
through this seam either, so there is nothing to supersede).

``_ReluctanceSystem`` (MEC) and ``GalerkinSystem`` (FEA) COMPOSE this pair: each names its own
``_extract_field`` hook (MEC: reduced loop flux -> per-branch ``|B|``; FEA: per-element/per-block
nodal potentials -> ``curl(A)``, inline in each valence kernel because T3/Q4/VEM have genuinely
different field formulas) and holds a ``FieldEvaluator`` instance -- never ``None`` after
construction, so ``assemble()`` composes it unconditionally instead of branching on
``field_evaluator is not None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from motronics.gauss.nonlinear import AssembledState

__all__ = ['AssemblyEquation', 'CoefficientState', 'FieldEvaluator']


@dataclass
class CoefficientState:
    """Field-dependent material coefficients a :class:`FieldEvaluator` hands to its equation.

    Attributes:
        nu: the reluctivity actually used to scale the stiffness -- possibly under-relaxed
            (FEA's ``nu_used = (1-alpha)*nu_prev + alpha*nu_target``); MEC never damps this value.
        dnu_db2: ``d(nu)/d(B^2)`` where a Newton tangent needs it, else ``None``. Not consumed by
            either equation TODAY (see module docstring) -- carried here so a future evaluator that
            wants to drive ``tangent()`` through this seam has somewhere to put it, without a second
            protocol change.
        energy_density: the per-unit-measure magnetic energy, ``nu_target * B^2 / 2``, ALWAYS from
            the true (undamped) reluctivity at the current field -- never the damped ``nu`` above.
            Measure-free at this boundary (chart owns the measure: the equation multiplies by its
            own geometric factor -- MEC's ``l_mat * A_mat``, FEA's ``delta * depth`` -- to get an
            absolute energy, per AGENTS.md's "the chart supplies a continuous measure, it does not
            supply a fictitious physical length").

    """

    nu: np.ndarray | float
    dnu_db2: np.ndarray | float | None
    energy_density: np.ndarray | float


@runtime_checkable
class FieldEvaluator(Protocol):
    """Granularity-labelled seam mapping an already-extracted field to material coefficients.

    ``granularity`` names the discretization unit the evaluator is keyed on (``'branch'``,
    ``'element'``, ``'edge'``) so a caller can tell what an unfamiliar evaluator produces without
    inspecting its class -- the same role :mod:`~motronics.hamilton.discretizations.reduced_order`
    documentation gives ``chart.kind``: a property the code CONSULTS, not a name that merely says so.

    This Protocol intentionally carries NO ``evaluate`` method: the field-evaluation calling
    convention differs by granularity (MEC's branch evaluators take a per-branch ``|B|`` array and
    the owning system as geometry context; FEA's element evaluator takes one scalar ``|B|`` per
    call inside the T3/Q4/VEM assembly loop; the edge evaluator takes one element's ``(3, 2)`` edge
    vectors) -- see the concrete evaluators in
    :mod:`~motronics.hamilton.formulations.magnetic_circuit._sparse` (``PerBranch``,
    ``PerElementFromBranches``) and :mod:`~motronics.hamilton.discretizations.fem._nu_element`
    (``PerElement``, ``PerEdgeFromElement``) for the granularity-specific method each implements.
    Forcing one signature here would mean rewriting the per-element hot loops purely for interface
    symmetry, trading the bit-identity every golden depends on for a cosmetic uniformity this
    Protocol does not need in order to do its job: naming what an evaluator IS.
    """

    granularity: str


class AssemblyEquation(Protocol):
    """The narrowed :class:`~motronics.gauss.nonlinear.NonlinearSystem` seam an equation composes.

    An equation (``_ReluctanceSystem``, ``GalerkinSystem``) implements this AND
    :class:`~motronics.gauss.nonlinear.NonlinearSystem` in full (the iteration strategies still
    drive it through that wider Protocol); this narrower one is the shape ``assemble`` actually
    fills FROM a :class:`CoefficientState` once ``_extract_field`` and the held
    :class:`FieldEvaluator` have produced one -- i.e. the composition
    ``equation.assemble(x)`` internally does
    ``coefficients = equation.evaluator.evaluate_*(equation._extract_field(x), ...)`` before
    building the returned :class:`~motronics.gauss.nonlinear.AssembledState`.
    """

    is_linear: bool

    def initial_guess(self) -> np.ndarray: ...

    def _extract_field(self, x: np.ndarray) -> object:
        """Discretization-specific hook: DOF iterate -> the field a FieldEvaluator consumes.

        The ONE place solver-specific physics re-enters this seam (MEC: reduced loop flux ->
        per-branch ``|B|``; FEA: nodal potentials -> ``curl(A)``, inline per valence kernel since
        T3/Q4/VEM have different field formulas). Everything downstream of this hook -- the
        evaluator lookup, the coefficient state, the energy/stiffness assembly -- is granularity-
        aware but no longer needs to know which discretization produced the field.
        """

    def assemble(self, x: np.ndarray) -> AssembledState: ...
