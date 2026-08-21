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

Granularity is ONE axis, independent of which equation is being solved -- MEC and FEA are two
DISCRETIZATIONS, and each granularity (branch / element / edge) is a Protocol any discretization
can implement (MEC's ``PerElementFromBranches`` and FEA's ``PerElement`` both satisfy
:class:`ElementFieldEvaluator`, from two different equations). What this module deliberately does
NOT do is force one calling CONVENTION across granularities: :class:`BranchFieldEvaluator`,
:class:`ElementFieldEvaluator` and :class:`EdgeFieldEvaluator` each declare their OWN method
(``evaluate_branch`` / ``evaluate_element`` / ``evaluate_edges``) rather than a shared ``evaluate``,
because unifying the signature would mean rewriting the T3/Q4/VEM per-element kernels' hot loops
purely for interface uniformity, risking the bit-identity every golden here depends on (AGENTS.md's
"implement, don't exclude" cuts against a refactor whose only benefit is symmetry). This module also
does not replace ``tangent()``, which stays on the equation itself (MEC and FEA compute their Newton
differential term directly from the material's reluctivity SLOPE, never through an evaluator -- P2
never routed a tangent through this seam either, so there is nothing to supersede).

``_ReluctanceSystem`` (MEC) and ``GalerkinSystem`` (FEA) COMPOSE this pair: each names its own
``_extract_field`` hook (MEC: reduced loop flux -> per-branch ``|B|``; FEA: per-element/per-block
nodal potentials -> ``curl(A)``, inline in each valence kernel because T3/Q4/VEM have genuinely
different field formulas) and holds a ``FieldEvaluator`` instance -- never ``None`` after
construction, so ``assemble()`` composes it unconditionally instead of branching on
``field_evaluator is not None``.

**What these Protocols do NOT do: nothing at runtime checks that an evaluator actually
conforms.** ``@runtime_checkable`` only makes ``isinstance()`` legal to call against them; nothing
in this codebase calls it. Every call site (``_sparse.py``'s ``self.field_evaluator.evaluate_branch``,
the T3/Q4/VEM kernels' ``field_evaluator.evaluate_element``) either accesses the method directly
(an ``AttributeError`` is the only enforcement: it fires if the method is MISSING, never if it is
present but wrong) or resolves it via ``getattr(..., 'evaluate_edges', None)`` duck-typing, exactly
as before this Protocol pair existed. That is a deliberate, MEASURED choice, not an oversight: a
caller-supplied test evaluator (``test_field_evaluator_seam.py``'s ``_ScaledElementEvaluator``)
duck-types ``evaluate_element`` without declaring ``granularity``, so an ``isinstance``-gated
dispatch would silently fall back to the default evaluator for that evaluator -- the exact
"declaration that lies" failure this seam exists to prevent (a threaded parameter that looks
consulted but isn't). So: these Protocols are STRUCTURAL documentation with a type checker's
blessing, consulted by pyright at edit time, never by a runtime check. A class with the right
method NAME and a plausible ``granularity`` string satisfies all three even if its body computes
something physically wrong -- Protocols check shape, never behaviour, and this module adds no
behavioural check on top. If a future change wants a REAL runtime guard (e.g. a registry that
validates a caller-supplied evaluator before a solve), it needs to be added explicitly and will
have the same duck-typing-safety question this note describes -- it does not come free with the
Protocol declaration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from motronics.gauss.nonlinear import AssembledState

__all__ = [
    'AssemblyEquation',
    'BranchFieldEvaluator',
    'CoefficientState',
    'EdgeFieldEvaluator',
    'ElementFieldEvaluator',
    'FieldEvaluator',
]


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

    This BASE Protocol intentionally carries no ``evaluate`` method of its own: the field-evaluation
    calling convention differs by granularity (a per-branch ``|B|`` array plus the owning system as
    geometry context; a single scalar ``|B|`` per call inside the T3/Q4/VEM assembly loop; one
    element's ``(3, 2)`` edge-vector block), so each granularity gets its OWN narrower Protocol
    below (:class:`BranchFieldEvaluator`, :class:`ElementFieldEvaluator`, :class:`EdgeFieldEvaluator`)
    rather than one forced shared signature. Forcing one signature here would mean rewriting the
    per-element hot loops purely for interface symmetry, trading the bit-identity every golden
    depends on for a cosmetic uniformity this base Protocol does not need in order to do its job:
    naming what an evaluator IS (``granularity``) independently of how it is called.
    """

    granularity: str


@runtime_checkable
class BranchFieldEvaluator(FieldEvaluator, Protocol):
    """A :class:`FieldEvaluator` keyed on MEC's branch granularity (``granularity == 'branch'``).

    Closes the composition pyright's structural check demands: ``_ReluctanceSystem.field_evaluator``
    is typed as THIS narrower Protocol (not the bare ``FieldEvaluator``), so calling
    ``.evaluate_branch(...)`` on it is checked, not merely duck-typed at the call site. Every branch
    evaluator (:class:`~motronics.hamilton.formulations.magnetic_circuit._sparse.PerBranch`,
    ``PerElementFromBranches``) implements this.
    """

    def evaluate_branch(self, b_abs: np.ndarray, system: object) -> tuple[np.ndarray, np.ndarray]:
        """Returns ``(R_mat_diag, energy_branch)`` over ALL branches.

        ``system`` is the owning ``_ReluctanceSystem`` -- typed ``object`` here (not imported: that
        would invert the ``gauss`` -> ``hamilton`` dependency arrow) but ALWAYS the geometry context
        an evaluator needs (``l_div_A``, ``A_mat``, ``l_mat``, ``group_material_list``, ``material``).
        """
        ...


@runtime_checkable
class ElementFieldEvaluator(FieldEvaluator, Protocol):
    """A :class:`FieldEvaluator` keyed on FEA's element granularity (``granularity == 'element'``).

    Implemented by :class:`~motronics.hamilton.discretizations.fem._nu_element.PerElement` (the
    default) and by MEC's ``PerElementFromBranches`` (P4's element-granularity experiment on the
    MEC discretization -- the same PROTOCOL, a different discretization, exactly the point that
    granularity is one axis independent of which equation is being solved).
    """

    def evaluate_element(
        self,
        law: object,
        b_norm: float,
        max_B_norm: float,
        nu_prev: float,
        alpha: float,
        temperature: float | None = None,
    ) -> tuple[float, float]:
        """Returns ``(nu_target, reluctivity)`` -- see :func:`eval_nu_element`'s docstring."""
        ...


@runtime_checkable
class EdgeFieldEvaluator(FieldEvaluator, Protocol):
    """A :class:`FieldEvaluator` keyed on the P4 edge granularity (``granularity == 'edge'``).

    Implemented by
    :class:`~motronics.hamilton.discretizations.fem._nu_element.PerEdgeFromElement`, the
    discriminating-experiment evaluator that gives FEA's T3 kernel MEC's per-branch-like
    granularity.
    """

    def evaluate_edges(
        self,
        law: object,
        current_B: np.ndarray,
        edge_vectors: np.ndarray,
        max_B_norm: float,
        temperature: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns ``(nu_edge, b_edge)`` for one element's edges."""
        ...


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
