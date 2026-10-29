"""The numerical statement of a time-marched problem: weights, integrator, window, problem protocols.

A spatial discretisation compiles to a :class:`SemiDiscreteProblem` exposing
``residual(u, u_dot, u_ddot, t)`` and ``jacobian(..., weights)``; a temporal analysis drives it
only through a :class:`DerivativeWeights` triple weighting the ``u`` / ``u_dot`` / ``u_ddot``
operators. The marchers read a :class:`TimeWindow`. None of this names a physical field, so the
engine owns it and the product's study specification builds on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

__all__ = [
    'NEWMARK_BETA',
    'NEWMARK_GAMMA',
    'AffineConstraintAdmissionProblem',
    'AffineConstraintProjectionProblem',
    'AlternativeJacobianProblem',
    'DerivativeWeights',
    'FactorReusableProblem',
    'HistoryProblem',
    'InitialJointEvaluation',
    'IntegratorKind',
    'ResidualAssemblyBound',
    'ResidualFloorProblem',
    'SemiDiscreteProblem',
    'StepEvaluationProblem',
    'TimeWindow',
]

#: Newmark defaults (average-acceleration, unconditionally stable, second order): the
#: current-step implicit weights use these; the history terms are the integrator's job. One owner.
NEWMARK_BETA = 0.25
NEWMARK_GAMMA = 0.5


@dataclass(frozen=True)
class DerivativeWeights:
    """Weights ``(alpha0, alpha1, alpha2)`` on the ``u`` / ``u_dot`` / ``u_ddot`` operators.

    The semi-discrete operator a jacobian assembles is
    ``alpha0*K + alpha1*C + alpha2*M`` (stiffness / damping / mass). A study supplies ONLY
    these three numbers; the operators themselves come from the discretisation. Complex-typed so the harmonic
    ``alpha1 = j*omega`` is expressible; a real study leaves them real.
    """

    alpha0: complex = 1.0
    alpha1: complex = 0.0
    alpha2: complex = 0.0

    @property
    def collapses_to_stationary(self) -> bool:
        """True when only ``alpha0`` is nonzero -- the equation reduces to the stiffness ``K``."""
        return self.alpha1 == 0 and self.alpha2 == 0


@runtime_checkable
class SemiDiscreteProblem(Protocol):
    """The seam a spatial discretisation implements: residual and weighted jacobian of a state.

    A backend exposes the residual and jacobian as functions of the state and its time
    derivatives; a temporal analysis drives them purely through :class:`DerivativeWeights`.
    ``residual`` evaluates ``R(u, u_dot, u_ddot, t)``; ``jacobian`` returns the weighted operator
    ``alpha0*dR/du + alpha1*dR/du_dot + alpha2*dR/du_ddot``. Nonlinear (Newton/Picard) and
    linear backends consume this without either knowing the temporal analysis.
    """

    def residual(
        self,
        u: NDArray[np.number],
        u_dot: NDArray[np.number],
        u_ddot: NDArray[np.number],
        t: float,
    ) -> NDArray[np.number]:
        """Evaluate the semi-discrete residual at the given state and time derivatives."""
        ...

    def jacobian(
        self,
        u: NDArray[np.number],
        u_dot: NDArray[np.number],
        u_ddot: NDArray[np.number],
        t: float,
        weights: DerivativeWeights,
    ) -> object:
        """Return the weighted tangent operator (a matrix-like) for the given weights."""
        ...


@runtime_checkable
class StepEvaluationProblem(SemiDiscreteProblem, Protocol):
    """A physical owner with a one-sided constitutive view for an incoming step.

    The returned problem evaluates at the same real time, never a perturbed clock.
    It owns endpoint law selection; accepted observations and outgoing projections
    still belong to the original owner, not the speculative step view.
    """

    def step_problem(self, time: float) -> SemiDiscreteProblem:
        """Bind this solve's endpoint law without advancing physical history."""
        ...


@runtime_checkable
class AffineConstraintAdmissionProblem(SemiDiscreteProblem, Protocol):
    """An owner that certifies its actual affine plane at accepted-state arithmetic resolution.

    This is distinct from trial projection: a projected floating state need not
    satisfy multiple affine relations bitwise. Only the physical row owner can
    certify their performed arithmetic, independently of the solve tolerance.
    """

    def affine_state_is_admitted(self, state: NDArray[np.floating], time: float) -> bool: ...


@runtime_checkable
class AffineConstraintProjectionProblem(SemiDiscreteProblem, Protocol):
    """An owner of exact, derivative-free affine state constraints.

    Projection is a pure numerical trial operation at the same physical time;
    it does not admit a new stored state or advance accepted history. The owner
    selects its actual constraint rows/map, never a residual-size heuristic.
    """

    def project_affine_state(self, state: NDArray[np.floating], time: float) -> NDArray[np.floating]: ...


@runtime_checkable
class FactorReusableProblem(SemiDiscreteProblem, Protocol):
    """A semi-discrete problem that can prove when its weighted Jacobian is unchanged."""

    def factor_reuse_key(
        self,
        u: NDArray[np.number],
        u_dot: NDArray[np.number],
        u_ddot: NDArray[np.number],
        t: float,
        weights: DerivativeWeights,
    ) -> object:
        """Return a key containing every input that changes :meth:`jacobian`."""
        ...


@runtime_checkable
class ResidualFloorProblem(SemiDiscreteProblem, Protocol):
    """A problem that can bound its own floating-point residual cancellation."""

    def residual_roundoff_floor(
        self, u: NDArray[np.floating], form: object, residual: NDArray[np.floating] | None = None
    ) -> float | NDArray[np.floating] | None:
        """Return the residual floor -- a scalar for ``||r||_2`` or one bound per row -- or ``None``.

        ``residual`` is the vector the caller will judge. A problem MAY then return any smaller
        bound that still covers it (the verdict is the same), never a smaller one that does not.
        """
        ...


@runtime_checkable
class AlternativeJacobianProblem(SemiDiscreteProblem, Protocol):
    """A problem offering a useful alternative Jacobian for a stalled Newton direction."""

    @property
    def has_tangent_jacobian(self) -> bool:
        """Whether the alternative differs usefully from the default Jacobian."""
        ...

    def jacobian_tangent(
        self,
        u: NDArray[np.number],
        u_dot: NDArray[np.number],
        u_ddot: NDArray[np.number],
        t: float,
        weights: DerivativeWeights,
    ) -> object:
        """Return the alternative weighted Jacobian at the given state."""
        ...


@runtime_checkable
class HistoryProblem(SemiDiscreteProblem, Protocol):
    """A problem whose constitutive law carries a HISTORY (hysteresis) that an accepted step commits.

    Inside one step every residual/jacobian evaluation is a pure function of the iterate, taken
    against the history committed by the LAST accepted step; :meth:`commit_history` makes ``u`` at
    ``t`` the new history. A marcher calls it once per step that joins the trajectory -- never on a
    rejected or retried attempt -- so a restart from a committed history is the same march.
    """

    def commit_history(self, u: NDArray[np.number], t: float) -> None:
        """Accept ``u`` at time ``t`` as the history every later evaluation is taken against."""
        ...


class IntegratorKind(StrEnum):
    """The time integrator a :class:`TimeWindow` marches with.

    ``BDF1`` / ``BDF2`` are the first / second-order backward-difference schemes for a
    first-order (eddy-current) system; ``TRAPEZOID`` is the A-stable second-order trapezoidal rule
    (SPICE's default) and ``TR_BDF2`` its L-stable two-stage composite (a trapezoid stage then a
    BDF2 stage, both with the SAME jacobian weight); ``NEWMARK`` / ``GENERALIZED_ALPHA`` are the
    second-order (u-ddot) schemes.
    """

    BDF1 = 'bdf1'
    BDF2 = 'bdf2'
    TRAPEZOID = 'trapezoid'
    TR_BDF2 = 'tr_bdf2'
    NEWMARK = 'newmark'
    GENERALIZED_ALPHA = 'generalized_alpha'


@dataclass(frozen=True)
class TimeWindow:
    """A march over ``[t0, t1]`` with nominal step ``dt`` and a chosen integrator."""

    t0: float
    t1: float
    dt: float
    integrator: IntegratorKind = IntegratorKind.BDF2

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            msg = f'transient dt must be positive, got {self.dt}'
            raise ValueError(msg)
        if self.t1 < self.t0:
            msg = f'transient interval must be non-decreasing, got [{self.t0}, {self.t1}]'
            raise ValueError(msg)


@dataclass(frozen=True)
class ResidualAssemblyBound:
    """Actual row-local residual construction error, as two NAMED terms.

    ``values``/``known``: the performed ARITHMETIC's rounding; an unknown row is not a zero proof.
    ``law``: the CONSTITUTIVE LAW's certified evaluation bound propagated to each row -- a law the
    residual reads only through a certified inverse (a hysteresis model's ``H(B)``) is known to that
    inverse's own acceptance bound, never better. Zero where every law is evaluated in closed form.
    It is certified independently of ``known``: an uncounted arithmetic does not unknow it.
    """

    values: NDArray[np.float64]
    known: NDArray[np.bool_]
    law: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        values, known = np.array(self.values, dtype=float, copy=True), np.array(self.known, copy=True)
        law = np.zeros_like(values) if self.law is None else np.array(self.law, dtype=float, copy=True)
        if (
            values.ndim != 1
            or known.shape != values.shape
            or law.shape != values.shape
            or known.dtype.kind != 'b'
            or not np.isfinite(values).all()
            or not np.isfinite(law).all()
            or np.any(values < 0)
            or np.any(law < 0)
        ):
            msg = 'residual assembly bounds require aligned finite nonnegative values and law terms and a known mask'
            raise ValueError(msg)
        values.flags.writeable = known.flags.writeable = law.flags.writeable = False
        object.__setattr__(self, 'values', values)
        object.__setattr__(self, 'known', known)
        object.__setattr__(self, 'law', law)

    def allowance(self) -> NDArray[np.float64]:
        """What each row's residual is known to: its certified arithmetic plus its law term."""
        return np.where(self.known, self.values, 0.0) + self.law

    def take(self, rows) -> ResidualAssemblyBound:
        """The bound on a subset of rows."""
        return ResidualAssemblyBound(self.values[rows], self.known[rows], self.law[rows])


@dataclass(frozen=True)
class InitialJointEvaluation:
    """Original-equation columns at fixed independent rate, not reconstructed sums.

    Matrices retain their owner's sparse representation. Differentiated columns
    are derivatives of the original lifted ``F_u v + M a + F_t`` before algebraic projection;
    ``None`` declares unavailable authority, never a zero derivative. Rate maps
    describe the actual lift only and do not certify residual columns.
    """

    physical_rate: np.ndarray
    residual_state_columns: object
    residual_rate_columns: object
    differentiated_state_columns: object | None
    differentiated_rate_columns: object | None
    physical_rate_state_map: object
    physical_rate_rate_map: object
    differentiated_value: np.ndarray | None = None
    physical_acceleration_offset: np.ndarray | None = None
    differentiated_bound: ResidualAssemblyBound | None = None

    def __post_init__(self) -> None:
        if np.iscomplexobj(self.physical_rate):
            msg = 'joint coordinates require a finite physical rate vector with real values'
            raise ValueError(msg)
        rate = np.array(self.physical_rate, dtype=float, copy=True)
        if rate.ndim != 1 or not np.isfinite(rate).all():
            msg = 'joint coordinates require a finite physical rate vector'
            raise ValueError(msg)
        rate.flags.writeable = False
        object.__setattr__(self, 'physical_rate', rate)
        for name in ('differentiated_value', 'physical_acceleration_offset'):
            value = self.differentiated_value if name == 'differentiated_value' else self.physical_acceleration_offset
            if value is None:
                continue
            if np.iscomplexobj(value) or any(
                isinstance(item, (bool, np.bool_)) for item in np.asarray(value, dtype=object).flat
            ):
                msg = 'joint differentiated values require real physical authority'
                raise ValueError(msg)
            vector = np.array(value, dtype=float, copy=True)
            if vector.shape != rate.shape or not np.isfinite(vector).all():
                msg = 'joint differentiated values require finite aligned physical vectors'
                raise ValueError(msg)
            vector.flags.writeable = False
            object.__setattr__(self, name, vector)
        if self.differentiated_bound is not None and (
            self.differentiated_value is None
            or not isinstance(self.differentiated_bound, ResidualAssemblyBound)
            or self.differentiated_bound.values.shape != rate.shape
        ):
            msg = 'joint differentiated bound requires its same aligned performed value'
            raise ValueError(msg)
