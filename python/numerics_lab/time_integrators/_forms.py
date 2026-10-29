"""Integrator state, affine step forms, trajectories and arithmetic-floor algebra.

No imports from the package facade: both the concrete integrators and the march
consume this leaf, so importing a concrete scheme never creates a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from numerics_lab.formulation import DerivativeWeights
from numerics_lab.linear import solve_linear_system
from numerics_lab.linear.resolution import FLOOR_SAFETY, row_bar, system_resolution
from numerics_lab.time_integrators.checkpoint import TransientCheckpoint

__all__ = [
    'ITimeIntegrator',
    'IntegratorState',
    'ProjectedRestartIntegrator',
    'StepForm',
    'TransientSolution',
    'change_below_residual_resolution',
    'residual_floor',
    'residual_floor_by_row',
    'residual_floor_of_magnitude',
    'seeded_bdf2_state',
]


#: Safety factor on the estimated roundoff floor, owned by :mod:`numerics_lab.linear.resolution`
#: and DELIBERATELY the same 64 as the DC operating point's ``laplace.circuit.core.mna_solves._FLOOR_SAFETY``.
_FLOOR_SAFETY = FLOOR_SAFETY

#: The machine epsilon the floor is written in (double precision -- the marcher is ``float64``).
_EPS = float(np.finfo(float).eps)


def residual_floor(tangent, u: NDArray[np.floating]) -> float:
    """The smallest residual norm this system can express at double precision, in RESIDUAL units.

    THE SAME CRITERION the DC operating point already uses -- ``mna_solves._roundoff_floor`` /
    ``CircuitDAE.op_residual_floor`` -- written here in the norm THIS layer measures its residual
    in. The residual is a row sum of signed contributions ``sum_j J_ij u_j``; summing terms of
    magnitude ``M`` in floating point cannot resolve below ``eps * M``, so ``(|J| . |u|)_i`` is
    the magnitude per row and ``eps`` times it is the smallest residual that row can MEAN. The op
    solve compares ``max|R|`` and so takes the row MAX; this marcher compares ``||R||_2`` and so
    takes the same vector's 2-norm. One criterion, each layer in its own norm.

    THE UNIT. It carries the residual's own units, because ``|J| . |u|`` does -- amperes on a KCL
    row, volts on a branch row. That is what the ABSOLUTE ``tol = 1e-12`` could not do, and
    ``rem/porting.md`` names that constant class as the one that cannot survive a change of scale.
    MEASURED 2026-09-12 on a linear series R-L marched by BDF2: the exact one-iteration solve
    lands at 6.17e-13 on a 24 V supply and 4.95e-11 on a 2400 V one -- the SAME solve, and only
    the second one was reported as having exhausted its Newton ceiling. The op solve's own table
    records the same shape from the other side: one circuit's floor spans 56x across its four
    gate states, so no single constant can sit above the lowest and below the highest.

    COMPUTED ONLY ON EXHAUSTION, never per iteration -- the DC op's rule, for its reason: it costs
    an ``|J| . |u|`` matvec, and it can only ever change the verdict where the plain bars have
    already been missed. So it alters no iteration count and no trajectory: strictly the LABEL.
    """
    magnitude = _abs_matvec(tangent, np.abs(np.asarray(u, dtype=float)).reshape(-1))
    return residual_floor_of_magnitude(magnitude)


def residual_floor_by_row(tangent, u: NDArray[np.floating]) -> NDArray[np.floating]:
    """The same arithmetic floor, retaining each equation's own residual units."""
    magnitude = _abs_matvec(tangent, np.abs(np.asarray(u, dtype=float)).reshape(-1))
    return _roundoff_floor(magnitude)


def _abs_matvec(tangent, x: NDArray[np.floating]) -> NDArray[np.floating]:
    if sp.issparse(tangent):
        return np.asarray(abs(tangent) @ x).reshape(-1)
    dense = np.abs(np.asarray(tangent, dtype=float))
    return (dense * x if dense.ndim == 0 else dense @ x).reshape(-1)


def change_below_residual_resolution(
    tangent, before: NDArray[np.floating], after: NDArray[np.floating], forcing: NDArray[np.floating] | None
) -> bool:
    """Whether the iterate change ``after - before`` is invisible to the system's arithmetic.

    Row ``i`` cannot resolve a change below ``_FLOOR_SAFETY * eps`` of its own ``(|J| . |u|)_i``; a
    HOMOGENEOUS row (``forcing``, the residual at the zero state, exactly zero) whose operands are
    themselves under the SYSTEM's resolution is noise and is judged as zero
    (:func:`~numerics_lab.linear.resolution.row_bar`). A change under that on EVERY row cannot be
    confirmed or corrected by another Newton iteration: it is rounding in the iterate. Judged only
    by its own operands, a field row whose operands are all rounding has no scale of its own --
    MEASURED 2026-10-14 on case_l t = 2.05e-7: nodal A at 2.7e-40 moved 2% per
    iteration beside 300 V circuit rows that did not move, and was refused 50 times.

    This is what a per-field RELATIVE change cannot see on a field that is physically zero. Measured
    2026-10-13 on case_l at rest: the 96834 nodal A values sat at 1e-40 Wb/m,
    pure solve noise, and their change relative to their own noise scale stayed 0.03..2 on every
    iteration, so the relative bar refused a step whose residual was already at its floor and the
    adaptive march halved the step to its minimum.
    """
    before = np.asarray(before, dtype=float).reshape(-1)
    after = np.asarray(after, dtype=float).reshape(-1)
    change = _abs_matvec(tangent, np.abs(after - before))
    magnitude = _abs_matvec(tangent, np.maximum(np.abs(before), np.abs(after)))
    if not (np.isfinite(change).all() and np.isfinite(magnitude).all()):
        return False
    own = _FLOOR_SAFETY * _EPS * magnitude  # the residual_floor criterion, per row
    declared = None if forcing is None else np.ravel(forcing)
    return bool(np.all(change <= row_bar(own, magnitude, system_resolution(magnitude), declared)))


def residual_floor_of_magnitude(magnitude: NDArray[np.floating]) -> float:
    """THE FLOOR RULE ITSELF, for a caller that builds ``|J| . |u|`` without ever forming ``J``.

    :func:`residual_floor` is this function applied to ``|J| . |u|``; the bordered field<->circuit
    system (a downstream monolithic coupling module) can supply the magnitude but NOT the
    matrix -- ``J`` there is a 2x2 block system whose ``(n1, n1)`` corner is a 52185-DOF sparse
    field tangent, so an assembly is the defect that module exists to avoid. It builds the stacked
    magnitude from its four blocks and hands it here.

    ONE HOME FOR ``_FLOOR_SAFETY * _EPS * ||.|``, because a second spelling of it is exactly the
    drift the downstream backtrack cap's comment warns about: a
    criterion written twice is two criteria as soon as one of them is edited.
    """
    magnitude = np.asarray(magnitude, dtype=float).reshape(-1)
    return float(_roundoff_floor(float(np.linalg.norm(magnitude)))) if magnitude.size else 0.0


def _roundoff_floor(magnitude):
    return _FLOOR_SAFETY * _EPS * np.asarray(magnitude, dtype=float)


@dataclass
class IntegratorState:
    """The marching history an integrator carries between steps.

    ``us`` holds past state vectors most-recent-LAST (BDF2 reads the last two); ``v`` / ``a`` are
    the velocity ``u_dot`` and acceleration ``u_ddot`` at the last step (the second-order schemes
    and the trapezoid propagate them, BDF leaves them ``None``). ``t`` is the last step's time.

    ``h`` is the spacing of the last step (``None`` right after a start/restart): it IS history,
    because a multistep form differentiates against it, so it lives here and is checkpointed and
    rolled back with the vectors rather than tracked by each marcher. ``stage`` is the next stage
    of a multi-stage scheme (TR-BDF2); single-stage schemes keep it 0.
    ``prepared_candidate`` is an immutable handle to an owner's speculative hidden-history
    snapshot. Copies retain this explicit authority; portable checkpoints restore the owner's
    actual history instead of serializing process-local handles.
    """

    us: list[NDArray[np.floating]]
    t: float
    v: NDArray[np.floating] | None = None
    a: NDArray[np.floating] | None = None
    h: float | None = None
    stage: int = 0
    prepared_candidate: object | None = None

    def copy(self) -> IntegratorState:
        """A deep copy -- a checkpoint must never alias the arrays a probe step goes on to touch."""
        return IntegratorState(
            us=[np.array(u, dtype=float, copy=True) for u in self.us],
            t=self.t,
            v=None if self.v is None else np.array(self.v, dtype=float, copy=True),
            a=None if self.a is None else np.array(self.a, dtype=float, copy=True),
            h=self.h,
            stage=self.stage,
            prepared_candidate=self.prepared_candidate,
        )

    @property
    def last_u(self) -> NDArray[np.floating]:
        return self.us[-1]


@dataclass(frozen=True)
class StepForm:
    """The affine per-step form an integrator hands the driver for the unknown ``u_{n+1}``.

    Every implicit derivative is affine in the unknown: ``u_arg = alpha0*u + h_u`` (the state the
    residual is evaluated at -- the identity ``u`` for BDF/Newmark, an intermediate point for
    generalized-alpha), ``u_dot = alpha1*(u-rate_origin) + h_dot`` (zero origin when absent),
    ``u_ddot = alpha2*u + h_ddot``. The slopes
    are exactly the :class:`DerivativeWeights` the jacobian consumes; the offsets are the history.
    ``t_eval`` is the time the residual/source is sampled at (shifted by ``alpha_f*dt`` for
    generalized-alpha).
    """

    weights: DerivativeWeights
    h_u: NDArray[np.floating]
    h_dot: NDArray[np.floating]
    h_ddot: NDArray[np.floating]
    t_eval: float
    #: Rate history is centered here to avoid subtracting large weighted states.
    rate_origin: NDArray[np.floating] | None = None

    def u_arg(self, u: NDArray[np.floating]) -> NDArray[np.floating]:
        return self.weights.alpha0.real * u + self.h_u

    def u_dot(self, u: NDArray[np.floating]) -> NDArray[np.floating]:
        delta = u if self.rate_origin is None else u - self.rate_origin
        return self.weights.alpha1.real * delta + self.h_dot

    def u_ddot(self, u: NDArray[np.floating]) -> NDArray[np.floating]:
        return self.weights.alpha2.real * u + self.h_ddot


class ITimeIntegrator(Protocol):
    """The seam a time integrator implements -- mirrors ``gauss.nonlinear.IterationStrategy``.

    ``weights(dt)`` is the textbook steady-state ``(alpha0, alpha1, alpha2)`` triple (what an
    analysis formulation carries); ``start`` seeds the history; ``prepare`` produces the per-step
    :class:`StepForm` from the current history; ``advance`` folds the just-solved ``u_{n+1}`` back
    into the history (BDF pushes the state, the second-order schemes also update ``v`` / ``a``).
    """

    name: str
    order: int
    #: Implicit solves per step (TR-BDF2: 2). The marcher runs ``prepare``/solve/``advance`` this
    #: many times per step; the intermediate stage states never join the trajectory.
    stages: int
    #: The local-truncation-error constant ``C`` in ``LTE = C h^(order+1) u^(order+1)``: what the
    #: step-size controller scales its predictor-corrector difference by.
    error_constant: float

    def weights(self, dt: float) -> DerivativeWeights: ...

    def start(
        self,
        u0: NDArray[np.floating],
        t0: float,
        v0: NDArray[np.floating] | None = None,
        a0: NDArray[np.floating] | None = None,
    ) -> IntegratorState: ...

    def prepare(self, state: IntegratorState, dt: float, t_new: float) -> StepForm: ...

    def advance(
        self, state: IntegratorState, u_new: NDArray[np.floating], dt: float, t_new: float, form: StepForm
    ) -> IntegratorState: ...

    def restart(self, state: IntegratorState, u: NDArray[np.floating], t: float) -> IntegratorState:
        """Drop the derivative history at a discontinuity (an event), keeping what stays continuous."""
        ...


@runtime_checkable
class ProjectedRestartIntegrator(Protocol):
    """Optional actual scheme-owned seed of an explicit outgoing derivative pair."""

    def restart_at_pair(self, state, u, t, rate, acceleration) -> IntegratorState: ...


#: THE CONCRETE INTEGRATORS AND THEIR REGISTRY live in `_integrators.py`, split 2026-08-22
#: because this file crossed the 800-line band. It imports :class:`IntegratorState`,
#: :class:`ITimeIntegrator` and :class:`StepForm` FROM THIS MODULE (a circular import that works
#: because they are defined above this import); the public API is unchanged.
@dataclass
class TransientSolution:
    """The time march: parallel ``times`` and ``states`` including the initial condition.

    The CONVERGENCE CONTRACT (the FVS plan's near-term item): a march that exhausted its
    per-step Newton ceiling somewhere is LABELLED, never silent -- the worst step residual
    and the offending step times ride the solution, and :meth:`converged` is the machine-
    readable tag. The structured warning is emitted by the marcher at the first exhaustion
    (AGENTS.md: a log is a signal only when the CALLER can tell -- the caller reads the tag).

    ``nonconverged_times`` names only steps that LANDED in ``times``. The eventful march also
    takes bisection probe steps that ``_rollback`` discards; an exhausted probe costs time and
    slightly degrades the located instant, but its error never enters the trajectory, so it is
    recorded separately in ``nonconverged_probe_times`` and does NOT clear :meth:`converged`.
    Conflating the two made the tag ~9x pessimistic on a switching march (srdab_closed_loop,
    2026-08-15: 2094 reported, 239 landed) -- always False wherever it mattered, so no caller
    could use it.
    """

    times: list[float] = field(default_factory=list)
    states: list[NDArray[np.floating]] = field(default_factory=list)
    #: Actual residual-evaluation rates at committed times[1:], with no invented t0 rate.
    #: None means a producer did not record them; it is NOT an all-zero derivative.
    state_rates: list[NDArray[np.floating]] | None = None
    #: Actual final-stage residual times at times[1:], including left limits at edges.
    #: None means unrecorded; consumers must not infer them by tolerant matching.
    residual_times: list[float] | None = None
    #: Sparse pre-handler residual states keyed by accepted rate index. None is unrecorded;
    #: an empty dict verifies that every published state retains its residual-side value.
    rate_state_overrides: dict[int, NDArray[np.floating]] | None = None
    worst_step_residual: float = 0.0
    nonconverged_times: list[float] = field(default_factory=list)
    #: Exhausted steps that were DISCARDED by an event rollback (bisection probes). Diagnostic
    #: only -- these never reached ``times``. A large count means the locator worked hard at the
    #: toggles, not that the answer is contaminated.
    nonconverged_probe_times: list[float] = field(default_factory=list)
    #: Steps whose ``converged`` verdict was carried by :func:`residual_floor` rather than by
    #: ``tol``/``tol_rel``: they EXHAUSTED the Newton ceiling and landed at the smallest
    #: residual double precision can express for that system. The floor is the correctly
    #: SCALED bar, so such a step IS converged and this list is not a failure record. It
    #: exists because without it the two verdicts are the same value -- a march that genuinely
    #: stopped converging, but whose residual sits at the arithmetic floor, reads
    #: ``converged=True`` with nothing to distinguish it, and a test asserting convergence
    #: cannot tell the physics from the arithmetic. Like ``worst_step_residual`` and unlike
    #: ``nonconverged_times``, this includes the eventful march's DISCARDED bisection probes:
    #: it is a diagnostic about the SOLVES, not a property of the trajectory.
    floor_converged_times: list[float] = field(default_factory=list)
    #: The located state events as ``(function index, time)`` pairs (empty without ``event_fns``).
    events: list[tuple[int, float]] = field(default_factory=list)
    #: Owned final integration state: segmented marches preserve continuous velocity/acceleration
    #: while restarting derivative history at a sampled-input discontinuity.
    final_integrator_state: IntegratorState | None = None
    checkpoint: TransientCheckpoint | None = None
    corrected_times: list[float] = field(default_factory=list)
    #: Actual runtime coarse correction model; absent when correction is not enabled.
    correction_model: str | None = None
    #: Accepted steps whose predictor-curvature indicator exceeded its requested bar.
    #: Independent of Newton convergence; a minimum step can force their acceptance.
    indicator_unmet_times: list[float] = field(default_factory=list)
    #: Discontinuities dispatched only at accepted fixed-lattice states, not located probes.
    landed_events: list[tuple[int, float]] = field(default_factory=list)
    #: Genuine accepted observer law changes that restarted derivative history, not located event IDs.
    accepted_discontinuities: list[float] = field(default_factory=list)
    #: Newton iterations over every step the march took, discarded bisection probes included: the
    #: solver WORK, which is what a per-step cost is divided by.
    newton_iterations: int = 0

    @property
    def history_restarts(self) -> tuple[float, ...]:
        """Actual derivative-history restart times: accepted law changes, landed and located events."""
        return tuple(
            sorted(
                set(self.accepted_discontinuities)
                | {time for _index, time in self.landed_events}
                | {time for _index, time in self.events}
            )
        )

    @property
    def indicator_satisfied(self) -> bool:
        return not self.indicator_unmet_times

    @property
    def converged(self) -> bool:
        """The accuracy tag: every step IN THIS SOLUTION met the per-step Newton tolerance."""
        return not self.nonconverged_times

    @property
    def converged_within_the_ceiling(self) -> bool:
        """The STRONGER tag: every step converged on its BAR, none of them on the floor.

        :attr:`converged` cannot distinguish the two, and that is the whole reason both exist:
        a caller that means "the physics converged" rather than "the arithmetic could say no
        more" asks for this one. A step ON the floor exhausted its ceiling, so this is also
        the tag to read when deciding whether ``max_newton`` is set high enough.
        """
        return self.converged and not self.floor_converged_times


def _default_solve(matrix: object, rhs: NDArray[np.floating]) -> NDArray[np.floating]:
    """Solve ``matrix @ x = rhs`` for a dense ndarray or a scipy-sparse jacobian (physics-blind)."""
    b = np.asarray(rhs, dtype=float)
    if sp.issparse(matrix):
        return np.asarray(solve_linear_system(sp.csc_matrix(matrix), b))
    m = np.asarray(matrix, dtype=float)
    if m.ndim == 0:  # scalar jacobian (1-DOF problem written as a plain number)
        return b / float(m)
    return np.linalg.solve(m, b)


def seeded_bdf2_state(state, history, integrator, restart, dt):
    """Validate a fixed-step two-state seed with history times t0-dt and t0."""
    arrays = tuple(np.asarray(value, dtype=float) for value in history)
    if (
        integrator.name != 'bdf2'
        or restart is not None
        or len(arrays) != 2
        or any(value.shape != state.last_u.shape or not np.all(np.isfinite(value)) for value in arrays)
        or not np.array_equal(arrays[-1], state.last_u)
        or not np.isfinite(dt)
        or dt <= 0
    ):
        msg = 'initial_history requires exactly two finite BDF2 states ending at u0 without restart'
        raise ValueError(msg)
    return IntegratorState(us=[value.copy() for value in arrays], t=state.t, h=float(dt))
