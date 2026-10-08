"""The time-integration CONTRACT: the state, the step form, the integrator protocol, the transient
solution, and the round-off floor a residual is judged against.

Split out of the package `__init__` on 2026-09-29, as the leaf the integrators and the step solve
are both built on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from motronics.contracts.spec.analysis_formulation import (
    DerivativeWeights,
)

__all__ = [
    'ITimeIntegrator',
    'IntegratorState',
    'StepForm',
    'TransientSolution',
    'residual_floor',
    'residual_floor_of_magnitude',
    'same_instant',
]


#: Safety factor on the estimated roundoff floor, DELIBERATELY the same 64 as the DC operating
#: point's ``laplace.circuit.core.mna_solves._FLOOR_SAFETY``, and for the same stated reason: the
#: estimate bounds the rounding error of ONE row sum, while the accumulated error over a row of
#: ``n`` assembled terms grows like ``sqrt(n)`` at best and ``n`` at worst. 64 buys that headroom
#: and still sits ORDERS below any residual a stalled-but-wrong Newton shows -- measured
#: 2026-09-12 at 1e8x on the planted stall, against a 64x allowance.
_FLOOR_SAFETY = 64.0


#: The machine epsilon the floor is written in (double precision -- the marcher is ``float64``).
_EPS = float(np.finfo(float).eps)


def same_instant(a: float, b: float) -> bool:
    """Join only arithmetic roundoff, not physically distinct clock instants."""
    return abs(a - b) <= 4 * max(abs(np.spacing(a)), abs(np.spacing(b)))


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
    u_abs = np.abs(np.asarray(u, dtype=float)).reshape(-1)
    if sp.issparse(tangent):
        # NEVER `.toarray()` here. `op_residual_floor` may densify because an operating point is a
        # circuit; this marcher also carries FEA/MEC tangents, where a dense copy is the whole mesh
        # squared. `abs()` keeps the sparse storage and the matvec stays O(nnz).
        magnitude = abs(tangent) @ u_abs
    else:
        dense = np.abs(np.asarray(tangent, dtype=float))
        magnitude = dense * u_abs if dense.ndim == 0 else dense @ u_abs
    return residual_floor_of_magnitude(magnitude)


def residual_floor_of_magnitude(magnitude: NDArray[np.floating]) -> float:
    """THE FLOOR RULE ITSELF, for a caller that builds ``|J| . |u|`` without ever forming ``J``.

    :func:`residual_floor` is this function applied to ``|J| . |u|``; the bordered field<->circuit
    system (:mod:`motronics.hamilton.coupling.monolithic`) can supply the magnitude but NOT the
    matrix -- ``J`` there is a 2x2 block system whose ``(n1, n1)`` corner is a 52185-DOF sparse
    field tangent, so an assembly is the defect that module exists to avoid. It builds the stacked
    magnitude from its four blocks and hands it here.

    ONE HOME FOR ``_FLOOR_SAFETY * _EPS * ||.|``, because a second spelling of it is exactly the
    drift :data:`~motronics.hamilton.coupling.monolithic._BACKTRACK_CAP`'s comment warns about: a
    criterion written twice is two criteria as soon as one of them is edited.
    """
    magnitude = np.asarray(magnitude, dtype=float).reshape(-1)
    return _FLOOR_SAFETY * _EPS * float(np.linalg.norm(magnitude)) if magnitude.size else 0.0


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
    """

    us: list[NDArray[np.floating]]
    t: float
    v: NDArray[np.floating] | None = None
    a: NDArray[np.floating] | None = None
    h: float | None = None
    stage: int = 0

    def copy(self) -> IntegratorState:
        """A deep copy -- a checkpoint must never alias the arrays a probe step goes on to touch."""
        return IntegratorState(
            us=[np.array(u, dtype=float, copy=True) for u in self.us],
            t=self.t,
            v=None if self.v is None else np.array(self.v, dtype=float, copy=True),
            a=None if self.a is None else np.array(self.a, dtype=float, copy=True),
            h=self.h,
            stage=self.stage,
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
        self, state: IntegratorState, u_new: NDArray[np.floating], dt: float, t_new: float
    ) -> IntegratorState: ...

    def restart(self, state: IntegratorState, u: NDArray[np.floating], t: float) -> IntegratorState:
        """Drop the derivative history at a discontinuity (an event), keeping what stays continuous."""
        ...


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
