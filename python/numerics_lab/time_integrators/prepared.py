"""Physics-free transaction authority for hidden prepared numerical stages.

Candidate tokens are fresh immutable handles, never mutable field arrays. Their owner's
numerical/material snapshots, not tokens or outer vectors, own the hidden history and
its portable checkpoint. A run rejects any reused pending or previously issued handle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from numpy.typing import NDArray

from numerics_lab.formulation import SemiDiscreteProblem
from numerics_lab.time_integrators._forms import (
    IntegratorState,
    ITimeIntegrator,
    ProjectedRestartIntegrator,
    StepForm,
)
from numerics_lab.time_integrators._landing import AcceptedProjectionResult

__all__ = [
    'PreparedCandidates',
    'PreparedRestartContext',
    'PreparedStageContext',
    'PreparedStepAttempt',
    'PreparedStepContext',
    'PreparedStepProblem',
    'finished_candidate',
    'prepared_stage',
    'restart_prepared',
]


@dataclass(frozen=True)
class PreparedStepContext:
    integrator: ITimeIntegrator
    state: IntegratorState
    dt: float
    t_endpoint: float
    parent_candidate: object | None


@dataclass(frozen=True)
class PreparedStageContext:
    """Actual scheme clock and declared incoming endpoint side.

    ``endpoint_left`` applies only to the stage evaluated at the declared cut
    endpoint. The prepared owner binds its genuine incoming law there; earlier
    stages keep their actual clocks and regular right-side evaluation.
    """

    form: StepForm
    state: IntegratorState
    index: int
    endpoint_left: bool = False


@dataclass(frozen=True)
class PreparedRestartContext:
    integrator: ITimeIntegrator
    state: IntegratorState
    prepared_restart: object | None = None


@runtime_checkable
class PreparedStepAttempt(Protocol):
    def prepare_stage(self, context: PreparedStageContext) -> SemiDiscreteProblem: ...

    def stage_solved(self, u: NDArray) -> None: ...

    def finish(self) -> object: ...

    def abort(self) -> None: ...


@runtime_checkable
class PreparedStepProblem(Protocol):
    def begin_step(self, context: PreparedStepContext) -> PreparedStepAttempt: ...

    def accept_step(self, candidate: object) -> None: ...

    def reject_step(self, candidate: object) -> None: ...

    def restart_step(self, context: PreparedRestartContext) -> None: ...


def restart_prepared(
    problem, integrator: ITimeIntegrator, state: IntegratorState, u: NDArray | AcceptedProjectionResult, t: float
):
    projection = u if isinstance(u, AcceptedProjectionResult) else None
    if (
        projection is not None
        and projection.prepared_restart is not None
        and not isinstance(problem, PreparedStepProblem)
    ):
        msg = 'prepared restart handle requires its actual prepared owner'
        raise TypeError(msg)
    if projection is not None and isinstance(integrator, ProjectedRestartIntegrator):
        restarted = integrator.restart_at_pair(
            state,
            projection.state.copy(),
            t,
            None if projection.rate is None else projection.rate.copy(),
            None if projection.acceleration is None else projection.acceleration.copy(),
        )
    elif projection is not None and (projection.rate is not None or projection.acceleration is not None):
        msg = 'actual integrator must own explicit projected derivative restart'
        raise TypeError(msg)
    else:
        restarted = integrator.restart(state, projection.state.copy() if projection is not None else u, t)
    if isinstance(problem, PreparedStepProblem):
        problem.restart_step(
            PreparedRestartContext(
                integrator, restarted.copy(), None if projection is None else projection.prepared_restart
            )
        )
    return restarted


def prepared_stage(attempt: PreparedStepAttempt, context: PreparedStageContext) -> SemiDiscreteProblem:
    problem = attempt.prepare_stage(context)
    if not isinstance(problem, SemiDiscreteProblem):
        msg = 'prepared stage owner must return a semi-discrete problem'
        raise TypeError(msg)
    return problem


def finished_candidate(
    attempt: PreparedStepAttempt, parent: object | None, registry: PreparedCandidates | None
) -> object:
    candidate = attempt.finish()
    if candidate is None:
        msg = 'prepared step must return an explicit candidate token'
        raise ValueError(msg)
    if candidate is parent:
        msg = 'prepared step must return a fresh distinct candidate, not its parent'
        raise ValueError(msg)
    if registry is not None:
        registry.add(candidate, parent)
    return candidate


class PreparedCandidates:
    """Explicit token ancestry; discarded probes cannot replace a selected outcome."""

    def __init__(self, owner: PreparedStepProblem) -> None:
        self.owner = owner
        self.pending: list[tuple[object, object | None]] = []
        # O(total attempts) handle retention prevents identity reuse; lookup is O(1).
        # Owners release rejected snapshot payloads in reject_step, not through handles.
        self.issued: dict[int, object] = {}

    def add(self, candidate: object, parent: object | None) -> None:
        if id(candidate) in self.issued:
            msg = 'prepared step must return a fresh distinct candidate, not a previously issued handle'
            raise ValueError(msg)
        self.issued[id(candidate)] = candidate
        self.pending.append((candidate, parent))

    def accept(self, selected: object) -> None:
        self.owner.accept_step(selected)
        descendants = [selected]
        keep = []
        pending, self.pending = self.pending, []
        for candidate, parent in pending:
            if candidate is selected:
                continue
            if any(parent is ancestor for ancestor in descendants):
                descendants.append(candidate)
                keep.append((candidate, parent))
            else:
                self.owner.reject_step(candidate)
        self.pending = keep

    def close(self) -> None:
        pending, self.pending = self.pending, []
        for candidate, _parent in pending:
            self.owner.reject_step(candidate)
