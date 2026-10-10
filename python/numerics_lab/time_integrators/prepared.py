"""Physics-free transaction authority for hidden prepared numerical stages.

Candidate tokens are fresh immutable handles, never mutable field arrays. Their owner's
numerical/material snapshots, not tokens or outer vectors, own the hidden history and
its portable checkpoint. A run rejects any reused pending or previously issued handle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from numpy.typing import NDArray

from motronics.contracts.spec.analysis_formulation import SemiDiscreteProblem
from motronics.gauss.time_integrators._forms import IntegratorState, ITimeIntegrator, StepForm


@dataclass(frozen=True)
class PreparedStepContext:
    integrator: ITimeIntegrator
    state: IntegratorState
    dt: float
    t_endpoint: float
    parent_candidate: object | None


@dataclass(frozen=True)
class PreparedStageContext:
    form: StepForm
    state: IntegratorState
    index: int


@dataclass(frozen=True)
class PreparedRestartContext:
    integrator: ITimeIntegrator
    state: IntegratorState


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


def restart_prepared(problem, integrator: ITimeIntegrator, state: IntegratorState, u: NDArray, t: float):
    restarted = integrator.restart(state, u, t)
    if isinstance(problem, PreparedStepProblem):
        problem.restart_step(PreparedRestartContext(integrator, restarted.copy()))
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
