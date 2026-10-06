"""Fresh-cycle shooting adapter for a stateless first-order semidiscrete problem."""

from dataclasses import dataclass, replace

import numpy as np

from motronics.contracts.spec.analysis_formulation import HistoryProblem
from motronics.contracts.spec.study_spec import IntegratorKind
from motronics.gauss.time_integrators import integrate_transient
from motronics.gauss.time_integrators.checkpoint import HistoryPresence
from motronics.gauss.time_integrators.shooting import periodic_shooting

__all__ = [
    'TransientShootingResult',
    'shoot_transient_cycle',
]


@dataclass(frozen=True)
class TransientShootingResult:
    state: np.ndarray
    residual_norm: float
    cycle_evaluations: int
    initial_history: tuple[np.ndarray, ...] | None
    state_scale: float


def shoot_transient_cycle(
    problem, study, initial, *, period, max_newton=100, step_tol=1e-12, atol=1e-9, rtol=1e-7, max_iterations=30, sign=1
):
    """Shoot a discrete BDF orbit with complete fresh numerical history every trial.

    BDF2 unknowns are [u(t0-dt),u(t0)] and its map returns both final states.
    Previous dt is the declared fixed dt, with history times t0-dt and t0.
    The producer supplies a stateless operator whose geometry and excitation both
    repeat over the declared period; moving meshes must preserve node identities.
    """
    if study.integrator not in (IntegratorKind.BDF1, IntegratorKind.BDF2):
        msg = 'shooting requires first-order BDF1 or BDF2 integration'
        raise ValueError(msg)
    if isinstance(problem, HistoryProblem) and (not isinstance(problem, HistoryPresence) or problem.has_history):
        msg = 'shooting cannot reuse mutable material history across trial cycles'
        raise ValueError(msg)
    steps = period / study.dt
    if (
        not np.isfinite(period)
        or period <= 0
        or not np.isclose(steps, round(steps), rtol=1e-12, atol=0)
        or round(steps) < 1
    ):
        msg = 'shooting period must contain a positive whole number of fixed steps'
        raise ValueError(msg)
    cycle_study = replace(study, t1=study.t0 + period)
    augmented = study.integrator is IntegratorKind.BDF2
    initial = np.asarray(initial, dtype=float)
    seed = np.tile(initial, 2) if augmented else initial

    def cycle_map(state):
        history = tuple(state.reshape(2, -1)) if augmented else None
        cycle = integrate_transient(
            problem,
            cycle_study,
            history[-1] if augmented else state,
            initial_history=history,
            max_newton=max_newton,
            tol=step_tol,
        )
        if not cycle.converged:
            msg = 'shooting trial cycle contains unconverged steps'
            raise RuntimeError(msg)
        return np.concatenate(cycle.states[-2:]) if augmented else cycle.states[-1]

    result = periodic_shooting(cycle_map, seed, atol=atol, rtol=rtol, max_iterations=max_iterations, sign=sign)
    history = tuple(result.state.reshape(2, -1)) if augmented else None
    return TransientShootingResult(
        history[-1] if augmented else result.state,
        result.residual_norm,
        result.cycle_evaluations,
        history,
        float(np.linalg.norm(result.state, ord=np.inf)),
    )
