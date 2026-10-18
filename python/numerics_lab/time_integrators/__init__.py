"""Physics-blind time-integration runtime -- BDF / Newmark / generalized-alpha (plan R5).

This is the temporal counterpart of ``gauss/nonlinear`` (iteration strategies) and
``gauss/linear`` (linear-solve backends): a registry of pluggable TIME INTEGRATORS that a
transient study marches with, decoupled from any physics. The orthogonality hinge is
:class:`~numerics_lab.formulation.DerivativeWeights`: an analysis formulation
supplies ONLY the triple ``(alpha0, alpha1, alpha2)`` weighting the ``u`` / ``u_dot`` /
``u_ddot`` operators; a time integrator is exactly the machinery that PRODUCES those weights
per step (plus the history-dependent affine offsets) and advances a
:class:`~numerics_lab.formulation.SemiDiscreteProblem` one step.

The load-bearing idea: an implicit integrator writes each derivative at the new step as an
AFFINE function of the unknown ``u_{n+1}``,

    u_dot   = alpha1 * u_{n+1} + h_dot        (history)
    u_ddot  = alpha2 * u_{n+1} + h_ddot

so the residual root-find for ``u_{n+1}`` is driven purely through ``residual`` /
``jacobian(weights)`` -- the integrator never sees the physics, only these numbers. The
``alpha1`` / ``alpha2`` slopes ARE the :class:`DerivativeWeights` a study would carry; the
``h_dot`` / ``h_ddot`` offsets are the history the backend cannot know. First-order schemes
(BDF1/BDF2) leave ``alpha2 = 0`` (the eddy-current / diffusion form ``u_dot``); second-order
schemes (Newmark, generalized-alpha) carry ``alpha2`` (the structural ``u_ddot``).

This module imports ONLY its own engine + numpy/scipy (enforced by
``test_import_purity.py::test_importing_gauss_pulls_no_physics_into_sys_modules``); it touches no
mesh, no material, no solver.
"""

from __future__ import annotations

from numerics_lab.time_integrators._forms import (
    IntegratorState,
    ITimeIntegrator,
    StepForm,
    TransientSolution,
    residual_floor,
    residual_floor_of_magnitude,
)
from numerics_lab.time_integrators._integrators import (
    INTEGRATOR_REGISTRY,
    BDF1Integrator,
    BDF2Integrator,
    GeneralizedAlphaIntegrator,
    NewmarkIntegrator,
    TrapezoidIntegrator,
    TRBDF2Integrator,
    get_integrator,
)
from numerics_lab.time_integrators._step import (
    HELD_STALL_RATIO,
    HeldFactor,
    TrialNotEvaluable,
    _solve_step,
)
from numerics_lab.time_integrators.explicit import integrate_explicit_sampled
from numerics_lab.time_integrators.factor_reuse import FactorCache
from numerics_lab.time_integrators.instants import canonical, is_future, same_instant
from numerics_lab.time_integrators.march import (
    BudgetExceeded,
    EventSurface,
    RunBudget,
    StepNotConverged,
    StepTolerance,
    integrate_transient,
)
from numerics_lab.time_integrators.multirate import (
    Hold,
    RateTransition,
    SampledTask,
    ScheduledTask,
    base_sample_time,
    integrate_sampled,
    sample_hits,
    schedule,
)

__all__ = [
    'HELD_STALL_RATIO',
    'INTEGRATOR_REGISTRY',
    'BDF1Integrator',
    'BDF2Integrator',
    'BudgetExceeded',
    'EventSurface',
    'FactorCache',
    'GeneralizedAlphaIntegrator',
    'HeldFactor',
    'Hold',
    'ITimeIntegrator',
    'IntegratorState',
    'NewmarkIntegrator',
    'RateTransition',
    'RunBudget',
    'SampledTask',
    'ScheduledTask',
    'StepForm',
    'StepNotConverged',
    'StepTolerance',
    'TRBDF2Integrator',
    'TransientSolution',
    'TrapezoidIntegrator',
    'TrialNotEvaluable',
    '_solve_step',
    'base_sample_time',
    'canonical',
    'get_integrator',
    'integrate_explicit_sampled',
    'integrate_sampled',
    'integrate_transient',
    'is_future',
    'residual_floor',
    'residual_floor_of_magnitude',
    'same_instant',
    'sample_hits',
    'schedule',
]
