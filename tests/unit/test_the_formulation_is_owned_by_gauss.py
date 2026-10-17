"""The numerical half of a time-marched problem is owned by `gauss.formulation` (plan Track B0.1).

The weights, the integrator kind, the time window and the semi-discrete problem protocols are
statements about NUMERICS, not about a motor, so the engine owns them and contracts imports them.
"""

from __future__ import annotations

import pytest

import motronics.contracts
from motronics.contracts.spec import analysis_formulation, study_spec
from motronics.gauss import formulation

_OWNED = (
    'NEWMARK_BETA',
    'NEWMARK_GAMMA',
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
)


def test_formulation_publishes_every_owned_name() -> None:
    assert set(_OWNED) <= set(formulation.__all__)


def test_contracts_does_not_republish_an_owned_name() -> None:
    republished = set(_OWNED) & (
        set(motronics.contracts.__all__) | set(analysis_formulation.__all__) | set(study_spec.__all__)
    )
    assert republished == set()


def test_a_transient_study_is_a_time_window() -> None:
    study = study_spec.TransientStudy(t0=0.0, t1=1.0, dt=0.5)
    assert isinstance(study, formulation.TimeWindow)
    assert study.integrator is formulation.IntegratorKind.BDF2


@pytest.mark.parametrize(('t0', 't1', 'dt'), [(0.0, 1.0, 0.0), (1.0, 0.0, 0.1)])
def test_a_window_refuses_a_bad_step_or_span(t0: float, t1: float, dt: float) -> None:
    with pytest.raises(ValueError, match='transient'):
        formulation.TimeWindow(t0=t0, t1=t1, dt=dt)


def test_a_valid_window_is_accepted() -> None:
    # planted control: the refusal above is not a refusal of every window
    assert formulation.TimeWindow(t0=0.0, t1=1.0, dt=0.1).dt == 0.1
