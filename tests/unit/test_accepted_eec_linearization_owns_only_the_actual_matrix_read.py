"""UNRUN controls: EEC reads accepted jets, not a whole-march phase or trial guess."""

import asyncio
from concurrent.futures import CancelledError
from contextlib import contextmanager

import numpy as np
import pytest

from numerics_lab.formulation import DerivativeWeights, IntegratorKind, TimeWindow
from numerics_lab.time_integrators import integrate_transient
from numerics_lab.time_integrators.periodic import (
    accepted_jacobian,
    explicit_error_correction,
    nonlinear_error_correction,
)


class _ReadOwner:
    def __init__(self) -> None:
        self.active = False
        self.calls = []
        self.failure = None
        self.matrix = np.array([[2.0]])

    def residual(self, *_args):
        msg = 'this control owns a matrix read, not a numerical solve'
        raise AssertionError(msg)

    @contextmanager
    def accepted_linearization_scope(self, time):
        assert not self.active
        self.calls.append(('enter', time))
        self.active = True
        try:
            yield
        finally:
            self.active = False
            self.calls.append(('exit', time))

    def jacobian(self, state, rate, acceleration, time, weights):
        assert self.active
        self.calls.append(('jacobian', state, rate, acceleration, time, weights))
        if self.failure is not None:
            raise self.failure
        return self.matrix


def test_scope_preserves_actual_arguments_and_matrix_identity_then_exits_immediately():
    owner = _ReadOwner()
    state, rate, acceleration = np.array([1.0]), np.array([3.0]), np.array([5.0])
    weights = DerivativeWeights(2j, 3j, 4j)
    matrix = accepted_jacobian(owner, state, rate, acceleration, 7.0, weights)
    assert matrix is owner.matrix
    assert owner.calls[0] == ('enter', 7.0) and owner.calls[-1] == ('exit', 7.0)
    actual = owner.calls[1]
    assert actual[1] is state and actual[2] is rate and actual[3] is acceleration
    assert actual[4] == 7.0 and actual[5] is weights
    assert not owner.active


@pytest.mark.parametrize(
    'failure', [RuntimeError('matrix'), KeyboardInterrupt(), SystemExit(), CancelledError(), asyncio.CancelledError()]
)
def test_matrix_failure_and_typed_stop_leave_scope_without_replacement_or_cause_loss(failure):
    owner, cause = _ReadOwner(), ValueError('original cause')
    owner.failure, failure.__cause__ = failure, cause
    with pytest.raises(type(failure)) as raised:
        accepted_jacobian(owner, np.zeros(1), np.zeros(1), np.zeros(1), 1.0, DerivativeWeights())
    assert raised.value is failure and failure.__cause__ is cause
    assert not owner.active and owner.calls[-1] == ('exit', 1.0)


def test_missing_optional_capability_keeps_original_jacobian_path_and_identity():
    class Ordinary:
        def jacobian(self, state, rate, acceleration, time, weights):
            assert state is rate is acceleration
            assert time == 1.0 and weights is actual_weights
            return matrix

    matrix, actual_weights, zero = np.eye(1), DerivativeWeights(), np.zeros(1)
    assert accepted_jacobian(Ordinary(), zero, zero, zero, 1.0, actual_weights) is matrix
    assert not hasattr(Ordinary(), 'accepted_linearization_scope')


def test_explicit_and_nonlinear_correction_reads_are_scoped_but_the_solve_is_not():
    owner, current, previous = _ReadOwner(), np.array([3.0]), np.array([1.0])

    def solve(matrix, value):
        assert not owner.active, 'the matrix scope must not extend across the algorithmic correction solve'
        return np.linalg.solve(matrix, value)

    explicit_error_correction(owner, current, previous, 1.0, 2.0, solve=solve)
    reads = [call for call in owner.calls if call[0] == 'jacobian']
    assert len(reads) == 2
    assert reads[0][-1] == DerivativeWeights(alpha0=1.0)
    assert reads[1][-1] == DerivativeWeights(alpha0=0.0, alpha1=1.0)
    assert reads[0][2] is reads[1][2] is reads[0][3] is reads[1][3]
    owner.calls.clear()
    nonlinear_error_correction(owner, current, previous, 2.0, np.eye(1), solve=solve)
    assert [call[0] for call in owner.calls] == ['enter', 'jacobian', 'exit']


class _Trial:
    def __init__(self, owner) -> None:
        self.owner = owner

    def residual(self, *args):
        assert not self.owner.active
        return self.owner.residual(*args)

    def jacobian(self, state, rate, acceleration, time, weights):
        assert not self.owner.active
        self.owner.trial_calls += 1
        return self.owner.matrix_at(state, weights)


class _Cubic:
    def residual(self, state, rate, acceleration, time):
        return 2.0 * rate + 0.4 * state + 0.2 * state**3 - np.cos(2 * np.pi * time)

    def matrix_at(self, state, weights):
        return np.eye(1) * (2.0 * weights.alpha1.real + weights.alpha0.real * (0.4 + 0.6 * state[0] ** 2))

    def jacobian(self, state, rate, acceleration, time, weights):
        return self.matrix_at(state, weights)


class _ScopedCubic(_Cubic):
    def __init__(self) -> None:
        self.active = False
        self.accepted_time = None
        self.accepted_reads, self.trial_calls = [], 0

    def step_problem(self, _time):
        return _Trial(self)

    def accept(self, time, _state):
        assert not self.active
        self.accepted_time = time
        return False

    @contextmanager
    def accepted_linearization_scope(self, time):
        assert not self.active and self.accepted_time == time
        self.active = True
        try:
            yield
        finally:
            self.active = False

    def jacobian(self, state, rate, acceleration, time, weights):
        assert self.active and self.accepted_time == time
        self.accepted_reads.append((time, weights))
        return self.matrix_at(state, weights)


def test_real_march_scopes_cycle_accumulation_and_correction_but_never_trial_newton():
    window = TimeWindow(0.0, 0.75, 0.25, IntegratorKind.BDF1)
    problem = _ScopedCubic()
    actual = integrate_transient(
        problem,
        window,
        np.array([0.7]),
        max_newton=30,
        eec_period=0.5,
        eec_nonlinear_first_order=True,
        on_accept=problem.accept,
    )
    ordinary = integrate_transient(
        _Cubic(),
        window,
        np.array([0.7]),
        max_newton=30,
        eec_period=0.5,
        eec_nonlinear_first_order=True,
    )
    assert not actual.nonconverged_times and actual.corrected_times == [0.5]
    assert problem.trial_calls > 0 and not problem.active
    assert len(problem.accepted_reads) == len(actual.times) - 1 + len(actual.corrected_times)
    np.testing.assert_allclose(actual.states, ordinary.states, rtol=1e-13, atol=1e-13)
