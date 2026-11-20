"""The three optional engine hooks, mirrored one for one in `crates/numerics/src/nonlinear/tests/hooks.rs`.

A reported ``state_change`` the convergence rule requires settled; a ``TrialNotEvaluable`` a line search
shrinks past (and every other strategy aborts on); a ``factorize`` the held-tangent Newtons reuse. Each acts
when used, and an unused hook leaves the result unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from numerics_lab.line_search import TrialNotEvaluable
from numerics_lab.nonlinear import ITERATION_REGISTRY, AssembledState, get_strategy

#: Every strategy the fixture can run (`coefficient-damped` needs the raw pre-fold pair).
STRATEGIES = sorted(n for n in ITERATION_REGISTRY if n != 'coefficient-damped')
LINE_SEARCHED = ('newton', 'newton-linesearch', 'newton-held-factor', 'modified-newton')
HELD = ('newton-held-factor', 'modified-newton')


class _Affine:
    """``g(x) = slope x + b``; residual ``x - g(x)``; a diagonal tangent that may be deliberately off."""

    is_linear = False

    def __init__(self, slope, b, *, tangent=None, state_change=lambda _k: 0.0, unevaluable=lambda _x: False) -> None:
        self.slope = np.asarray(slope, float)
        self.b = np.asarray(b, float)
        self.diag = 1.0 - self.slope if tangent is None else np.asarray(tangent, float)
        self.state_change = state_change
        self.unevaluable = unevaluable
        self.assembled = 0
        self.max_seen = -np.inf
        self.system_solves = 0

    def initial_guess(self):
        return np.zeros(self.b.size)

    def assemble(self, x, /):
        x = np.asarray(x, float)
        self.max_seen = max(self.max_seen, float(np.max(x)))
        if self.unevaluable(x):
            msg = f'x = {x}'
            raise TrialNotEvaluable(msg)
        change = self.state_change(self.assembled)
        self.assembled += 1
        return AssembledState(
            system_matrix=None, total_energy=float(np.sum(x**2)), payload=x.copy(), state_change=change
        )

    def solve_linear(self, state, /):
        return self.slope * state.payload + self.b

    def residual(self, x, state, /):
        x = np.asarray(x, float)
        return x - (self.slope * x + self.b)

    def residual_norm(self, x, state, /):
        return float(np.linalg.norm(self.residual(x, state)))

    def tangent(self, x, state, /):
        return self.diag

    def solve_system(self, matrix, rhs, /):
        self.system_solves += 1
        return np.asarray(rhs, float) / np.asarray(matrix, float)

    def solve_secant(self, state, rhs, /):
        return np.asarray(rhs, float)

    def on_iteration(self, i, state, err, alpha, increment, /):
        pass


class _Factorizing(_Affine):
    """The same system offering ``factorize``: one factor per tangent, counted with its solves."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.factorized = 0
        self.factored_solves = 0

    def factorize(self, matrix, /):
        self.factorized += 1
        inv = 1.0 / np.asarray(matrix, float)

        def solve(rhs):
            self.factored_solves += 1
            return np.asarray(rhs, float) * inv

        return solve


def _run(name, system, max_iter=60):
    return get_strategy(name).solve(system, max_iter=max_iter)


def _affine(**kw):
    return _Affine([0.5, 0.25], [1.0, -2.0], **kw)


# -- 1. state_change ----------------------------------------------------------------------------


@pytest.mark.parametrize('name', STRATEGIES)
def test_a_state_change_above_the_bar_blocks_convergence(name):
    assert _run(name, _affine()).converged, 'PLANTED CONTROL: the baseline converges'
    assert not _run(name, _affine(state_change=lambda _k: 1.0)).converged


@pytest.mark.parametrize('name', STRATEGIES)
def test_a_settled_state_change_reproduces_the_unreported_run(name):
    base = _run(name, _affine())
    res = _run(name, _affine(state_change=lambda _k: 1e-9))
    assert res.iterations == base.iterations
    np.testing.assert_array_equal(res.x, base.x)


# -- 2. TrialNotEvaluable -----------------------------------------------------------------------


def _overshooting(unevaluable=lambda x: x[0] > 3.0):
    # Root 2; a tangent 0.25 instead of 0.5 sends the full step from 0 to 4.
    return _Affine([0.5], [1.0], tangent=[0.25], unevaluable=unevaluable)


@pytest.mark.parametrize('name', LINE_SEARCHED)
def test_a_line_search_shrinks_past_an_unevaluable_trial(name):
    control = _overshooting(unevaluable=lambda _x: False)
    _run('newton', control, 50)
    assert control.max_seen > 3.0, 'PLANTED CONTROL: the full step lands in the refused region'
    res = _run(name, _overshooting(), 50)
    assert res.converged
    assert res.x[0] == pytest.approx(2.0, abs=1e-6)


@pytest.mark.parametrize('name', LINE_SEARCHED)
def test_a_line_search_with_no_evaluable_trial_refuses(name):
    with pytest.raises(TrialNotEvaluable):
        _run(name, _overshooting(unevaluable=lambda x: x[0] > 1e-300), 50)


@pytest.mark.parametrize('name', [n for n in STRATEGIES if n not in LINE_SEARCHED and n != 'continuation'])
def test_a_strategy_without_a_line_search_aborts(name):
    with pytest.raises(TrialNotEvaluable):
        _run(name, _overshooting(unevaluable=lambda x: x[0] > 0.5), 50)


# -- 3. held factor -----------------------------------------------------------------------------


def _modified(cls):
    # A tangent 20% off contracts every step linearly, so a held factor serves several iterations.
    return cls([0.5], [1.0], tangent=[0.4])


@pytest.mark.parametrize('name', HELD)
def test_a_held_factor_is_factorized_once_and_reused(name):
    system = _modified(_Factorizing)
    assert _run(name, system, 50).converged
    assert system.factored_solves > system.factorized > 0
    assert system.system_solves == 0


@pytest.mark.parametrize('name', HELD)
def test_a_held_factor_matches_refactorisation(name):
    plain = _modified(_Affine)
    base = _run(name, plain, 50)
    assert plain.system_solves > 0, 'PLANTED CONTROL: the plain run solves through solve_system'
    res = _run(name, _modified(_Factorizing), 50)
    assert res.iterations == base.iterations
    np.testing.assert_allclose(res.x, base.x, rtol=0.0, atol=1e-12)


def test_full_newton_does_not_factorize():
    system = _modified(_Factorizing)
    _run('newton', system, 50)
    assert system.factorized == 0
