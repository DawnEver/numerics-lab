"""A step refused by the per-row bar NAMES the row: index, residual, bar, its scale and floor, and its owner.

THE GAP (a downstream repo 2026-10-24, the case_a native ladder): a refinement refused its first step with
"residual 4.0e-10" -- a NORM, which says nothing about the per-row bar that refused it, nor which row.
Deciding whether a bar is mis-scaled or a row is ill-conditioned needs that row by name.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from numerics_lab.formulation import DerivativeWeights
from numerics_lab.time_integrators._step import RowVerdict, _solve_step


class _Problem:
    """Two decoupled rows; the second is driven by a solver that never moves it."""

    def residual(self, state, _rate, _acceleration, _time):
        return np.array([state[0] - 1.0, state[1] - 5.0])

    def jacobian(self, _state, _rate, _acceleration, _time, weights):
        return weights.alpha0 * np.eye(2)

    def row_owner(self, index):
        return f'block-{"ab"[index]}'


_FORM = SimpleNamespace(
    t_eval=0.0,
    weights=DerivativeWeights(1.0, 0.0, 0.0),
    u_arg=lambda state: state,
    u_dot=np.zeros_like,
    u_ddot=np.zeros_like,
)


def _stuck_on_row_one(matrix, rhs):
    out = np.linalg.solve(matrix, rhs)
    out[1] = 0.0
    return out


def _solve(problem, report):
    return _solve_step(
        problem,
        _FORM,
        np.zeros(2),
        solve=_stuck_on_row_one,
        max_newton=2,
        tol=1e-50,
        row_acceptance=True,
        row_report=report,
    )


def test_a_refused_step_reports_the_row_over_its_bar_by_the_most():
    report = []
    _u, converged, _it, _residual, _floor = _solve(_Problem(), report)
    assert not converged
    verdict = report[-1]
    assert isinstance(verdict, RowVerdict)
    assert verdict.row == 1
    assert verdict.owner == 'block-b'
    assert verdict.residual == pytest.approx(5.0)
    assert verdict.ratio == pytest.approx(verdict.residual / verdict.bar)
    assert verdict.bar == pytest.approx(max(verdict.own_bar, verdict.floor))
    assert 'row 1' in str(verdict)
    assert 'block-b' in str(verdict)


def test_a_problem_without_an_owner_hook_still_names_the_index():
    class Bare(_Problem):
        row_owner = None

    report = []
    _solve(Bare(), report)
    assert report[-1].row == 1
    assert report[-1].owner is None


def test_the_refusal_message_carries_the_row():
    from numerics_lab.time_integrators.march import StepNotConverged  # noqa: PLC0415

    verdict = RowVerdict(row=7, owner='field', residual=4e-10, bar=1e-10, own_bar=1e-10, floor=1e-12, scale=1.0)
    error = StepNotConverged(1e-6, 4e-10, np.zeros(1), 0.0, np.zeros(1), worst_row=verdict)
    assert 'row 7' in str(error)
    assert error.worst_row is verdict
