"""A row's arithmetic floor counts the terms its tangent cannot see: its source and its rate.

``eps |J| |u|`` is the rounding of the state-proportional terms only. A winding row
``lambda' + R i - v`` whose linkage crosses zero while 512 V drives it has ``|J| |u| ~ 0``, yet
the row is a 512 V sum and cannot resolve below one ulp of 512 V. MEASURED 2026-10-22 on the
case_a dq-map march: the TR stage to t = 19.2116 ms held that row at 1.137e-13 V
(= ulp(512)) for 20 iterations against a tangent floor of 4.19e-14 V, and the march stopped.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from numerics_lab.formulation import DerivativeWeights
from numerics_lab.time_integrators._forms import _roundoff_floor, residual_floor_by_row
from numerics_lab.time_integrators._step import ProblemCapabilities, _at_roundoff_floor, _unseen_term_floor

_V, _R, _A1 = 512.3, 0.016, 3.4e6


class _Winding:
    """``lambda' + R lambda - V``: one row whose source and rate dwarf its state."""

    def residual(self, u, rate, _acceleration, _time):
        return np.array([rate[0] + _R * u[0] - _V])

    def jacobian(self, _u, _rate, _acceleration, _time, weights):
        return np.array([[weights.alpha1.real + weights.alpha0.real * _R]])


def _form():
    weights = DerivativeWeights(1.0, _A1, 0.0)
    return SimpleNamespace(
        t_eval=0.0, weights=weights, u_arg=lambda u: u, u_dot=lambda _u: np.array([_V]), u_ddot=np.zeros_like
    )


@pytest.mark.parametrize(('defect', 'accepted'), [(float(np.spacing(_V)), True), (1e-9, False)])
def test_a_source_driven_row_at_one_ulp_of_its_source_is_at_its_floor(defect, accepted):
    problem, form = _Winding(), _form()
    u = np.array([1e-12])  # the linkage crossing zero
    tangent = problem.jacobian(u, None, None, 0.0, form.weights)
    capabilities = ProblemCapabilities.resolve(problem)
    assert _at_roundoff_floor(problem, tangent, u, form, capabilities, np.array([defect])) is accepted


class _RefusesTheRestPoint(_Winding):
    """A constitutive law that raises at zero rate, as a law evaluated outside its domain does."""

    def residual(self, u, rate, _acceleration, _time):
        if not np.any(rate):
            msg = 'outside the constitutive domain at zero rate'
            raise ValueError(msg)
        return super().residual(u, rate, _acceleration, _time)


def test_a_law_that_refuses_the_rest_point_reads_as_an_unknowable_floor_not_a_crash():
    problem, form = _RefusesTheRestPoint(), _form()
    u = np.array([1e-12])
    tangent = problem.jacobian(u, None, None, 0.0, form.weights)
    capabilities = ProblemCapabilities.resolve(problem)
    assert _at_roundoff_floor(problem, tangent, u, form, capabilities, np.array([float(np.spacing(_V))])) is False


_ACCELERATION = 2.0e9


class _Inertial:
    """``m u'' + k u = 0``-like row whose acceleration term dwarfs its state: its floor is that term's rounding."""

    def residual(self, u, _rate, acceleration, _time):
        return np.array([acceleration[0] + 1e-3 * u[0]])

    def jacobian(self, _u, _rate, _acceleration, _time, weights):
        return np.array([[weights.alpha2.real + weights.alpha0.real * 1e-3]])


@pytest.mark.parametrize(('defect', 'accepted'), [(float(np.spacing(_ACCELERATION)), True), (1e-3, False)])
def test_a_second_order_rows_acceleration_sets_its_floor(defect, accepted):
    """PLANTED: without the ``F_u'' u''`` term this row's floor is ~0 and one ulp of its acceleration is refused."""
    problem = _Inertial()
    form = SimpleNamespace(
        t_eval=0.0,
        weights=DerivativeWeights(1.0, 0.0, 1.0),
        u_arg=lambda u: u,
        u_dot=np.zeros_like,
        u_ddot=lambda _u: np.array([_ACCELERATION]),
    )
    u = np.array([1e-12])
    tangent = problem.jacobian(u, None, None, 0.0, form.weights)
    capabilities = ProblemCapabilities.resolve(problem)
    assert _at_roundoff_floor(problem, tangent, u, form, capabilities, np.array([defect])) is accepted


class _CountsTheRestPoint(_Winding):
    """Counts the zero-rate residual reads ``F(u, 0)``: one judgment needs the source ONCE."""

    def __init__(self) -> None:
        self.rest_reads = 0

    def residual(self, u, rate, _acceleration, _time):
        self.rest_reads += not np.any(rate)
        return super().residual(u, rate, _acceleration, _time)


def test_a_refused_tangent_floor_reads_the_rows_source_once():
    """Both the unseen-term floor and the noise judgment read ``F(u, 0)``; it is one state and one time."""
    problem, form = _CountsTheRestPoint(), _form()
    u = np.array([1e-12])
    tangent = problem.jacobian(u, None, None, 0.0, form.weights)
    capabilities = ProblemCapabilities.resolve(problem)
    _at_roundoff_floor(problem, tangent, u, form, capabilities, np.array([float(np.spacing(_V))]))
    assert problem.rest_reads == 1


class _Counting(_Winding):
    """The winding, counting the Jacobians the floor assembles."""

    def __init__(self) -> None:
        self.assembled = []

    def jacobian(self, u, rate, acceleration, time, weights):
        self.assembled.append((weights.alpha0, weights.alpha1, weights.alpha2))
        return super().jacobian(u, rate, acceleration, time, weights)


def _unseen(problem, u_ddot):
    form = SimpleNamespace(
        t_eval=0.0, u_arg=lambda u: u, u_dot=lambda _u: np.array([_V]), u_ddot=lambda _u: np.array([u_ddot])
    )
    return _unseen_term_floor(problem, form, np.array([1e-12]), np.array([_V]))


def test_a_first_order_march_never_assembles_the_acceleration_jacobian():
    """#55: a zero acceleration adds eps|J|0 = 0, so its full Jacobian is not assembled -- the floor is unchanged."""
    problem = _Counting()
    floor = _unseen(problem, 0.0)
    assert problem.assembled == [(0, 1, 0)]
    rate = _Winding().jacobian(None, None, None, 0.0, DerivativeWeights(0, 1, 0))
    full = residual_floor_by_row(rate, np.array([_V])) + _roundoff_floor(np.array([_V]))
    np.testing.assert_array_equal(floor, full)


def test_a_second_order_march_still_assembles_both():
    """PLANTED CONTROL: a nonzero acceleration still assembles its Jacobian."""
    problem = _Counting()
    _unseen(problem, 2.0)
    assert problem.assembled == [(0, 1, 0), (0, 0, 1)]
