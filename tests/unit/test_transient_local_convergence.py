"""Residual convergence cannot hide a small electrical unknown behind voltage scale."""

import numpy as np
import pytest

from numerics_lab.time_integrators import BDF1Integrator, _solve_step


class _MixedUnits:
    def residual(self, u, u_dot, u_ddot, t):
        return np.array([u[0] - 1e6, 1e-12 * (u[1] - 1e-9)])

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.diag([1.0, 1e-12])

    def local_convergence_fields(self, u, t):
        return (u[:1], u[1:])


def _step(max_newton, tol=1e-12, gmin_stages=None, problem=None, guess=None):
    problem = _MixedUnits() if problem is None else problem
    guess = np.array([1e6, 0.0]) if guess is None else guess
    integrator = BDF1Integrator()
    form = integrator.prepare(integrator.start(guess, 0.0), 0.1, 0.1)
    return _solve_step(
        problem,
        form,
        guess,
        solve=np.linalg.solve,
        max_newton=max_newton,
        tol=tol,
        tol_rel=1e-6,
        gmin_stages=gmin_stages,
    )


def test_small_current_is_not_hidden_by_large_voltage_or_global_residual():
    state, converged, _, _, _ = _step(4)
    assert converged
    assert state[1] == pytest.approx(1e-9, rel=1e-10, abs=0.0)


@pytest.mark.parametrize(('tol', 'gmin_stages'), [(1e-12, None), (1e-30, None), (1e-30, (1e-12,))])
def test_local_check_is_not_bypassed_by_ceiling_roundoff_or_gmin(tol, gmin_stages):
    _, converged, _, _, _ = _step(1, tol, gmin_stages)
    assert not converged


@pytest.mark.parametrize(('current_atol', 'accepted'), [(2e-9, True), (5e-10, False)])
def test_declared_current_variation_threshold_is_not_a_voltage_threshold(current_atol, accepted):
    problem = _MixedUnits()
    problem.local_convergence_atols = (1.0, current_atol)
    _, converged, _, _, _ = _step(1, problem=problem)
    assert converged is accepted


@pytest.mark.parametrize('bad', [0.0, -1.0, np.inf, True])
def test_absolute_local_threshold_must_be_finite_positive_and_nonboolean(bad):
    problem = _MixedUnits()
    problem.local_convergence_atols = (None, bad)
    with pytest.raises(ValueError, match='positive|threshold'):
        _step(4, problem=problem)


def test_exact_residual_cannot_accept_nonfinite_local_fields():
    class InvalidFields(_MixedUnits):
        def local_convergence_fields(self, u, t):
            return (np.array([np.nan]),)

    _, converged, _, _, _ = _step(1, problem=InvalidFields(), guess=np.array([1e6, 1e-9]))
    assert not converged


class _ZeroFieldNoise:
    """A physically ZERO field ``a`` re-solved to fresh rounding noise every evaluation.

    Row 1 sums ``a`` with an ampere-scale coupling term, so a change of ``a`` at ``noise`` is either
    below that row's rounding (``noise`` 1e-40) or a real, resolvable change (``noise`` 1e-3).
    """

    def __init__(self, noise) -> None:
        self.noise, self.calls = noise, 0

    def residual(self, u, u_dot, u_ddot, t):
        self.calls += 1
        return np.array([u[0] - 1.0, (u[0] - 1.0) + (u[1] - self.noise * self.calls)])

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.array([[1.0, 0.0], [1.0, 1.0]])

    def local_convergence_fields(self, u, t):
        return (u[:1], u[1:])


@pytest.mark.parametrize(('noise', 'accepted'), [(1e-40, True), (1e-3, False)])
def test_a_zero_field_changing_below_its_rows_rounding_is_converged(noise, accepted):
    # MEASURED 2026-10-13 on case_l at rest: nodal A at 1e-40 Wb/m changed by
    # 0.03..2 RELATIVE to its own noise every iteration, so the march halved to its minimum step.
    _, converged, _, _, _ = _step(8, tol=1e-12, problem=_ZeroFieldNoise(noise), guess=np.array([0.0, 0.0]))
    assert converged is accepted
