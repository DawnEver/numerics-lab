"""The noise judgment assembles its zero-rate Jacobian only when some row CAN be noise (numerics-lab #55).

``_noise_judged`` replaces a row's bar with the system resolution only where the row is homogeneous AND its
operands are at or below that resolution (`row_bar`). Where no row's operands are that small the homogeneity
mask cannot change a single bar, yet every Newton iterate assembled a full Jacobian of the coupled system to
compute it (measured downstream 2026-11-18: 2885 Jacobian assemblies for 599 Newton iterations).

Why not cache it per STEP, as the issue first asked: the mask reads ``F(u, 0) - F_u(u) u`` at the ITERATE,
and a nonlinear row's intercept moves with ``u``; a per-step cache changes the verdict. The skip is exact:
the trajectories below are compared with ``==`` against the pre-change march. The planted control is a row
whose operands ARE under the resolution: there the Jacobian is still assembled and a homogeneous row is
still judged at the resolution while a forced one keeps its own bar.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from numerics_lab.formulation import DerivativeWeights, IntegratorKind, TimeWindow
from numerics_lab.linear.resolution import FLOOR_SAFETY, row_bar
from numerics_lab.time_integrators import integrate_transient
from numerics_lab.time_integrators._step import _noise_judged


class _Coupled:
    """A stiff decay, a cubic algebraic row and a nonlinear coupled row; counts zero-rate state Jacobians."""

    def __init__(self) -> None:
        self.zero_rate_state_jacobians = 0

    def residual(self, u, u_dot, u_ddot, t):
        return np.array([
            u_dot[0] + 1e3 * u[0] - 1e3 * np.cos(t),
            (u[1] - 2.0) ** 3 + u[1],
            u_dot[2] + u[2] ** 3 - u[0],
        ])

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        a0, a1 = weights.alpha0.real, weights.alpha1.real
        if a0 == 1 and a1 == 0 and not np.any(u_dot):
            self.zero_rate_state_jacobians += 1
        return np.array([
            [1e3 * a0 + a1, 0.0, 0.0],
            [0.0, (3.0 * (u[1] - 2.0) ** 2 + 1.0) * a0, 0.0],
            [-a0, 0.0, 3.0 * u[2] ** 2 * a0 + a1],
        ])


_STUDY = TimeWindow(t0=0.0, t1=5e-3, dt=1e-3, integrator=IntegratorKind.BDF2)

#: The pre-change march (fe3ddc8's `_noise_judged`, 12 zero-rate Jacobians), bit for bit, as ``float.hex``.
_BEFORE = [
    ['0x0.0p+0', '0x1.8000000000000p+1', '0x1.0000000000000p-1'],
    ['0x1.ffffef390876cp-2', '0x1.0000000000000p+0', '0x1.00311d7768c71p-1'],
    ['0x1.9999780baaf7bp-1', '0x1.0000000000000p+0', '0x1.007c6823bf32ap-1'],
    ['0x1.e147588429095p-1', '0x1.0000000000000p+0', '0x1.00dc9be1c6f3fp-1'],
    ['0x1.fbe6c36e0a668p-1', '0x1.0000000000000p+0', '0x1.0148453fb6cb1p-1'],
    ['0x1.016e716f355efp+0', '0x1.0000000000000p+0', '0x1.01b8e20e3adebp-1'],
]


def _march():
    problem = _Coupled()
    solution = integrate_transient(problem, _STUDY, np.array([0.0, 3.0, 0.5]), max_newton=50, row_acceptance=True)
    return problem, solution


def test_the_trajectory_is_bit_identical_to_the_pre_change_march() -> None:
    _problem, solution = _march()
    assert solution.converged
    assert [[float(v).hex() for v in state] for state in solution.states] == _BEFORE


def test_no_zero_rate_jacobian_is_assembled_where_no_row_can_be_noise() -> None:
    problem, _solution = _march()
    assert problem.zero_rate_state_jacobians == 0


def _form():
    return SimpleNamespace(
        t_eval=0.0,
        weights=DerivativeWeights(1.0, 0.0, 0.0),
        u_arg=lambda s: s,
        u_dot=np.zeros_like,
        u_ddot=np.zeros_like,
    )


class _Linear:
    def __init__(self, source) -> None:
        self.source, self.jacobians = np.asarray(source, dtype=float), 0

    def residual(self, u, _rate, _acceleration, _time):
        return np.asarray(u, dtype=float) - self.source

    def jacobian(self, _u, _rate, _acceleration, _time, weights):
        self.jacobians += 1
        return weights.alpha0 * np.eye(self.source.size)


def _full_mask(problem, u, bars, operands, resolution):
    """The pre-change judgment: the mask read for every row, whatever its operands."""
    jacobian = problem.jacobian(u, 0 * u, 0 * u, 0.0, DerivativeWeights(1, 0, 0))
    intercept = problem.residual(u, 0, 0, 0.0) - jacobian @ u
    homogeneous = np.abs(intercept) <= FLOOR_SAFETY * float(np.finfo(float).eps) * (np.abs(jacobian) @ np.abs(u))
    return row_bar(bars, operands, resolution, np.where(homogeneous, 0.0, 1.0))


def test_planted_control_a_row_under_the_resolution_is_still_judged() -> None:
    # Row 1 is homogeneous, row 2 carries a tiny DECLARED source: both sit under the 1e6 system's resolution.
    problem = _Linear([1e6, 0.0, 1e-60])
    u = np.array([1e6, 0.0, 0.0])
    bars, operands, resolution = np.array([1.0, 1e-80, 1e-80]), np.array([1e6, 1e-70, 1e-70]), 1e-9
    judged = _noise_judged(problem, _form(), u, bars, operands, resolution)
    assert problem.jacobians == 1
    assert judged.tolist() == [1.0, resolution, 1e-80]
    assert judged.tolist() == _full_mask(_Linear([1e6, 0.0, 1e-60]), u, bars, operands, resolution).tolist()


def test_no_row_under_the_resolution_leaves_every_bar_unread() -> None:
    problem = _Linear([1e6, 0.0, 1e-60])
    u = np.array([1e6, 0.0, 0.0])
    bars, operands, resolution = np.array([1.0, 2.0, 3.0]), np.array([1e6, 1.0, 1.0]), 1e-9
    judged = _noise_judged(problem, _form(), u, bars, operands, resolution)
    assert problem.jacobians == 0
    assert judged.tolist() == _full_mask(_Linear([1e6, 0.0, 1e-60]), u, bars, operands, resolution).tolist()
