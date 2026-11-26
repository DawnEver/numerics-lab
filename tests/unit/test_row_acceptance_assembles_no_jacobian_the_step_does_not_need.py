"""A per-row verdict assembles no Jacobian beyond the step's own tangent and ONE state Jacobian (numerics-lab #55).

The verdict's scale reads ``|F_u| |u| + |F_u'| |u'| + |F(u, 0)|``. It used to assemble ``F_u`` and ``F_u'`` for
the scale, ``F_u`` AGAIN for the homogeneity screen, and the Newton step then assembled the weighted tangent
``F_u + alpha1 F_u'`` at the very same iterate: up to four full Jacobians of the coupled system where two carry
every fact (measured downstream 2026-11-18: 2885 assemblies for 599 Newton iterations, ~38 % of a coupled march).

The verdict now assembles the step's tangent ``J`` (handed on to the step) and ``F_u``; the rate Jacobian is
``(J - F_u) / alpha1``. The planted control is the count itself: the old verdict reads at least one more per
refused iterate and fails ``<= 2`` here. The acceptance is unchanged: the bit-exact golden of
``test_noise_judgment_assembles_only_when_a_row_can_be_noise`` and the row-verdict suites still hold.
"""

from __future__ import annotations

import numpy as np
import pytest

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import integrate_transient


class _Coupled:
    """A stiff decay, a cubic algebraic row and a nonlinear coupled row; counts EVERY Jacobian assembly."""

    def __init__(self) -> None:
        self.jacobians = 0

    def residual(self, u, u_dot, u_ddot, t):
        return np.array([
            u_dot[0] + 1e3 * u[0] - 1e3 * np.cos(t),
            (u[1] - 2.0) ** 3 + u[1],
            u_dot[2] + u[2] ** 3 - u[0],
        ])

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        self.jacobians += 1
        a0, a1 = weights.alpha0.real, weights.alpha1.real
        return np.array([
            [1e3 * a0 + a1, 0.0, 0.0],
            [0.0, (3.0 * (u[1] - 2.0) ** 2 + 1.0) * a0, 0.0],
            [-a0, 0.0, 3.0 * u[2] ** 2 * a0 + a1],
        ])


@pytest.mark.parametrize('integrator', [IntegratorKind.BDF1, IntegratorKind.BDF2])
def test_a_newton_iterate_assembles_at_most_two_jacobians(integrator) -> None:
    problem = _Coupled()
    study = TimeWindow(t0=0.0, t1=5e-3, dt=1e-3, integrator=integrator)
    solution = integrate_transient(problem, study, np.array([0.0, 3.0, 0.5]), max_newton=30, row_acceptance=True)
    assert solution.converged
    assert solution.newton_iterations > 0
    assert problem.jacobians <= 2 * solution.newton_iterations, (problem.jacobians, solution.newton_iterations)
