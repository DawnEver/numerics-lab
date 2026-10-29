"""Mass-normalized rigid modes retain inertia, damping and applied loads."""

import numpy as np
import pytest

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.modal_transient import integrate_modal_transient


@pytest.mark.parametrize('scale', [1e-12, 1.0, 1e12])
def test_truncated_semidefinite_pencil_retains_free_translation_at_every_operator_scale(scale):
    result = integrate_modal_transient(
        np.diag([0.0, 4.0, 9.0]) * scale,
        np.eye(3) * scale,
        np.zeros((3, 3)),
        np.array([2.0, 0.0, 0.0]) * scale,
        TimeWindow(0.3, 0.4, 0.001, IntegratorKind.NEWMARK),
        n_modes=1,
        v0=np.array([0.2, 0.0, 0.0]),
    )
    elapsed = np.asarray(result.solution.times) - 0.3
    np.testing.assert_allclose(
        np.asarray(result.solution.states)[:, 0], 0.2 * elapsed + elapsed**2, rtol=1e-9, atol=1e-12
    )
    assert result.eigenvalues[0] == 0.0
    assert result.initial_projection_error < 1e-12


def test_roundoff_negative_rigid_eigenvalue_keeps_full_projected_damping():
    result = integrate_modal_transient(
        np.diag([-np.finfo(float).eps, 4.0]),
        np.eye(2),
        np.array([[0.2, 0.1], [0.1, 0.3]]),
        [1.0, 0.0],
        TimeWindow(0.0, 0.1, 0.001, IntegratorKind.NEWMARK),
        n_modes=2,
    )
    assert result.eigenvalues[0] == 0.0
    assert np.asarray(result.solution.states)[-1, 0] > 0.004
    assert abs(np.asarray(result.solution.states)[-1, 1]) > 1e-6


def test_materially_negative_stiffness_is_not_misreported_as_a_rigid_mode():
    with pytest.raises(ValueError, match='negative|semidefinite|unstable'):
        integrate_modal_transient(
            np.diag([-1.0, 4.0, 9.0]),
            np.eye(3),
            np.zeros((3, 3)),
            np.zeros(3),
            TimeWindow(0.0, 0.1, 0.001, IntegratorKind.NEWMARK),
            n_modes=1,
        )
