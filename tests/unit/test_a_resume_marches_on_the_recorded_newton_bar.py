"""A checkpoint records the march's Newton bar, and a resume marches on it -- not on one its caller recomputes.

A caller sizes the bar off its problem's load (`hamilton.studies.transient._constrained.step_tolerance`). On a
resume it can only read that load at the resume instant, and through the restored material history: under a
cosine drive or a hysteretic law that is another number, and a Newton stopping on another bar is another
trajectory. The bar the straight march stopped on is part of its state, so the record carries it.
"""

import json

import numpy as np
import pytest

from numerics_lab.formulation import IntegratorKind, TimeWindow
from numerics_lab.time_integrators import integrate_transient
from numerics_lab.time_integrators.checkpoint import TransientCheckpoint

pytestmark = pytest.mark.unit


class _Cubic:
    """``u' + u + u^3 = cos(5 t)``: Newton needs several iterations, so where it stops is the bar's choice."""

    def residual(self, u, u_dot, u_ddot, t):
        return u_dot + u + u**3 - np.cos(5.0 * t)

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.diag(weights.alpha0.real + weights.alpha1.real * (1.0 + 3.0 * u**2))


def _march(t0, tol, **kw):
    study = TimeWindow(t0, 1.0, 0.1, IntegratorKind.BDF2)
    return integrate_transient(_Cubic(), study, np.zeros(1), tol=tol, max_newton=30, checkpoint_key='cubic', **kw)


def test_the_record_carries_the_bar_and_a_resume_marches_on_it():
    whole = _march(0.0, 1e-4)
    first = integrate_transient(
        _Cubic(),
        TimeWindow(0.0, 0.5, 0.1, IntegratorKind.BDF2),
        np.zeros(1),
        tol=1e-4,
        max_newton=30,
        checkpoint_key='cubic',
    )
    saved = TransientCheckpoint.from_mapping(json.loads(json.dumps(first.checkpoint.to_mapping())))
    assert saved.newton_tolerance == 1e-4
    # The caller hands the resume ANOTHER bar (its load read at the resume instant): the record's wins.
    resumed = _march(saved.time, 3e-3, restart=saved)
    np.testing.assert_array_equal(np.asarray(resumed.states), np.asarray(whole.states[5:]))
    assert resumed.checkpoint.to_mapping() == whole.checkpoint.to_mapping()
    # PLANTED CONTROL: the same resume with the bar stripped from the record is another trajectory.
    bare = _march(saved.time, 3e-3, restart=TransientCheckpoint.from_mapping(
        {k: v for k, v in saved.to_mapping().items() if k != 'newton_tolerance'}
    ))  # fmt: skip
    assert not np.array_equal(np.asarray(bare.states), np.asarray(whole.states[5:]))


@pytest.mark.parametrize('bad', [0.0, -1.0, float('inf')])
def test_a_record_refuses_a_bar_that_is_no_bar(bad):
    with pytest.raises(ValueError, match='checkpoint'):
        TransientCheckpoint('cubic', 'bdf2', 0.5, 0.1, ((0.0,), (1.0,)), newton_tolerance=bad)
