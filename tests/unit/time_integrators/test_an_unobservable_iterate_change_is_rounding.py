"""An iterate change the residual cannot see is rounding in the iterate, however large its |J| |du| bound.

THE MEASURED CASE (motronics, 2026-11-08): the three-phase transformer chopper's 330 A trip. Every event
probe of the step 1.0 -> 1.5036 ms was refused 50 times and the march stopped with "event probe at
t=0.00150363 did not converge". Every residual row sat at ~1e-5 of its own bar from the first iteration;
what refused was the local check on the nodal A field, which moved 1e-8 relative per iteration -- the
rounding of a 1e17-scale row propagated through an ill-conditioned field solve. The fallback that exists for
exactly that question measured the change by the BOUND ``|J| |du|`` (5e3..1.5e4 x the rows' rounding bar)
instead of the change's actual effect ``|J du|`` (0.02..0.11 x): a change another Newton iteration can
neither confirm nor correct was judged as if it were progress.

The question is whether the change MOVED a residual row above its rounding: ``|J du|``. A change that does
(the planted control below) is still refused.
"""

import numpy as np

from numerics_lab.time_integrators._forms import change_below_residual_resolution

# An ill-conditioned pair: (1, -1) is a near-null direction, (1, 1) is not.
_TANGENT = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-10]])
_STATE = np.array([1.0, 2.0])


def test_a_change_along_a_direction_the_residual_cannot_see_is_rounding() -> None:
    after = _STATE + 1e-7 * np.array([1.0, -1.0])
    # The bound reads 2e-7 per row, ~1e7 x the rows' rounding; the residual moves ~1e-17.
    assert change_below_residual_resolution(_TANGENT, _STATE, after, None)


def test_planted_control_a_change_the_residual_sees_is_refused() -> None:
    after = _STATE + 1e-7 * np.array([1.0, 1.0])
    assert not change_below_residual_resolution(_TANGENT, _STATE, after, None)


def test_planted_control_a_null_direction_change_that_is_not_small_is_refused() -> None:
    """The tangent's own near-null direction still moves the residual once the change is large enough."""
    after = _STATE + 1e-2 * np.array([1.0, -1.0])
    assert not change_below_residual_resolution(_TANGENT, _STATE, after, None)
