"""Per-element convergence must not be hidden by the bulk field."""

import numpy as np
import pytest

from motronics.gauss.nonlinear.local import local_change, newton_linear_tolerance


def test_small_element_and_zero_crossing_are_not_hidden():
    old = (np.array([[100.0, 0.0], [0.0, 0.0]]),)
    new = (np.array([[100.0, 0.0], [0.001, 0.0]]),)
    assert local_change(old, new) == 1.0
    assert local_change(old, old) == 0.0


def test_units_do_not_mix_and_invalid_states_do_not_converge():
    old = (np.array([1e9]), np.array([1e-9]))
    new = (np.array([1e9]), np.array([2e-9]))
    assert local_change(old, new) == 0.5
    assert np.isinf(local_change((np.array([0.0]),), (np.array([np.nan]),)))
    with pytest.raises(ValueError, match='fields changed'):
        local_change(old, ())


def test_newton_tolerance_tightens_with_original_rhs_residual():
    assert newton_linear_tolerance(1.0) == 1e-2
    assert newton_linear_tolerance(1e-4) == pytest.approx(1e-7)
    assert newton_linear_tolerance(0.0) == 1e-10


def test_local_units_do_not_underflow_or_overflow_the_relative_change():
    for scale in (1e-200, 1e200):
        old = (np.array([[scale, scale]]),)
        new = (np.array([[2 * scale, 2 * scale]]),)
        assert local_change(old, new) == pytest.approx(0.5)
