"""The ONE outer fixed point a staggered couple iterates lives in gauss, and its bar is the GAP.

:func:`~numerics_lab.time_integrators.waveform_relaxation.waveform_relaxation` owns the loop the
staggered composers used to carry privately. Its stopping test is the unrelaxed fixed-point gap
``max|G(x) - x|`` -- a test on the relaxed step could be met by lowering ``relaxation`` alone.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from numerics_lab.time_integrators.waveform_relaxation import waveform_relaxation


def test_a_contraction_converges_to_its_fixed_point_and_reports_the_gap_trace() -> None:
    # G(x) = 0.5 x + 1 has the fixed point 2; every gap halves.
    relaxed = waveform_relaxation(lambda x: (0.5 * x + 1.0, 'payload'), 0.0, tol=1e-12, max_iter=100)
    assert relaxed.converged
    assert relaxed.payload == 'payload'
    assert relaxed.state == pytest.approx(2.0, rel=0.0, abs=1e-11)
    assert relaxed.residuals[0] == 1.0
    assert all(b == pytest.approx(0.5 * a, rel=1e-12) for a, b in itertools.pairwise(relaxed.residuals))
    assert relaxed.iterations == len(relaxed.residuals)


def test_relaxation_moves_the_path_and_never_the_bar() -> None:
    """PLANTED: a gap held at 1.0 never converges, however small the relaxation makes the step."""
    # A power-of-two relaxation keeps every relaxed state exact, so the gap stays exactly 1.0.
    stuck = waveform_relaxation(lambda x: (x + 1.0, None), 0.0, relaxation=2.0**-30, tol=1e-6, max_iter=5)
    assert not stuck.converged
    assert stuck.residuals == [1.0] * 5
    assert stuck.state == 5 * 2.0**-30


def test_a_waveform_whose_seed_has_no_shape_is_measured_against_zero() -> None:
    proposal = np.array([1.0, -3.0, 2.0])
    relaxed = waveform_relaxation(lambda _x: (proposal, None), None, relaxation=0.5, tol=1e-12, max_iter=1)
    assert relaxed.residuals == [3.0]
    np.testing.assert_array_equal(relaxed.state, 0.5 * proposal)


@pytest.mark.parametrize(('relaxation', 'max_iter'), [(0.0, 1), (1.5, 1), (1.0, 0)])
def test_an_ill_posed_iteration_is_refused(relaxation: float, max_iter: int) -> None:
    with pytest.raises(ValueError, match='relaxation|max_iter'):
        waveform_relaxation(lambda x: (x, None), 0.0, relaxation=relaxation, tol=1.0, max_iter=max_iter)


def _sweeps(scale: float, **bar: float) -> int:
    # G(x) = 0.5 x + scale: the gap halves every sweep from `scale`, the fixed point is 2 * scale.
    return waveform_relaxation(lambda x: (0.5 * x + scale, None), 0.0, max_iter=100, **bar).iterations


def test_a_relative_bar_stops_at_the_same_sweep_in_any_unit() -> None:
    """``tol_rel`` is measured against the proposal's own peak, so a unit change cannot move the stop."""
    assert len({_sweeps(scale, tol_rel=1e-6) for scale in (1e-3, 1.0, 1e3)}) == 1
    # PLANTED: the same number spelled absolutely stops at a unit-dependent sweep.
    assert _sweeps(1e-3, tol=1e-6) < _sweeps(1e3, tol=1e-6)


def test_a_bar_of_zero_is_refused() -> None:
    with pytest.raises(ValueError, match='tol'):
        waveform_relaxation(lambda x: (x, None), 0.0, max_iter=3)
