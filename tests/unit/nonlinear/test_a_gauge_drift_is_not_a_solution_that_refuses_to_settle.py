"""A uniform shift in a gauge-free potential is not movement, and must not read as one.

`_has_converged` needs BOTH halves: the energy error under `rtol_energy_pct` AND
`increment_ratio(x, x_next)` under `rtol_energy_pct / 100`. The second half is the fixed-point
test -- has the iterate stopped moving. In 2D magnetostatics the unknown is the vector potential
`A`, which is defined only up to an additive constant unless a Dirichlet row pins it; that constant
carries no energy and no `B`, so an iterate drifting along it has reached the fixed point of every
physical quantity while `||x_next - x|| / ||x||` goes on reporting a solution that will not settle.

MEASURED 2026-09-06 on `case_c`, 400 consecutive increments instrumented inside the
newton and picard strategies:

    raw increment      3.5e-02 .. 2.7e-01     against a criterion needing < 1e-4
    mean-removed       1.0e-06 .. 1.8e-06     four orders below the raw, and UNDER the criterion
    constant fraction  1.0000                 to four decimals, on all 400
    energy             185.69 to six digits, flat throughout

All four one-hot solves ran to the 100-iteration cap. At 364.177 A the matrix they produced was
refused by `require_physical_inductance` with eigenvalue -5.76e-05 H; at 36.4 A it came back with a
19 % reciprocity residual. `L_ij = L_ji` is an energy theorem, so that is a statement about the
MEASUREMENT. The solve was physically converged; the CRITERION was measuring a gauge mode.

THE FLOOR ON THE FIX. Subtracting the mean unconditionally would be the wrong shape: a real change
can carry a nonzero mean (an anchored field growing everywhere), and removing it would under-report
genuine movement. So the relaxation is narrow -- it fires only when the increment is almost ENTIRELY
the constant mode, and the moment a shape change enters, the fraction falls and the raw ratio is
what answers. Both sides are tested here; a ratchet with only one side is as wrong as its opposite.
"""

from __future__ import annotations

import numpy as np

from numerics_lab.nonlinear import increment_ratio

_RNG = np.random.default_rng(20260906)
_X = _RNG.normal(size=512) * 3.0 + 10.0


def test_a_PURE_constant_shift_reads_as_no_movement():
    """The measured case: the whole increment is the gauge mode."""
    shift = 0.35 * float(np.linalg.norm(_X)) / np.sqrt(_X.size)
    assert increment_ratio(_X, _X + shift) < 1e-12, 'a gauge shift was counted as solution movement'


def test_the_RAW_ratio_would_have_called_that_same_shift_large():
    """The floor: without the fix there is a number here, and it is nowhere near tolerance.

    Recomputed from the definition rather than asserted as a constant, so this cannot pass by
    agreeing with a stale expectation.
    """
    shift = 0.35 * float(np.linalg.norm(_X)) / np.sqrt(_X.size)
    raw = float(np.linalg.norm(np.full_like(_X, shift))) / float(np.linalg.norm(_X))
    assert raw > 1e-4, f'the probe increment {raw:.3e} is too small to demonstrate anything'


def test_a_SHAPE_change_is_still_movement_even_riding_on_a_large_constant():
    """The other side of the ratchet. A real change must not be laundered by adding an offset."""
    shape = _RNG.normal(size=_X.size)
    shape *= 1e-2 * float(np.linalg.norm(_X)) / float(np.linalg.norm(shape))
    ratio = increment_ratio(_X, _X + shape + 5.0)
    assert ratio > 1e-3, f'a 1 % shape change hid behind a constant offset and read {ratio:.3e}'


def test_an_ANCHORED_increment_is_untouched_by_the_relaxation():
    """Where the gauge is pinned the mean increment is ~0, so this branch cannot fire at all.

    The same vector with and without the fix must give the same number -- computed here from the
    definition, not from a recorded value.

    THE DEFINITION IS EVALUATED ON THE INCREMENT THE FUNCTION CAN SEE, and that correction is the
    only reason this line is not `norm(delta)`. `increment_ratio` receives `x` and `x_next`, so the
    increment it measures is `(x + delta) - x`, which is not bit-identical to `delta`: measured
    here, 0.0010000000000000033 against 0.001, a relative 3.3e-15. Asserting against `norm(delta)`
    made this test unsatisfiable by ANY implementation -- it failed against the ORIGINAL
    `increment_ratio` too, before the relaxation existed, so it was never evidence about the
    relaxation at all. Exact equality is still the right assertion, because the claim is that the
    relaxation branch does not FIRE here, and a fired branch changes the value by orders rather
    than by an ulp.
    """
    delta = _RNG.normal(size=_X.size)
    delta -= float(np.mean(delta))  # a Dirichlet-anchored increment: no constant component
    delta *= 1e-3 * float(np.linalg.norm(_X)) / float(np.linalg.norm(delta))
    expected = float(np.linalg.norm((_X + delta) - _X)) / float(np.linalg.norm(_X))
    assert increment_ratio(_X, _X + delta) == expected


def test_a_ZERO_increment_is_still_zero():
    """The degenerate edge, where the constant fraction is undefined."""
    assert increment_ratio(_X, _X.copy()) == 0.0
