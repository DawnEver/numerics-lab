"""Damping that reaches the floor must still be able to come back off it.

`DampedPicardStrategy` already carries a recovery leg -- after `_ALPHA_CALM_STEPS` undisturbed
steps alpha grows back by `_ALPHA_GROW`, capped at `alpha_cap`, "the step size that last
misbehaved, never retried". That leg was added precisely because the ratchet used to be one-way.

It is still one-way, one level down, and this file is the reproduction. `alpha_cap = alpha_before`
runs on EVERY disturbance -- including one that arrives when alpha is already at `_ALPHA_MIN`,
where the shrink `max(_ALPHA_MIN, alpha * 0.6)` is a NO-OP. The cap collapses onto the floor, and
the recovery gate `alpha < alpha_cap` becomes `0.1 < 0.1`, false for the rest of the solve.

MEASURED on `case_h` (a downstream case sweep, 2026-08-27): 557 of 605 Picard iterations pinned at
`alpha: 0.100` with the energy error already reading 0.00%, every solve running to its
100-iteration cap. The iterate crawls a tenth of a step at a time, so the increment half of the
convergence test cannot be reached inside the cap -- the solve does not converge because it is
not allowed to take a full step, not because the physics needs one.
"""

from __future__ import annotations

import numpy as np
import pytest

from numerics_lab.nonlinear import AssembledState, get_strategy
from numerics_lab.nonlinear._seam import _ALPHA_MIN

pytestmark = pytest.mark.unit

#: The undamped step this fixture holds constant. NOT ``+ 10.0`` ON EVERY COMPONENT, which is what
#: it used to be and which `increment_ratio`'s gauge relaxation now reads as NO MOVEMENT AT ALL: a
#: uniform shift of a gauge-free potential carries no energy and no B, so the relaxation reports
#: the mean-removed part -- exactly zero for a constant vector -- and `_has_converged` fired on the
#: first flat-energy step. The run ended at 10 iterations instead of 40 and the recovery assertions
#: below had one sample to look at, or none, so they failed for a reason that had nothing to do
#: with alpha. Scaling the components gives the step a SHAPE, which is what makes it movement; the
#: MAGNITUDE is what these tests are about and it is still constant.
_STEP = np.array([10.0, 5.0, 2.5])


#: The undamped step both fixtures hold CONSTANT so the run cannot converge out from under the
#: question they ask about alpha. It carries SHAPE -- its entries are not all equal -- and that is
#: the whole of why it is a vector and not `+ 10.0`.
#:
#: IT USED TO BE `self.x + 10.0`, A UNIFORM SHIFT, AND A LATER PRODUCTION CHANGE MADE THAT MEAN THE
#: OPPOSITE OF WHAT IT SAYS (found 2026-09-22). `increment_ratio` gained a gauge relaxation on
#: 2026-09-06: in a gauge-free potential an increment that is the SAME in every entry moves no
#: physical quantity, so it is reported as no movement. That is correct, and it is precisely the
#: shape `+ 10.0` has. The increment these fixtures fed the strategy therefore read 0.0 from that
#: day on, the solve met the second half of `_has_converged` on its first step, and the run ended
#: after one undisturbed iteration -- so "alpha stayed pinned for all 1 undisturbed steps" was a
#: report about a fixture that no longer ran long enough to observe anything.
#:
#: THE DOCSTRINGS ALREADY PROMISED THIS AND THE CODE STOPPED DELIVERING IT: both say the increment
#: is held constant "so the run cannot converge out from under the question". Nothing refused the
#: sentence when it stopped being true. Mean-zero keeps it true against the relaxation by
#: construction -- a step whose shape part IS the whole step can never be read as gauge.
_UNDAMPED_STEP = np.array([10.0, 0.0, -10.0])


class _OscillateThenSettle:
    """Oscillate ABOVE tolerance long enough to drive alpha to the floor, then go genuinely flat.

    `solve_linear` holds the undamped increment CONSTANT so the run cannot converge out from under
    the question: what is observed is purely whether alpha recovers once nothing disturbs it.
    """

    is_linear = False

    def __init__(self, disturbed_steps: int = 8) -> None:
        self.n = 0
        self.disturbed_steps = disturbed_steps
        self.alphas: list[float] = []
        self.x = np.zeros(3)

    def initial_guess(self) -> np.ndarray:
        return np.zeros(3)

    def assemble(self, x, /):
        self.n += 1
        self.x = x
        energy = 100.0 + (5.0 if self.n % 2 else -5.0) if self.n <= self.disturbed_steps else 100.0
        return AssembledState(system_matrix=None, total_energy=energy, payload=None)

    def solve_linear(self, state, /) -> np.ndarray:
        return self.x + _UNDAMPED_STEP

    def on_iteration(self, i, state, err, alpha, increment) -> None:
        self.alphas.append(float(alpha))


def _run(max_iter: int = 40) -> list[float]:
    system = _OscillateThenSettle()
    get_strategy('picard-damped').solve(system, max_iter=max_iter)
    return system.alphas


def test_alpha_reaches_the_floor_when_the_energy_genuinely_oscillates() -> None:
    """The FLOOR of this test: if alpha never got damped, the recovery assertion would be vacuous."""
    alphas = _run()
    assert min(alphas) == pytest.approx(_ALPHA_MIN), (
        f'alpha never reached the floor, so this fixture no longer reproduces the condition: {sorted(set(alphas))}'
    )


def test_alpha_recovers_once_the_energy_settles() -> None:
    """The defect: a disturbance arriving AT the floor kills recovery for the rest of the solve.

    The tail here is ~30 consecutive undisturbed steps. `_ALPHA_CALM_STEPS` is 3, so recovery has
    roughly ten opportunities to fire and must take at least one.
    """
    alphas = _run()
    settled = alphas[9:]
    assert max(settled) > _ALPHA_MIN, (
        f'alpha stayed pinned at {_ALPHA_MIN} for all {len(settled)} undisturbed steps -- the recovery leg is '
        'dead, because a no-op shrink at the floor set alpha_cap to the floor and `alpha < alpha_cap` '
        'can never hold again'
    )


def test_the_cap_still_forbids_retrying_a_step_size_that_misbehaved() -> None:
    """THE OTHER SIDE OF THE RATCHET, so the fix cannot be "delete the cap".

    Recovery must stay bounded by the largest step size that actually misbehaved: alpha may climb
    back toward it but must never exceed it, or a genuine limit cycle would be re-entered at full
    step every time the energy briefly flattened.
    """
    alphas = _run()
    disturbed_max = max(alphas[:9])
    assert max(alphas[9:]) <= disturbed_max, (
        f'alpha recovered to {max(alphas[9:])}, above the {disturbed_max} step size that misbehaved'
    )


class _SettleAfter:
    """Oscillate above tolerance for `disturbed_steps`, then hold the energy exactly flat forever.

    The increment is held CONSTANT so the run cannot converge out from under the question. What is
    observed is the ceiling alpha is allowed to climb back to once nothing disturbs it again.
    """

    is_linear = False

    def __init__(self, disturbed_steps: int) -> None:
        self.n = 0
        self.disturbed_steps = disturbed_steps
        self.alphas: list[float] = []
        self.x = np.zeros(3)

    def initial_guess(self) -> np.ndarray:
        return np.zeros(3)

    def assemble(self, x, /):
        self.n += 1
        self.x = x
        energy = 100.0 + (5.0 if self.n % 2 else -5.0) if self.n <= self.disturbed_steps else 100.0
        return AssembledState(system_matrix=None, total_energy=energy, payload=None)

    def solve_linear(self, state, /) -> np.ndarray:
        return self.x + _UNDAMPED_STEP

    def on_iteration(self, i, state, err, alpha, increment) -> None:
        self.alphas.append(float(alpha))


def test_the_cap_itself_releases_once_the_solve_has_settled() -> None:
    """A step size that misbehaved FAR FROM THE SOLUTION must not be forbidden for the whole solve.

    THE SECOND RATCHET, and the one the first fix did not reach. `alpha_cap` records the step size
    that last misbehaved and `alpha` may climb back to it -- but the cap itself never rises, so an
    early, genuinely-meaningful disturbance sets a ceiling that outlives the condition that caused
    it. `mec-picard-runs-to-its-cap-while-already-converged` (2026-08-26) predicted exactly this:
    *"the `meaningful` guard prevents NEW reductions; it does not release an old cap."*

    MEASURED on case_h after releasing only the floor: 557 iterations pinned at 0.100 became
    447 at 0.108 -- recovery fired, and the sweep still needed 658 iterations because 0.108 was the
    ceiling. The physics near convergence is not the physics that misbehaved at iteration 3.

    A long calm tail must therefore let the CAP grow too, back toward the full Picard step.
    """
    system = _SettleAfter(disturbed_steps=8)
    get_strategy('picard-damped').solve(system, max_iter=120)
    settled = system.alphas[10:]
    assert max(settled) > 0.2, (
        f'alpha never climbed above {max(settled)} across {len(settled)} undisturbed steps -- the cap '
        f'is still a one-way ratchet, so a disturbance far from the solution throttles the whole tail'
    )


def test_the_released_cap_still_backs_off_when_the_energy_misbehaves_again() -> None:
    """THE OTHER SIDE: releasing the cap must not disarm the globaliser.

    If a solve starts oscillating again the damping has to return, or the release has traded
    robustness for speed -- the exact trade the cap existed to prevent.
    """
    system = _SettleAfter(disturbed_steps=200)  # never settles
    get_strategy('picard-damped').solve(system, max_iter=60)
    tail = system.alphas[20:]
    assert max(tail) <= 0.7, (
        f'alpha stayed at {max(tail)} while the energy oscillated the whole time -- the damping no '
        f'longer responds to a genuine limit cycle'
    )
