"""Damping that reaches the floor must still be able to come back off it.

`DampedPicardStrategy` already carries a recovery leg -- after `_ALPHA_CALM_STEPS` undisturbed
steps alpha grows back by `_ALPHA_GROW`, capped at `alpha_cap`, "the step size that last
misbehaved, never retried". That leg was added precisely because the ratchet used to be one-way.

It is still one-way, one level down, and this file is the reproduction. `alpha_cap = alpha_before`
runs on EVERY disturbance -- including one that arrives when alpha is already at `_ALPHA_MIN`,
where the shrink `max(_ALPHA_MIN, alpha * 0.6)` is a NO-OP. The cap collapses onto the floor, and
the recovery gate `alpha < alpha_cap` becomes `0.1 < 0.1`, false for the rest of the solve.

MEASURED on `dcm_fdm_rect` (`motronics sweep`, 2026-08-27): 557 of 605 Picard iterations pinned at
`alpha: 0.100` with the energy error already reading 0.00%, every solve running to its
100-iteration cap. The iterate crawls a tenth of a step at a time, so the increment half of the
convergence test cannot be reached inside the cap -- the solve does not converge because it is
not allowed to take a full step, not because the physics needs one.
"""

from __future__ import annotations

import numpy as np
import pytest

from motronics.gauss.nonlinear import _ALPHA_MIN, AssembledState, get_strategy

pytestmark = pytest.mark.unit


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
        return self.x + 10.0

    def on_iteration(self, i, state, err, alpha) -> None:
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
