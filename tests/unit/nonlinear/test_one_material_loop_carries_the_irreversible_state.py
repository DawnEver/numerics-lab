"""The material-update seam: coefficients per POINT, a committed/trial state, and the material term.

Pinned here without any physics: a toy reversible law (coefficient = f(|B|), no state) and a toy
one-way clamp (state only ever decreases) stand for steel and magnet recoil. What is asserted is the
bookkeeping every solver will rely on -- not a B-H curve.
"""

from __future__ import annotations

import numpy as np
import pytest

from motronics.gauss.nonlinear import AssembledState, DampedPicardStrategy
from motronics.gauss.nonlinear._seam import _has_converged, _StagnationWatch
from motronics.gauss.nonlinear.material_state import (
    CompositeUpdate,
    MaterialCoeffs,
    MaterialFields,
    MaterialLoop,
    MaterialUpdate,
    relax_linear,
)

pytestmark = pytest.mark.unit


class _Reversible:
    irreversible = False

    def init_state(self, fields, /):
        return None

    def evaluate(self, state, fields, temperature=None, /):
        return None, MaterialCoeffs(nu=1.0 + fields.b_norm**2)

    def relax(self, previous, target, alpha, /):
        return relax_linear(previous, target, alpha)

    def change(self, committed, trial, /):
        return 0.0


class _OneWayClamp:
    """State = per-point level, clamped DOWN to |B| and never back up."""

    irreversible = True

    def init_state(self, fields, /):
        return np.ones(fields.n)

    def evaluate(self, state, fields, temperature=None, /):
        trial = np.minimum(state, fields.b_norm)
        return trial, MaterialCoeffs(br=trial)

    def relax(self, previous, target, alpha, /):
        return relax_linear(previous, target, alpha)

    def change(self, committed, trial, /):
        return float(np.max(np.abs(committed - trial)) / max(float(np.max(np.abs(committed))), 1e-300))


def _fields(b):
    b = np.asarray(b, dtype=float)
    return MaterialFields(b=b, group=np.zeros(b.shape[0], dtype=int))


def test_both_toys_satisfy_the_protocol():
    assert isinstance(_Reversible(), MaterialUpdate)
    assert isinstance(_OneWayClamp(), MaterialUpdate)
    assert isinstance(CompositeUpdate([_Reversible(), _OneWayClamp()]), MaterialUpdate)


def test_fields_are_per_point_vectors_or_norms():
    f = MaterialFields(b=np.array([[3.0, 4.0], [0.0, 1.0]]), group=np.array([0, 1]))
    np.testing.assert_allclose(f.b_norm, [5.0, 1.0])
    with pytest.raises(ValueError, match='VECTOR'):
        _fields([1.0, 2.0]).b_vec


def test_the_composite_merges_disjoint_coefficients_and_refuses_a_shared_one():
    comp = CompositeUpdate([_Reversible(), _OneWayClamp()])
    f = _fields([0.5, 2.0])
    _state, coeffs = comp.evaluate(comp.init_state(f), f)
    np.testing.assert_allclose(coeffs.nu, [1.25, 5.0])
    np.testing.assert_allclose(coeffs.br, [0.5, 1.0])
    assert comp.irreversible
    with pytest.raises(ValueError, match="both set 'nu'"):
        CompositeUpdate([_Reversible(), _Reversible()]).evaluate((None, None), f)


def test_a_trial_state_is_not_committed_until_the_step_is_accepted():
    loop = MaterialLoop(_OneWayClamp(), _fields([1.0]))
    loop.evaluate(_fields([0.25]))
    assert loop.last_change == pytest.approx(0.75)
    np.testing.assert_allclose(loop.committed, [1.0])  # a rejected trial ratchets nothing
    loop.commit()
    np.testing.assert_allclose(loop.committed, [0.25])
    coeffs = loop.evaluate(_fields([0.9]))  # one-way: a recovering field cannot raise it
    np.testing.assert_allclose(coeffs.br, [0.25])
    assert loop.last_change == 0.0


def test_relax_blends_every_set_coefficient_by_one_global_alpha():
    out = relax_linear(MaterialCoeffs(nu=np.array([1.0]), br=None), MaterialCoeffs(nu=np.array([3.0])), 0.25)
    np.testing.assert_allclose(out.nu, [1.5])
    assert out.br is None


def test_the_one_rule_refuses_while_the_material_still_moves():
    x = np.ones(4)
    assert _has_converged(0.0, x, x, 0.01, material_change=0.0)
    assert not _has_converged(0.0, x, x, 0.01, material_change=1e-3)


def test_the_stagnation_escape_is_closed_while_the_material_moves():
    watch = _StagnationWatch(0.01, window=3)
    assert not any(watch.observe(0.0, 1e-3, 1e-2) for _ in range(10))
    assert [watch.observe(0.0, 1e-3, 0.0) for _ in range(3)][-1]


class _ClampSystem:
    """A scalar fixed point whose source is a one-way clamp: x = br, br = min(br, target).

    The material state walks 1.0 -> 0.5 over several steps (half the gap per step), so an iterate
    test alone would see ``x`` settle one step early; the material term is what holds the solve.
    """

    is_linear = False

    def __init__(self) -> None:
        self.br = 1.0
        self.trial = 1.0

    def initial_guess(self):
        return np.array([1.0])

    def assemble(self, x, /):
        self.br = self.trial  # the previous step was accepted
        self.trial = max(0.5, self.br - 0.5 * (self.br - 0.5))
        change = abs(self.br - self.trial) / self.br
        return AssembledState(system_matrix=None, total_energy=1.0, material_change=change)

    def solve_linear(self, state, /):
        return np.array([1.0])  # the iterate NEVER moves: only the material does

    def on_iteration(self, *_):
        pass


def test_a_still_iterate_over_a_moving_clamp_is_not_converged():
    system = _ClampSystem()
    result = DampedPicardStrategy().solve(system, max_iter=60)
    assert result.converged
    assert result.iterations > 10  # without the material term it stops at iteration 2
    assert system.br - 0.5 < 2e-4
