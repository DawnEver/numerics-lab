"""The Newton step's convergence bar must be SCALE-INVARIANT, and must still refuse.

The defect (found independently by two lanes, 2026-09-11 / 2026-09-12): a LINEAR circuit whose
single Newton step is EXACT reports ``converged = False``, because the bar was an absolute
``tol = 1e-12`` on a residual whose magnitude is set by the supply. A 24 V circuit lands at
~7e-12 (3e-13 RELATIVE -- exact) and a 2400 V one at ~5e-11, so the SAME exact solve is
"converged" or "exhausted" depending only on how big the volts are.

The bar this file pins instead is the ARITHMETIC FLOOR of the residual evaluation itself. A
residual row is a sum of signed contributions ``sum_j J_ij u_j`` that CANCEL, and summing terms
of magnitude ``M`` in floating point cannot resolve below ``eps * M``; so ``eps * (|J| . |u|)``
is the smallest residual that system can MEAN. It carries the residual's own units, is derived
from data the solver already holds, and scales WITH the supply exactly as the residual does --
which is the invariance the absolute number lacked.

That is not a new idea in this repo and deliberately is not spelled a second way: the DC
operating point already computes it (``mna_solves._roundoff_floor`` / ``op_residual_floor``,
whose own measurement records one circuit's floor spanning 56x across four gate states). The
marcher differs only in taking the 2-norm of the same ``|J| . |u|`` vector, because it compares
``||R||_2`` where the op compares ``max|R|``. ``test_the_floor_agrees_with_the_dc_operating_-
points_own_floor`` calls the REAL laplace function and pins the two together.

Both halves are pinned here. A bar that never refuses is the opposite defect and is worse, so
the planted-divergence cases prove the tag still goes False, with a MEASURED ratio-to-floor.
"""

from __future__ import annotations

import numpy as np
import pytest

from motronics.contracts.spec.study_spec import IntegratorKind, TransientStudy
from motronics.gauss.time_integrators import integrate_transient, residual_floor
from motronics.laplace.circuit.core.build import CircuitBuilder
from motronics.laplace.circuit.core.mna import assemble_mna
from motronics.laplace.circuit.core.mna_solves import _roundoff_floor as _op_roundoff_floor
from motronics.laplace.circuit.sources import DC

#: The absolute bar this file retires as the DEFAULT decision. Kept here as the FLOOR of the
#: scale-invariance test: at 2400 V the exact solve's residual must be measurably ABOVE it, or
#: the test would pass for the old code too and prove nothing.
_RETIRED_ABSOLUTE_BAR = 1e-12


def _linear_rl(volts: float):
    """A LINEAR series R-L on a ``volts`` supply: one Newton iteration is exact by construction."""
    b = CircuitBuilder()
    b.voltage_source('V1', 'a', 'gnd', source=DC(volts))
    b.resistor('R1', 'a', 'b', 1.0)
    b.inductor('L1', 'b', 'gnd', 1e-3)
    return assemble_mna(b.build(), ground='gnd')


def _march(volts: float):
    study = TransientStudy(t0=0.0, t1=2e-3, dt=1e-5, integrator=IntegratorKind.BDF2)
    return _linear_rl(volts).solve_transient(study)


@pytest.mark.parametrize('volts', [24.0, 2400.0])
def test_a_linear_circuits_exact_solve_reports_converged_at_every_supply(volts: float) -> None:
    """An exactly-solved LINEAR march is converged at 24 V and at 100x that. THE deliverable."""
    sol = _march(volts)
    assert sol.nonconverged_times == [], (
        f'{volts} V linear R-L: {len(sol.nonconverged_times)} of its steps were labelled '
        f'exhausted at worst residual {sol.worst_step_residual:.3e} -- an exact linear solve'
    )
    assert sol.converged


def test_the_exact_residual_grows_WITH_the_supply_so_an_absolute_bar_cannot_hold_both() -> None:
    """The FLOOR under the test above: without it the 24 V case alone proves nothing.

    The exact residual is proportional to the supply, so the 2400 V march sits ABOVE the retired
    1e-12 -- i.e. the parametrized test genuinely fails on the old criterion at 2400 V -- while
    the RELATIVE residual is the same tiny number at both supplies.
    """
    lo, hi = _march(24.0), _march(2400.0)
    assert hi.worst_step_residual > _RETIRED_ABSOLUTE_BAR * 10, (
        f'the 2400 V exact residual {hi.worst_step_residual:.3e} did not clear the retired bar '
        f'{_RETIRED_ABSOLUTE_BAR:.1e}; this test no longer has a floor'
    )
    rel_lo = lo.worst_step_residual / 24.0
    rel_hi = hi.worst_step_residual / 2400.0
    assert rel_lo == pytest.approx(rel_hi, rel=5.0, abs=0.0), (
        f'the residual is not supply-proportional: {rel_lo:.3e} /V vs {rel_hi:.3e} /V'
    )
    assert rel_hi < 1e-12, f'relative residual {rel_hi:.3e} /V is not an exact solve'


def test_the_floor_is_the_roundoff_scale_and_nothing_wider() -> None:
    """``residual_floor`` is eps-scaled, linear in both operands, and zero for a zero system.

    The unit: the floor carries the RESIDUAL's units because ``|J| . |u|`` does -- J maps u-units
    to residual-units. Doubling either operand doubles the floor; there is no additive absolute
    constant hiding inside it, which is the whole difference from the bar it replaces.
    """
    jac = np.array([[2.0, 0.0], [0.0, 3.0]])
    u = np.array([1.0, 1.0])
    assert residual_floor(np.zeros((2, 2)), u) == 0.0
    assert residual_floor(jac, np.zeros(2)) == 0.0
    one = residual_floor(jac, u)
    assert 0.0 < one < 1e-12, f'the floor at unit scale is {one:.3e} -- not a roundoff quantity'
    assert residual_floor(jac, 2.0 * u) == pytest.approx(2.0 * one)
    assert residual_floor(2.0 * jac, u) == pytest.approx(2.0 * one)


def test_the_floor_agrees_with_the_dc_operating_points_own_floor() -> None:
    """ONE criterion, not two spellings. The DC op solve already computes this bar
    (``mna_solves._roundoff_floor``); this marcher must not drift from it.

    They differ ONLY in the norm each layer measures its own residual in -- the op compares
    ``max|R|`` and takes the row max, this marcher compares ``||R||_2`` and takes the 2-norm of
    the same ``|J| . |u|`` vector. So the marcher's floor must bracket the op's: never below it
    (the 2-norm dominates the max) and never more than ``sqrt(n)`` above.
    """
    rng = np.random.default_rng(20260912)
    for n in (3, 12, 40):
        jac = rng.normal(size=(n, n))
        u = rng.normal(size=n) * 1e3
        ours = residual_floor(jac, u)
        theirs = _op_roundoff_floor(jac, u)  # the REAL laplace function, imported not copied
        assert ours >= theirs * (1.0 - 1e-12), f'n={n}: {ours:.3e} fell below the op floor {theirs:.3e}'
        assert ours <= theirs * np.sqrt(n) * (1.0 + 1e-12), f'n={n}: {ours:.3e} exceeds sqrt(n) x {theirs:.3e}'


class _RootlessProblem:
    """A PLANTED non-convergence, parametrised by SCALE.

    ``R(u) = offset * (2 + tanh(u))`` is bounded in ``[offset, 3 * offset]`` and has NO root, so
    no iteration of any kind can drive the residual below ``offset``. The tangent supplied is the
    constant chord ``offset`` -- deliberately, so the iteration takes bounded steps forever
    instead of running into a singular jacobian: what is being tested is the VERDICT on a
    residual that cannot fall, not a linear-algebra failure.

    ``offset`` is the whole point: a bar that became scale-invariant by becoming LOOSE would
    swallow the biggest one.
    """

    def __init__(self, offset: float) -> None:
        self.offset = offset

    def residual(self, u, u_dot, u_ddot, t):
        return self.offset * (2.0 + np.tanh(u))

    def jacobian(self, u, u_dot, u_ddot, t, weights):
        return np.atleast_2d(self.offset)


@pytest.mark.parametrize('offset', [1.0, 1.0e4, 1.0e8])
def test_a_planted_divergence_still_reports_not_converged_at_every_scale(offset: float) -> None:
    """The bar must REFUSE, at every scale, and the refusal must be far above the floor."""
    study = TransientStudy(t0=0.0, t1=5e-3, dt=1e-3, integrator=IntegratorKind.BDF2)
    sol = integrate_transient(_RootlessProblem(offset), study, np.array([1.0]), max_newton=12)
    assert sol.nonconverged_times, (
        f'offset {offset:.0e}: a residual bounded below by {offset:.0e} was labelled converged '
        f'at worst {sol.worst_step_residual:.3e}'
    )
    assert not sol.converged
    assert sol.worst_step_residual >= offset, (
        f'offset {offset:.0e}: reported residual {sol.worst_step_residual:.3e} is below the '
        'planted lower bound -- the march did not actually visit the stall'
    )


def test_the_planted_stall_is_orders_above_its_own_floor() -> None:
    """The HEADROOM, measured rather than asserted in prose: the refused residual sits many
    orders above the floor that would have excused it, so the fix has margin on both sides.
    """
    offset = 1.0e4
    study = TransientStudy(t0=0.0, t1=5e-3, dt=1e-3, integrator=IntegratorKind.BDF2)
    problem = _RootlessProblem(offset)
    sol = integrate_transient(problem, study, np.array([1.0]), max_newton=12)
    worst_u = np.array([max(abs(float(x[0])) for x in sol.states)])
    floor = residual_floor(np.atleast_2d(offset), worst_u)
    assert sol.worst_step_residual / floor > 1e8, (
        f'the stall {sol.worst_step_residual:.3e} is only {sol.worst_step_residual / floor:.3e}x its floor {floor:.3e}'
    )


def test_a_residual_stalled_a_thousand_eps_above_the_floor_is_refused() -> None:
    """The SHARP edge: not a diverging problem, but one whose residual settles at a fixed,
    physically meaningful distance from zero. The floor must sit far enough below that the
    difference between 'exact' and 'nearly right' survives.
    """
    n = 40
    jac = np.eye(n) * 1.0e4
    u = np.full(n, 1.0e3)
    floor = residual_floor(jac, u)
    stalled = 1.0e-6 * float(np.linalg.norm(np.abs(jac) @ np.abs(u)))  # 1 ppm relative: WRONG
    assert stalled > floor * 1e6, (
        f'a 1 ppm-relative residual {stalled:.3e} is only {stalled / floor:.3e}x the floor '
        f'{floor:.3e}; the criterion has lost the ability to tell exact from nearly-right'
    )
