"""Pluggable nonlinear-iteration strategies (a179884 seam 3).

The strategy drives the iterate; the equation only assembles. These synthetic systems exercise the
control flow independent of any real physics: a linear system converges in one substitution; a
contraction converges; an EXPANSIVE map makes plain Picard diverge (characterized, not excluded --
first principle) while damped-Picard's under-relaxation still tames it.
"""

import numpy as np
import pytest

from numerics_lab.nonlinear import (
    ITERATION_REGISTRY,
    AssembledState,
    CoefficientDampedPicardStrategy,
    DampedPicardStrategy,
    PicardStrategy,
    get_strategy,
)


class _FixedPointSystem:
    """A synthetic NonlinearSystem whose ``solve_linear`` returns an affine map ``g(x)=slope*x+b``.

    Fixed point x* solves x* = slope*x* + b -> x* = b/(1-slope). ``slope`` < 1 is a contraction
    (Picard converges); ``slope`` > 1 is expansive (Picard diverges). ``total_energy = sum(x**2)``
    drives the energy-change convergence test.
    """

    def __init__(self, slope: float, b: np.ndarray, *, is_linear: bool = False) -> None:
        self.slope = slope
        self.b = np.asarray(b, float)
        self.is_linear = is_linear
        self.n = self.b.size
        self.assembled = 0
        self.alphas: list[float] = []

    def initial_guess(self):
        return np.zeros(self.n)

    def assemble(self, x):
        self.assembled += 1
        return AssembledState(system_matrix=None, total_energy=float(np.sum(np.asarray(x) ** 2)), payload=np.asarray(x))

    def solve_linear(self, state):
        return self.slope * state.payload + self.b

    def on_iteration(self, i, state, err, alpha, increment):
        self.alphas.append(alpha)


class _LinearSystem:
    """A genuinely linear system: solve_linear returns a fixed target regardless of x, is_linear."""

    def __init__(self, target) -> None:
        self.target = np.asarray(target, float)
        self.is_linear = True

    def initial_guess(self):
        return np.zeros(self.target.size)

    def assemble(self, x):
        return AssembledState(system_matrix=None, total_energy=float(np.sum(np.asarray(x) ** 2)), payload=np.asarray(x))

    def solve_linear(self, state):
        return self.target

    def on_iteration(self, i, state, err, alpha, increment):
        pass


@pytest.mark.parametrize('strategy', [DampedPicardStrategy(), PicardStrategy()])
def test_linear_system_converges_to_target_in_one_substitution(strategy):
    """A linear system converges after a single solve (is_linear short-circuit); both strategies
    return the exact target, and the returned state is the converged iterate's.
    """
    target = np.array([3.0, -1.5])
    res = strategy.solve(_LinearSystem(target), max_iter=50)
    assert res.converged
    assert np.allclose(res.x, target)
    assert np.allclose(res.state.payload, target)  # final assembly is at the converged x


@pytest.mark.parametrize('strategy', [DampedPicardStrategy(), PicardStrategy()])
def test_contraction_converges_to_fixed_point(strategy):
    """A contraction (slope 0.5) converges to x* = b/(1-slope) under both strategies."""
    b = np.array([2.0, 4.0])
    sys = _FixedPointSystem(0.5, b)
    res = strategy.solve(sys, max_iter=500)
    assert res.converged
    assert np.allclose(res.x, b / (1.0 - 0.5), atol=1e-3)


def test_plain_picard_diverges_on_expansive_map_is_characterized_not_excluded():
    """First principle: plain Picard's failure on an expansive map is OBSERVABLE (converged=False,
    the iterate blows up), not hidden -- the datum a convergence matrix would xfail.
    """
    res = PicardStrategy().solve(_FixedPointSystem(2.0, np.array([1.0])), max_iter=30)
    assert not res.converged
    assert np.abs(res.x[0]) > 1e3  # it blew up, plainly


def test_damped_picard_engages_under_relaxation_on_oscillating_energy():
    """When the energy error GROWS (an oscillating/overshooting map, slope -1.5), damped-Picard
    halves alpha below 1.0 -- the adaptive under-relaxation that makes it the production default;
    plain Picard, by contrast, always takes the full step (alpha == 1.0).
    """
    damped = _FixedPointSystem(-1.5, np.array([1.0]))
    DampedPicardStrategy().solve(damped, max_iter=25)
    assert min(damped.alphas) < 1.0  # under-relaxation activated

    plain = _FixedPointSystem(-1.5, np.array([1.0]))
    PicardStrategy().solve(plain, max_iter=25)
    assert all(a == 1.0 for a in plain.alphas)  # undamped: full step always


def test_max_iter_zero_returns_initial_guess_and_no_state():
    res = DampedPicardStrategy().solve(_LinearSystem(np.array([1.0, 2.0])), max_iter=0)
    assert res.iterations == 0
    assert res.state is None
    assert np.allclose(res.x, np.zeros(2))


def test_registry_and_get_strategy():
    assert set(ITERATION_REGISTRY) >= {'picard', 'picard-damped'}
    assert get_strategy(None).name == 'newton'  # the default in every family since 2026-09-25
    assert get_strategy('picard').name == 'picard'
    with pytest.raises(KeyError):
        get_strategy('nonesuch')


# -- P5: FEA's deleted frozen_damped_solve moved onto the ITERATION_REGISTRY seam, as 'coefficient-damped' ---


class _NuMixSystem:
    """Minimal stand-in for ``GalerkinSystem``'s ``raw_state``/``fold_raw`` contract (duck-typed).

    Raw stiffness/source are linear in a single scalar ``k`` per call -- ``k`` stands in for the per-
    element reluctivity target ``nu_target``, since every real FEA kernel's raw stiffness AND magnet-
    source entries are linear in it (see ``CoefficientDampedPicardStrategy``'s docstring). Each call
    returns ``s_val=[k]``, ``T_mat=[2k]``, ``total_energy=k**2``, driven by a ``k`` sequence chosen to
    force a KNOWN alpha-ratchet at a KNOWN iteration, so the mixed value can be hand-computed and
    compared exactly -- this pins the alpha schedule (halve x0.6, floor 0.05, on an energy-delta sign
    flip) verbatim, not just its shape.
    """

    is_linear = False

    def __init__(self, k_seq) -> None:
        self.k_seq = list(k_seq)
        self.calls = 0
        self.fold_history: list[tuple[float, float]] = []  # (s_val, T_mat) actually folded, in order

    def initial_guess(self):
        return np.zeros(1)

    def raw_state(self, x):
        k = self.k_seq[min(self.calls, len(self.k_seq) - 1)]
        self.calls += 1
        return np.array([0]), np.array([0]), np.array([k]), np.array([2.0 * k]), float(k**2)

    def fold_raw(self, s_row, s_col, s_val, T_mat, total_energy=0.0):
        self.fold_history.append((float(s_val[0]), float(T_mat[0])))
        return AssembledState(system_matrix=s_val, total_energy=total_energy, payload={'T_mat': T_mat})

    def solve_linear(self, state):
        return np.array([1.0])  # value is irrelevant here: raw_state ignores x and follows k_seq

    def on_iteration(self, i, state, err, alpha, increment):
        pass


def test_coefficient_damped_registered_and_dispatchable_by_name():
    # 'coefficient-damped' is a registry entry, not only a hardcoded fast path in fea/_newton.py -- get_strategy
    # resolves it like any other named strategy.
    assert 'coefficient-damped' in ITERATION_REGISTRY
    assert isinstance(ITERATION_REGISTRY['coefficient-damped'], CoefficientDampedPicardStrategy)
    assert get_strategy('coefficient-damped').name == 'coefficient-damped'


def test_coefficient_damped_mixes_the_raw_matrix_exactly_like_the_deleted_per_element_mix():
    """Hand-computed check of both the matrix-mixing math and the alpha-ratchet schedule.

    ``k_seq = [1, 3, 2, 5]`` (energy ``k**2``) gives deltas +8 (sign +1, no ratchet yet -- there is no
    PRIOR sign to flip against), then -5 (sign -1, FLIPS vs +1 -> alpha ratchets to 0.6 for the NEXT
    mix). So the 4th call's mixed reluctivity is ``0.4*k2 + 0.6*k3 = 0.4*2 + 0.6*5 = 3.8``, computed
    here independently of the strategy.
    """
    system = _NuMixSystem([1.0, 3.0, 2.0, 5.0])
    CoefficientDampedPicardStrategy().solve(system, max_iter=4)
    assert len(system.fold_history) == 4
    # Calls 1-3: alpha stays 1.0 (the sign-flip trigger needs a PRIOR sign, so it cannot fire before
    # the 3rd energy sample) -- the mix reduces to nu_target every time.
    assert system.fold_history[0] == (1.0, 2.0)
    assert system.fold_history[1] == (3.0, 6.0)
    assert system.fold_history[2] == (2.0, 4.0)
    # Call 4: alpha ratcheted to 0.6 (energy sign flip between calls 2 and 3).
    expected_s = 0.4 * 2.0 + 0.6 * 5.0
    assert system.fold_history[3] == pytest.approx((expected_s, 2.0 * expected_s))


def test_coefficient_damped_strategy_rejects_a_system_without_raw_state_or_fold_raw():
    with pytest.raises(TypeError, match='raw_state/fold_raw'):
        CoefficientDampedPicardStrategy().solve(_LinearSystem(np.array([1.0])), max_iter=5)
