"""A residual is known only to the certified bound of every quantity it evaluates -- its LAW term included.

Ruling 2026-10-19 (coordinator, first principles): a constitutive law the residual reads only through a certified
inverse (the play model's ``H(B)``) is part of RESIDUAL ASSEMBLY uncertainty, not a solver's resolution. It is a
SEPARATE named term of :class:`ResidualAssemblyBound` (``law``), from the same bar the inverse is accepted against,
and an admission that refused a row below it asked for what no evaluation can certify (MEASURED 2026-10-19, sdm's
k = 2 hysteretic ring: two bubble rows at ~1e-17 against a 1e-27 bar, re-assembly noise ~1e-17).

PLANTED CONTROL: a zero law term leaves the state admission exactly where the plain bar left it.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.formulation import ResidualAssemblyBound
from numerics_lab.linear.arithmetic import affine_residual_bound, transport_residual_bound
from numerics_lab.linear.storage_border import StorageRows
from numerics_lab.nonlinear.finite_storage import _within_allowance, _within_law

pytestmark = pytest.mark.unit


def test_the_law_term_is_its_own_name_and_independent_of_the_arithmetic_mask():
    bound = ResidualAssemblyBound(np.array([1.0, 2.0]), np.array([True, False]), np.array([0.5, 0.25]))
    assert np.array_equal(bound.allowance(), [1.5, 0.25])  # an unknown arithmetic does not unknow the law
    assert np.array_equal(ResidualAssemblyBound(np.ones(2), np.ones(2, dtype=bool)).law, np.zeros(2))
    assert np.array_equal(bound.take([1]).law, [0.25])
    with pytest.raises(ValueError, match='law'):
        ResidualAssemblyBound(np.ones(2), np.ones(2, dtype=bool), np.array([-1.0, 0.0]))


def test_the_arithmetic_carries_the_law_term():
    bound = ResidualAssemblyBound(np.zeros(2), np.ones(2, dtype=bool), np.array([1.0, 2.0]))
    matrix = sp.csr_matrix([[1.0, -3.0], [0.0, 2.0]])
    assert np.array_equal(affine_residual_bound(matrix, np.ones(2), np.ones(2), bound).law, [1.0, 2.0])
    assert np.array_equal(transport_residual_bound(matrix, np.ones(2), bound).law, [7.0, 4.0])


def _storage():
    return StorageRows(np.array([], dtype=int), sp.identity(2, format='csr'))


def _evaluation(law):
    # The engine reads an evaluation by its ``residual`` and ``bound`` alone (the owner's type is the owner's).
    return lambda _state, _rate: SimpleNamespace(
        residual=np.zeros(2), bound=ResidualAssemblyBound(np.zeros(2), np.zeros(2, dtype=bool), law)
    )


def test_a_row_inside_its_law_term_is_admitted():
    projected, bar = np.array([1e-17, 0.0]), np.array([1e-27, 1e-27])
    assert _within_law(_evaluation(np.array([2e-17, 0.0])), np.zeros(2), np.zeros(2), _storage(), projected, bar)
    assert not _within_law(_evaluation(np.array([5e-18, 0.0])), np.zeros(2), np.zeros(2), _storage(), projected, bar)


def test_planted_control_a_zero_law_term_admits_nothing_the_plain_bar_refused():
    projected, bar = np.array([1e-17, 0.0]), np.array([1e-27, 1e-27])
    assert not _within_law(_evaluation(np.zeros(2)), np.zeros(2), np.zeros(2), _storage(), projected, bar)
    assert not _within_law(None, np.zeros(2), np.zeros(2), _storage(), projected, bar)


def test_the_frozen_rate_pair_is_judged_to_the_allowance_it_was_solved_to():
    """The pair stage solves its frozen rate to bar + construction allowance; judging the result to the bare bar made
    the stage a fixed point (MEASURED 2026-10-19, sdm k = 2 hysteretic ring: 100 passes, zero state columns, rows
    1158/1554 at ~1e-9 inside a 6e-9 allowance and the bubble rows at ~2e-18 inside 4e-13, refused at ~1e-27).
    """
    residual, bar = np.array([1e-9, 2e-18]), np.array([3e-10, 2e-27])
    assert _within_allowance(residual, bar, np.array([6e-9, 4e-13]))
    assert not _within_allowance(residual, bar, np.array([6e-9, 1e-18]))


def test_planted_control_a_zero_allowance_is_the_plain_bar():
    bar = np.array([3e-10, 2e-27])
    assert not _within_allowance(np.array([1e-9, 0.0]), bar, np.zeros(2))
    assert _within_allowance(np.array([1e-10, 0.0]), bar, np.zeros(2))
