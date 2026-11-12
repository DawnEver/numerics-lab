"""An ALGEBRAIC row is never held below the rounding of the unknowns it reads, as the problem declares them.

``RowAcceptance`` bars a row at ``ADMISSION_BAR`` of ``|F_u| |u| + ...``, reading ``|u|`` at face value. An
unknown that a cancelling constraint places -- an ideal transformer's ``i_0 + i_1 - i_2 = 0`` with
``i_0, i_2 ~ 3.7 A`` -- is resolved only to the rounding of that sum (~1e-15 A), however small its own value
(``i_1 ~ 5e-11 A``). A row reading it was barred at 1e-20 A against a 2.2e-17 A residual: Newton converged by
its third iteration, then backtracked to its ceiling every step (measured downstream 2026-11-12 on an LLC
march: 24 Newton iterations and 82 trials per step, 553 of 1000 steps labelled non-converged). The rate's
CONSTRUCTION floor already reads ``state_resolution``; an algebraic row now gets the same floor through
``|F_u|``: ``FLOOR_SAFETY eps |F_u| resolved``.
"""

from types import SimpleNamespace

import numpy as np

from numerics_lab.formulation import DerivativeWeights
from numerics_lab.time_integrators._step import RowAcceptance

_BIG = 3.7  # A: the two currents the ampere-turn row cancels
_SMALL = 5.3e-11  # A: the idle winding's current, placed only through that cancelling row


class _Transformer:
    """``[i0 + i1 - i2, i0 - BIG, i1 - SMALL]``: an algebraic system whose small unknown sits in a cancelling sum."""

    def __init__(self, declares: bool) -> None:
        if declares:
            self.state_resolution = self._resolution

    @staticmethod
    def _resolution(u: np.ndarray) -> np.ndarray:
        # the cancelling sum's own magnitude: i1 is known only to the rounding of |i0| + |i1| + |i2|
        total = float(np.sum(np.abs(u)))
        return np.array([abs(u[0]), total, abs(u[2])])

    def residual(self, state, _rate, _acceleration, _time):
        i0, i1, i2 = state
        return np.array([i0 + i1 - i2, i0 - _BIG, i1 - _SMALL])

    def jacobian(self, _state, _rate, _acceleration, _time, weights):
        return weights.alpha0 * np.array([[1.0, 1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _static_form():
    return SimpleNamespace(
        t_eval=0.0,
        weights=DerivativeWeights(1.0, 0.0, 0.0),
        u_arg=lambda state: state,
        u_dot=np.zeros_like,
        u_ddot=np.zeros_like,
    )


def _state(small_error: float) -> np.ndarray:
    i1 = _SMALL + small_error
    return np.array([_BIG, i1, _BIG + i1])


def _accepts(declares: bool, small_error: float) -> bool:
    problem = _Transformer(declares)
    u = _state(small_error)
    return RowAcceptance(problem, _static_form()).accepts(u, problem.residual(u, None, None, 0.0))


def test_a_row_at_the_rounding_of_its_declared_resolution_is_accepted():
    assert _accepts(declares=True, small_error=2.2e-17)  # A: the rounding the cancelling sum leaves


def test_without_a_declared_resolution_the_face_value_bar_still_refuses_it():
    # PLANTED: the declaration is what licenses the floor; a problem that declares nothing keeps |u|.
    assert not _accepts(declares=False, small_error=2.2e-17)


def test_a_genuinely_unconverged_row_is_still_refused():
    # PLANTED: 1e-11 A is ~100x the floor FLOOR_SAFETY eps * 7.4 A (~1.05e-13 A): a real error, not rounding.
    assert not _accepts(declares=True, small_error=1e-11)
