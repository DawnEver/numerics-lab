"""On a deeply saturating B-H slab, a strategy that declares convergence holds the Newton field.

THE DEFECTS THIS PINS (measured downstream 2026-11-06 on an FEA machine, reproduced here):

* `coefficient-damped` declared convergence with the field still off: it relaxes the
  COEFFICIENTS, so under a small alpha the iterate creeps while the mixed reluctivity is still far
  from the one the iterate implies, and the iterate's increment alone reads settled.
* `anderson` diverged, and on milder sources declared convergence at 82 % field error: its mix
  can land next to the last iterate while that iterate is far from its own Picard image.
* `modified-newton` diverged: a freshly refreshed tangent's full step from a zero field overshoots
  deep into saturation, and nothing searched along it.

THE FIXTURE. A 1-D slab, ``-(nu(B^2) A')' = J`` with ``A = 0`` at both ends, linear elements,
an air gap in the middle, and an iron law that saturates TO VACUUM (``nu`` from ``1/mu_r`` to 1),
so a substitution from a zero field lands ~180x past the solution -- the deep-saturation shape.
Full Newton (a residual stop at 1e-8) is the reference field.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from numerics_lab.nonlinear import AssembledState, get_strategy

_RTOL_ENERGY_PCT = 0.01  # the production fixed-point bar: increment and field held to 1e-4
_FIELD_BAR = _RTOL_ENERGY_PCT / 100.0
_MAX_ITER = 100
_GX, _GW = np.polynomial.legendre.leggauss(12)


class _SaturatingSlab:
    """The slab, carrying the full seam: raw/fold (coefficient-damped), tangent, secant, source scaling."""

    is_linear = False

    def __init__(self, *, j: float, n: int = 40, mu_r: float = 1000.0, bs: float = 1.8, p: int = 4) -> None:
        self.j, self.n, self.mu_r, self.bs, self.p = j, n, mu_r, bs, p
        self.h = 1.0 / n
        self.air = np.zeros(n, dtype=bool)
        self.air[n // 2 - 1 : n // 2 + 1] = True
        self.b = np.full(n - 1, j / n)
        # Element e couples free nodes e-1 and e; the boundary nodes 0 and n are dropped.
        rows, cols, elems, signs = [], [], [], []
        for e in range(n):
            for a in (e - 1, e):
                for c in (e - 1, e):
                    if 0 <= a < n - 1 and 0 <= c < n - 1:
                        rows.append(a)
                        cols.append(c)
                        elems.append(e)
                        signs.append(1.0 if a == c else -1.0)
        self._rows, self._cols = np.array(rows), np.array(cols)
        self._elems, self._signs = np.array(elems), np.array(signs)

    def scale_source(self, lam: float, /):
        return _SaturatingSlab(j=self.j * lam, n=self.n, mu_r=self.mu_r, bs=self.bs, p=self.p)

    def initial_guess(self) -> np.ndarray:
        return np.zeros(self.n - 1)

    def _b2(self, x) -> np.ndarray:
        return (np.diff(np.concatenate([[0.0], x, [0.0]])) / self.h) ** 2

    def _nu(self, b2):
        s = b2**self.p / (b2**self.p + self.bs ** (2 * self.p))
        return np.where(self.air, 1.0, 1.0 / self.mu_r + (1.0 - 1.0 / self.mu_r) * s)

    def _dnu(self, b2):
        bs2p = self.bs ** (2 * self.p)
        d = self.p * b2 ** (self.p - 1) * bs2p / (b2**self.p + bs2p) ** 2
        return np.where(self.air, 0.0, (1.0 - 1.0 / self.mu_r) * d)

    def _energy(self, b2) -> float:
        # sum_e h * int_0^B H dB = h/2 * int_0^{B^2} nu(t) dt, by Gauss-Legendre.
        t = 0.5 * (_GX[:, None] + 1.0) * b2[None, :]
        integral = 0.5 * b2 * np.sum(_GW[:, None] * np.vstack([self._nu(row) for row in t]), axis=0)
        return float(np.sum(0.5 * self.h * integral))

    def _matrix(self, weights) -> sp.csr_matrix:
        vals = weights[self._elems] * self._signs / self.h
        return sp.csr_matrix((vals, (self._rows, self._cols)), shape=(self.n - 1, self.n - 1))

    def raw_state(self, x):
        b2 = self._b2(x)
        vals = self._nu(b2)[self._elems] * self._signs / self.h
        return self._rows, self._cols, vals, self.b.copy(), self._energy(b2)

    def fold_raw(self, s_row, s_col, s_val, t_mat, total_energy) -> AssembledState:
        matrix = sp.csr_matrix((s_val, (s_row, s_col)), shape=(self.n - 1, self.n - 1))
        return AssembledState(system_matrix=(matrix, np.asarray(t_mat, dtype=float)), total_energy=total_energy)

    def assemble(self, x) -> AssembledState:
        return self.fold_raw(*self.raw_state(x))

    def solve_linear(self, state):
        matrix, rhs = state.system_matrix
        return spsolve(matrix.tocsc(), rhs)

    def solve_secant(self, state, rhs):
        return spsolve(state.system_matrix[0].tocsc(), rhs)

    def residual(self, x, state):
        matrix, rhs = state.system_matrix
        return matrix @ x - rhs

    def residual_norm(self, x, state) -> float:
        return float(np.linalg.norm(self.residual(x, state)))

    def rhs_norm(self, state) -> float:
        return float(np.linalg.norm(state.system_matrix[1]))

    def tangent(self, x, state):
        b2 = self._b2(x)
        return self._matrix(self._nu(b2) + 2.0 * b2 * self._dnu(b2))

    def solve_system(self, matrix, rhs):
        return spsolve(sp.csc_matrix(matrix), rhs)

    def on_iteration(self, *_args) -> None:
        return None


def _newton_field(**slab) -> np.ndarray:
    result = get_strategy('newton').solve(_SaturatingSlab(**slab), max_iter=_MAX_ITER, rtol_energy_pct=1e-6)
    assert result.converged, 'the reference itself must converge'
    return result.x


def _field_error(x, reference) -> float:
    return float(np.abs(x - reference).max() / np.abs(reference).max())


@pytest.mark.parametrize('j', [1.0, 3.0, 6.0, 12.0])
def test_the_slab_is_deeply_saturated_and_newton_solves_it(j):
    """THE FLOOR: the plant must be saturated, else the assertions below are about a linear map."""
    x = _newton_field(j=j)
    slab = _SaturatingSlab(j=j)
    b_max = float(np.sqrt(slab._b2(x)[~slab.air].max()))
    assert b_max > 1.5, f'iron peak {b_max:.2f} T is not into the knee of a 1.8 T law'
    first = slab.solve_linear(slab.assemble(slab.initial_guess()))
    assert _field_error(first, x) > 10.0, 'a substitution from zero must overshoot deep into saturation'


def test_coefficient_damped_does_not_declare_convergence_off_the_field():
    """Measured before the fix: converged at iteration 9 with the field 9.2e-4 off, 9x its own bar.

    The mixed coefficients lagged the iterate's own by far more than the bar while the iterate's
    increment read settled. After: the coefficient lag is held to the increment's bar, and the
    solve stops at 16 iterations, 1.0e-5 off.
    """
    slab = {'j': 20.0, 'n': 120}
    reference = _newton_field(**slab)
    result = get_strategy('coefficient-damped').solve(
        _SaturatingSlab(**slab), max_iter=_MAX_ITER, rtol_energy_pct=_RTOL_ENERGY_PCT
    )
    assert result.converged, 'the slab is solvable by coefficient-damped within the cap'
    assert _field_error(result.x, reference) < _FIELD_BAR


@pytest.mark.parametrize('j', [0.2, 1.0, 3.0, 6.0, 12.0])
@pytest.mark.parametrize('name', ['anderson', 'modified-newton'])
def test_a_globalised_strategy_converges_to_the_newton_field(name, j):
    """Where Newton converges, anderson and modified-newton converge to the same field.

    Measured before the fix, field error: anderson 1.4 (capped) / 0.82 / 0.55 at j = 0.2 / 1 / 3,
    the last two DECLARED CONVERGED; modified-newton 43 / 65 / 16 at the cap. Every cell red
    before. After: every cell converged within the bar, anderson in 35-45 iterations and
    modified-newton in 10-15.
    """
    reference = _newton_field(j=j)
    result = get_strategy(name).solve(_SaturatingSlab(j=j), max_iter=_MAX_ITER, rtol_energy_pct=_RTOL_ENERGY_PCT)
    assert result.converged
    assert _field_error(result.x, reference) < _FIELD_BAR
