"""NewtonKrylov may precondition its GMRES (numerics-lab #36); unpreconditioned stays the selectable default."""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from tests.unit.test_nonlinear import _LinearSystem

from numerics_lab.nonlinear import NewtonKrylovStrategy
from numerics_lab.nonlinear import _strategies_newton as owner

_N = 40


def _matrix():
    """A nonsymmetric, badly scaled tridiagonal operator: unpreconditioned GMRES needs many iterations."""
    main = np.logspace(0, 6, _N)
    return sp.diags([main, 0.3 * main[:-1], -0.2 * main[1:]], [0, 1, -1], format='csr')


class _Sparse(_LinearSystem):
    def __init__(self, target) -> None:
        super().__init__(target)
        self.matrix = _matrix()
        self.rhs = self.matrix @ self.target

    def residual(self, x, _state, /):
        return self.matrix @ x - self.rhs

    def tangent(self, _x, _state, /):
        return self.matrix


def _counted(monkeypatch):
    seen = []
    original = owner.gmres

    def gmres(op, b, **kw):
        seen.append(kw.get('M'))
        return original(op, b, **kw)

    monkeypatch.setattr(owner, 'gmres', gmres)
    return seen


def test_the_default_runs_gmres_unpreconditioned(monkeypatch):
    """PLANTED CONTROL: no preconditioner declared -> GMRES gets none, as before."""
    seen = _counted(monkeypatch)
    target = np.linspace(1.0, 2.0, _N)
    NewtonKrylovStrategy().solve(_Sparse(target), max_iter=3)
    assert seen == [None]


@pytest.mark.parametrize('name', ['ilu', 'jacobi'])
def test_a_named_preconditioner_is_built_on_the_exact_jacobian(monkeypatch, name):
    seen = _counted(monkeypatch)
    target = np.linspace(1.0, 2.0, _N)
    result = NewtonKrylovStrategy(preconditioner=name).solve(_Sparse(target), max_iter=3)
    assert len(seen) == 1
    assert isinstance(seen[0], LinearOperator)
    np.testing.assert_allclose(result.x, target, rtol=1e-6)


def test_ilu_reaches_the_root_the_unpreconditioned_step_misses():
    """The point of #36: on a badly scaled operator one preconditioned step lands where the plain one does not."""
    target = np.linspace(1.0, 2.0, _N)
    plain = NewtonKrylovStrategy().solve(_Sparse(target), max_iter=3).x
    ilu = NewtonKrylovStrategy(preconditioner='ilu').solve(_Sparse(target), max_iter=3).x
    assert np.max(np.abs(ilu - target)) < np.max(np.abs(plain - target))


def test_an_unknown_preconditioner_is_refused_by_name():
    with pytest.raises(KeyError, match='unknown preconditioner'):
        NewtonKrylovStrategy(preconditioner='nope').solve(_Sparse(np.ones(_N)), max_iter=2)
