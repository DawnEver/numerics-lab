"""The native direct factorisation reports an allocation failure as one, and lets a caller release its patterns.

2026-10-20, an 806,777-unknown fixed-secant seed: faer's ``OutOfMemory`` was reported as "the linear system
is rank-deficient -- is a gauge/reference set?", sending the reader after a gauge. And the pattern cache kept
the seed's factor alive through the coupled Newton that followed it, because a count bound cannot tell a
pattern that will be solved again from one that will not -- only the caller knows that.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from numerics_lab.linear import LinearSolveError, native_direct
from numerics_lab.linear.errors import DirectSolveOutOfMemory


class _Refusing:
    """A stand-in extension whose refactor raises what the binding raises for ``detail``."""

    detail = ''

    def __init__(self, *_args) -> None:
        pass

    def refactor(self, _values):
        raise ValueError(self.detail)


def _system(n: int = 6):
    return sp.diags([np.full(n - 1, -1.0), np.full(n, 4.0), np.full(n - 1, -1.5)], [-1, 0, 1], format='csr')


def test_an_allocation_failure_is_out_of_memory_and_never_rank_deficient(monkeypatch) -> None:
    monkeypatch.setattr(_Refusing, 'detail', 'OutOfMemory { stage: "lu numeric" }')
    monkeypatch.setattr(native_direct, '_extension_or_reason', lambda: (_Refusing, None))
    native_direct._cache.clear()
    with pytest.raises(DirectSolveOutOfMemory) as caught:
        native_direct.native_solve(_system(), np.ones(6))
    assert isinstance(caught.value, MemoryError)
    assert isinstance(caught.value, LinearSolveError)  # every existing handler still sees it
    assert 'lu numeric' in str(caught.value)
    assert 'rank-deficient' not in str(caught.value)


def test_a_singular_refusal_is_still_rank_deficient(monkeypatch) -> None:
    # The planted control: the OOM branch must not swallow the refusal it was split from.
    monkeypatch.setattr(_Refusing, 'detail', 'Singular { detail: "lu numeric: SymbolicSingular" }')
    monkeypatch.setattr(native_direct, '_extension_or_reason', lambda: (_Refusing, None))
    native_direct._cache.clear()
    with pytest.raises(LinearSolveError, match='rank-deficient') as caught:
        native_direct.native_solve(_system(), np.ones(6))
    assert not isinstance(caught.value, MemoryError)


def test_a_scope_releases_the_patterns_analysed_inside_it_and_keeps_the_rest() -> None:
    native_direct._cache.clear()
    outer = _system(8)
    native_direct.native_solve(outer, np.ones(8))
    with native_direct.released_on_exit():
        native_direct.native_solve(_system(6), np.ones(6))
        native_direct.native_solve(outer, np.ones(8))  # reused, not analysed here: it stays
        assert len(native_direct._cache) == 2
    assert [key[0] for key in native_direct._cache] == [8]


def test_a_held_factor_outlives_the_scope_that_released_its_pattern() -> None:
    native_direct._cache.clear()
    a = _system(6)
    with native_direct.released_on_exit():
        held = native_direct.native_factor(a)
    assert held is not None
    assert not native_direct._cache
    np.testing.assert_allclose(a @ held.solve(np.ones(6)), np.ones(6), rtol=1e-12)
