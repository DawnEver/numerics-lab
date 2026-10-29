"""The native direct factorisation behind the ``direct`` backend -- faer, pattern-cached.

``numerics_lab._native.DirectFactorization`` (``crates/numerics/src/linear/direct.rs``) is a
supernodal LU, or a supernodal Cholesky when the operator is verified EXACTLY symmetric, whose
symbolic analysis is paid once per sparsity pattern and whose numeric storage is reused. A Newton
loop reassembles one mesh, so its pattern is constant and only the values move; this module holds
one analysis per pattern and refactorises it per solve.

WHY IT IS THE DEFAULT, measured (2026-09-26, the faer-bench sweep in
``.claude/memory/2026/09/26/faer-loses-small-systems-because-auto-picks-simplicial.md``): with
the three levers ``direct.rs`` pulls, faer beats SciPy's SuperLU at every production size measured
(5k to 129k unknowns), so there is no size threshold.

THE PATTERN IS CHECKED, NEVER ASSUMED. The cache key contains the complete canonical
``indptr``/``indices`` bytes: dictionary hash collisions still require exact byte equality.
The native values-only refactor therefore always receives its analysed pattern. A HELD factor (:class:`NativeFactor`, what
``factor_matrix`` hands a held-factor Newton or a march cache) shares its analysis with later solves
on the same pattern, so it records the generation it was factorised at and re-factorises its own
values if another solve has since overwritten the shared storage -- a stale factor is never
back-substituted.

THE FALLBACK IS VISIBLE. When the native path is switched off with
``NUMERICS_LAB_DIRECT_NATIVE=0`` (the A/B selector), the ``direct`` backend runs SciPy ``spsolve``
and says so: one WARNING per
process naming the reason, and every solve is counted by backend in :data:`DIRECT_SOLVES`. A
complex or non-sparse operator is outside the native's reach and is counted as ``superlu`` too.
"""

from __future__ import annotations

import logging
import os
from collections import Counter, OrderedDict
from contextlib import contextmanager

import numpy as np
import scipy.sparse as sp

from numerics_lab import _native
from numerics_lab.linear.constrained.fixed_values import lift_dirichlet_rhs
from numerics_lab.linear.errors import DirectSolveOutOfMemory, LinearSolveError

__all__ = ['DIRECT_SOLVES', 'NativeFactor', 'native_available', 'native_factor', 'native_solve', 'released_on_exit']

logger = logging.getLogger(__name__)

#: ``NUMERICS_LAB_DIRECT_NATIVE=0`` forces SciPy SuperLU -- the A/B selector, read per call.
_ENV = 'NUMERICS_LAB_DIRECT_NATIVE'

#: Analyses held at once. A Newton loop alternates at most a secant and a tangent pattern, a
#: coupled march adds one more; beyond that an evicted pattern simply re-analyses.
_CACHE_SIZE = 3

#: Solves by the backend that answered them: ``faer-lu``, ``faer-cholesky`` or ``superlu``.
DIRECT_SOLVES: Counter[str] = Counter()

_warned: set[str] = set()
_cache: OrderedDict[tuple, _Entry] = OrderedDict()


def _extension_or_reason():
    """``(DirectFactorization class, None)`` or ``(None, why the native path cannot run)``."""
    if os.environ.get(_ENV, '1') == '0':
        return None, f'{_ENV}=0'
    return _native.DirectFactorization, None


def native_available() -> bool:
    """Whether the native factorisation would serve a real sparse solve right now."""
    return _extension_or_reason()[0] is not None


def _fallback(reason: str) -> None:
    DIRECT_SOLVES['superlu'] += 1
    if reason not in _warned:
        _warned.add(reason)
        logger.warning('direct linear solve falls back to SciPy SuperLU: %s', reason)


class _Entry:
    """One analysed pattern and the generation of its last refactor."""

    __slots__ = ('factorization', 'generation', 'indices', 'indptr', 'shape')

    def __init__(self, factorization, csr) -> None:
        self.factorization = factorization
        self.generation = 0
        self.shape = csr.shape
        self.indptr = csr.indptr.copy()
        self.indices = csr.indices.copy()


@contextmanager
def released_on_exit():
    """Drop, on exit, every pattern this scope ANALYSED; patterns it only reused stay cached.

    The count bound cannot tell a pattern that will be solved again from one that will not; the caller
    can. A stage that ends for good -- a fixed-secant seed handing its iterate to a coupled Newton on a
    different system -- wraps itself in this so its factor is not held beside the next stage's (measured
    2026-10-20: an 806,777-unknown seed's factor is several GiB). A :class:`NativeFactor` the scope handed
    out keeps its own reference and stays usable; it is only no longer shared.
    """
    before = set(_cache)
    try:
        yield
    finally:
        for key in [key for key in _cache if key not in before]:
            del _cache[key]


def _canonical(matrix):
    """A canonical float64 CSR of ``matrix``, or ``None`` when the native cannot take it."""
    if not sp.issparse(matrix) or np.issubdtype(matrix.dtype, np.complexfloating):
        return None
    csr = matrix.tocsr() if matrix.format != 'csr' else matrix
    if csr.shape[0] != csr.shape[1]:
        return None
    if not csr.has_canonical_format:
        csr = csr.copy()
        csr.sum_duplicates()
    if csr.dtype != np.float64:
        csr = csr.astype(np.float64)
    return csr


def _refactor(cls, csr) -> tuple[_Entry, str]:
    """Refactorise ``csr`` into the cached analysis of its pattern (analysing it if new)."""
    indptr = np.asarray(csr.indptr, dtype=np.int64)
    indices = np.asarray(csr.indices, dtype=np.int64)
    key = (csr.shape[0], indptr.tobytes(), indices.tobytes())
    entry = _cache.get(key)
    if entry is not None and (
        entry.shape != csr.shape
        or not np.array_equal(entry.indptr, indptr)
        or not np.array_equal(entry.indices, indices)
    ):
        entry = None
    if entry is not None:
        _cache.move_to_end(key)
        try:
            kind = entry.factorization.refactor(csr.data)
        except ValueError as exc:
            if 'PatternChanged' not in str(exc):
                raise _singular(exc, csr.shape[0]) from exc
            entry = None  # defensive support for an extension that also validates its pattern
        else:
            entry.generation += 1
            return entry, kind
    try:
        factorization = cls(csr.shape[0], csr.shape[1], indptr, indices, csr.data)
        kind = factorization.refactor(csr.data)
    except ValueError as exc:
        raise _singular(exc, csr.shape[0]) from exc
    entry = _Entry(factorization, csr)
    entry.generation = 1
    _cache[key] = entry
    while len(_cache) > _CACHE_SIZE:
        _cache.popitem(last=False)
    return entry, kind


def _singular(exc: Exception, n: int) -> LinearSolveError:
    """The refusal ``exc`` IS: the binding spells ``DirectFailure`` by its variant name, so an allocation
    failure arrives as ``OutOfMemory { stage: ... }`` and is never reported as a rank statement.
    """
    if str(exc).startswith('OutOfMemory'):
        return DirectSolveOutOfMemory(
            f'direct solver faer ran out of memory factorising the n={n} system ({exc}); the factor does not '
            'fit in the memory this process can get -- this says nothing about the rank of the system'
        )
    return LinearSolveError(
        f'direct solver faer refused the n={n} system ({exc}); the linear system is rank-deficient '
        '-- is a gauge/reference set and is the (loop) basis full-rank at this mesh resolution?'
    )


def _back_substitute(factorization, b: np.ndarray, n: int) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64)
    if b.ndim == 1:
        return np.asarray(factorization.back_substitute(np.ascontiguousarray(b), 1))
    k = b.shape[1]
    flat = np.asarray(factorization.back_substitute(np.ravel(b, order='F'), k))
    return flat.reshape((n, k), order='F')


def native_solve(matrix, rhs) -> np.ndarray | None:
    """``matrix^{-1} rhs`` through the native factorisation, or ``None`` when it cannot serve.

    ``None`` is the caller's cue to run SciPy; the reason has already been logged and counted.
    """
    cls, reason = _extension_or_reason()
    if cls is None:
        assert reason is not None
        _fallback(reason)
        return None
    csr = _canonical(matrix)
    if csr is None or sp.issparse(rhs) or np.iscomplexobj(rhs):
        _fallback('the operator or rhs is complex, dense or non-square (outside the native reach)')
        return None
    entry, kind = _refactor(cls, csr)
    DIRECT_SOLVES[f'faer-{kind}'] += 1
    x = _back_substitute(entry.factorization, rhs, csr.shape[0])
    if not np.all(np.isfinite(x)):
        # The BELT, as SuperLU's promoted MatrixRankWarning is on the SciPy half: a zero pivot the
        # LU did not refuse surfaces as inf/NaN, and it is named as the singular system it is.
        msg = (
            f'direct solver faer reports the n={csr.shape[0]} matrix is (near-)singular (a {kind} '
            'factorisation produced a non-finite solution); the linear system is rank-deficient -- '
            'is a gauge/reference set and is the (loop) basis full-rank at this mesh resolution?'
        )
        raise LinearSolveError(msg)
    return x


class NativeFactor:
    """A held native factor with SciPy ``splu``'s ``solve(b)`` shape (``b`` 1-D or 2-D).

    It shares the pattern's analysis with every other solve on that pattern, so it keeps its own
    VALUES and the generation it was factorised at; a back-substitution after some other solve has
    refactorised the shared storage first re-factorises these values, never answers from theirs.
    """

    def __init__(self, cls, csr, entry: _Entry, kind: str, *, original=None, pinned=None, free=None) -> None:
        self._cls = cls
        self._csr = csr
        self._entry = entry
        self._generation = entry.generation
        self.kind = kind
        self._original = original
        self._pinned = pinned
        self._free = free

    def solve(self, b) -> np.ndarray:
        if self._entry.generation != self._generation:
            self._entry, self.kind = _refactor(self._cls, self._csr)
            self._generation = self._entry.generation
        DIRECT_SOLVES[f'faer-{self.kind}'] += 1
        if self._original is not None:
            assert self._pinned is not None and self._free is not None
            b = np.asarray(b, dtype=float)
            lifted = lift_dirichlet_rhs(self._original, b, self._pinned, b[self._pinned])
            lifted[self._free] = _back_substitute(self._entry.factorization, lifted[self._free], self._csr.shape[0])
            return lifted
        return _back_substitute(self._entry.factorization, b, self._csr.shape[0])


def native_factor(matrix) -> NativeFactor | None:
    """Factorise ``matrix`` natively for repeated back-substitution, or ``None`` (logged)."""
    cls, reason = _extension_or_reason()
    if cls is None:
        assert reason is not None
        _fallback(reason)
        return None
    csr = _canonical(matrix)
    if csr is None:
        _fallback('the operator is complex, dense or non-square (outside the native reach)')
        return None
    # The held factor keeps its values across other solves, so they must not alias a buffer the
    # caller may go on to mutate.
    csr = csr.copy() if csr is matrix else csr
    # Exact algebraic unit rows specify their unknown directly. Eliminating them is a solve
    # coordinate change, not a change to the full residual or a declaration that it is symmetric.
    lengths = np.diff(csr.indptr)
    candidates = np.flatnonzero(lengths == 1)
    slots = csr.indptr[candidates]
    pinned = candidates[(csr.indices[slots] == candidates) & (csr.data[slots] == 1.0)]
    if 0 < pinned.size < csr.shape[0]:
        free = np.setdiff1d(np.arange(csr.shape[0]), pinned, assume_unique=True)
        reduced = csr[free][:, free].tocsr()
        if (reduced != reduced.T).nnz == 0:
            entry, kind = _refactor(cls, reduced)
            return NativeFactor(cls, reduced, entry, kind, original=csr, pinned=pinned, free=free)
    entry, kind = _refactor(cls, csr)
    return NativeFactor(cls, csr, entry, kind)
