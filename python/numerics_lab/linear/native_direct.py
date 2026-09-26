"""The native direct factorisation behind the ``direct`` backend -- faer, pattern-cached.

``motronics_native.gauss.DirectFactorization`` (``rust/crates/gauss/src/linear/direct.rs``) is a
supernodal LU, or a supernodal Cholesky when the operator is verified EXACTLY symmetric, whose
symbolic analysis is paid once per sparsity pattern and whose numeric storage is reused. A Newton
loop reassembles one mesh, so its pattern is constant and only the values move; this module holds
one analysis per pattern and refactorises it per solve.

WHY IT IS THE DEFAULT, measured (2026-09-26, the faer-bench sweep in
``.claude/memory/2026/09/26/faer-loses-small-systems-because-auto-picks-simplicial.md``): with
the three levers ``direct.rs`` pulls, faer beats SciPy's SuperLU at every production size measured
(5k to 129k unknowns), so there is no size threshold.

THE PATTERN IS CHECKED, NEVER ASSUMED. The cache key is a CRC of ``indptr``/``indices``, which is a
lookup, not a proof: the extension compares the full pattern on every refactor and refuses a
different one, and a refusal here re-analyses. A HELD factor (:class:`NativeFactor`, what
``factor_matrix`` hands a held-factor Newton or a march cache) shares its analysis with later solves
on the same pattern, so it records the generation it was factorised at and re-factorises its own
values if another solve has since overwritten the shared storage -- a stale factor is never
back-substituted.

THE FALLBACK IS VISIBLE. When the extension is absent, predates ``refactor``, or is switched off
with ``MOTRONICS_DIRECT_NATIVE=0`` (the A/B selector, an env var for the same reason as
``fem/_native.py``'s), the ``direct`` backend runs SciPy ``spsolve`` and says so: one WARNING per
process naming the reason, and every solve is counted by backend in :data:`DIRECT_SOLVES`. A
complex or non-sparse operator is outside the native's reach and is counted as ``superlu`` too.
"""

from __future__ import annotations

import logging
import os
import zlib
from collections import Counter, OrderedDict

import numpy as np
import scipy.sparse as sp

from motronics.gauss.linear.errors import LinearSolveError

__all__ = ['DIRECT_SOLVES', 'NativeFactor', 'native_available', 'native_factor', 'native_solve']

logger = logging.getLogger(__name__)

#: ``MOTRONICS_DIRECT_NATIVE=0`` forces SciPy SuperLU -- the A/B selector, read per call.
_ENV = 'MOTRONICS_DIRECT_NATIVE'

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
    try:
        from motronics_native import gauss  # noqa: PLC0415 -- optional native extension
    except ImportError as exc:
        return None, f'motronics_native.gauss is not importable ({exc})'
    cls = getattr(gauss, 'DirectFactorization', None)
    if cls is None or not hasattr(cls, 'refactor'):
        return None, 'the installed motronics_native predates DirectFactorization.refactor'
    return cls, None


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

    __slots__ = ('factorization', 'generation')

    def __init__(self, factorization) -> None:
        self.factorization = factorization
        self.generation = 0


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
    key = (csr.shape[0], csr.nnz, zlib.crc32(indptr.tobytes()), zlib.crc32(indices.tobytes()))
    entry = _cache.get(key)
    if entry is not None:
        _cache.move_to_end(key)
        try:
            kind = entry.factorization.refactor(csr.data)
        except ValueError as exc:
            if 'PatternChanged' not in str(exc):
                raise _singular(exc, csr.shape[0]) from exc
            entry = None  # a CRC collision: the extension's full comparison refused it
        else:
            entry.generation += 1
            return entry, kind
    try:
        factorization = cls(csr.shape[0], csr.shape[1], indptr, indices, csr.data)
        kind = factorization.refactor(csr.data)
    except ValueError as exc:
        raise _singular(exc, csr.shape[0]) from exc
    entry = _Entry(factorization)
    entry.generation = 1
    _cache[key] = entry
    while len(_cache) > _CACHE_SIZE:
        _cache.popitem(last=False)
    return entry, kind


def _singular(exc: Exception, n: int) -> LinearSolveError:
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

    def __init__(self, cls, csr, entry: _Entry, kind: str) -> None:
        self._cls = cls
        self._csr = csr
        self._entry = entry
        self._generation = entry.generation
        self.kind = kind

    def solve(self, b) -> np.ndarray:
        if self._entry.generation != self._generation:
            self._entry, self.kind = _refactor(self._cls, self._csr)
            self._generation = self._entry.generation
        DIRECT_SOLVES[f'faer-{self.kind}'] += 1
        return _back_substitute(self._entry.factorization, b, self._csr.shape[0])


def native_factor(matrix) -> NativeFactor | None:
    """Factorise ``matrix`` natively for repeated back-substitution, or ``None`` (logged)."""
    cls, reason = _extension_or_reason()
    if cls is None:
        _fallback(reason)
        return None
    csr = _canonical(matrix)
    if csr is None:
        _fallback('the operator is complex, dense or non-square (outside the native reach)')
        return None
    # The held factor keeps its values across other solves, so they must not alias a buffer the
    # caller may go on to mutate.
    csr = csr.copy() if csr is matrix else csr
    entry, kind = _refactor(cls, csr)
    return NativeFactor(cls, csr, entry, kind)
