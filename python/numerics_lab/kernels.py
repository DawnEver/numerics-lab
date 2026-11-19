"""Small native kernels with no Python twin: the dense transcriptions and the block algebra.

Moved from motronics-studio on 2026-11-19 (``crates/numerics/src/dense/``, ``linear/direct/fill.rs``,
``linear/condense.rs``); each is the Rust kernel, reached here so a consumer imports a public module,
never ``_native``.

* :func:`expm` -- ``scipy.linalg.expm`` bit for bit up to :data:`EXPM_BIT_IDENTICAL_MAX_N`, round-off past it.
* :func:`dense_solve` / :func:`dense_cond` -- ``np.linalg.solve`` / ``np.linalg.cond`` (``n <= COND_MAX_N``) bit for bit.
* :func:`cholesky_factor_len` -- ``nnz(L)`` of the AMD Cholesky factor from the symbolic analysis alone.
* :func:`condense` / :class:`CondensedCells` / :func:`block_matvec` -- block static condensation.
* :func:`guard_scan` / :func:`first_crossing` / :func:`span_key` -- events on the exact trajectory ``expm(M s) z0``
  (``crates/numerics/src/lti/``).
"""

from __future__ import annotations

from numerics_lab._native import (
    COND_MAX_N,
    EXPM_BIT_IDENTICAL_MAX_N,
    SPAN_KEY_DIGITS,
    CondensedCells,
    block_matvec,
    cholesky_factor_len,
    condense,
    dense_cond,
    dense_solve,
    expm,
    first_crossing,
    guard_scan,
    span_key,
)

__all__ = [
    'COND_MAX_N',
    'EXPM_BIT_IDENTICAL_MAX_N',
    'SPAN_KEY_DIGITS',
    'CondensedCells',
    'block_matvec',
    'cholesky_factor_len',
    'condense',
    'dense_cond',
    'dense_solve',
    'expm',
    'first_crossing',
    'guard_scan',
    'span_key',
]
