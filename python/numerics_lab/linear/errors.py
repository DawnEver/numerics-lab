"""The linear seam's own error type -- one exception class, and a TRUE LEAF.

Split out of :mod:`numerics_lab.linear` on 2026-09-17 so that the two halves which RAISE it can
both import it: the backends (``direct`` / the Krylov wrappers, which promote SuperLU's
``MatrixRankWarning`` from a warning into a failure) and the direct-solve guard in
:mod:`numerics_lab.linear.refinement` (which refuses a FINITE vector that does not solve its
system). Defining it in either half would put a cycle between them; defining it in a third place
that imports nothing is what makes this name reachable in one hop from both.

Re-exported UNCHANGED from :mod:`numerics_lab.linear`, so the spelling every caller in the tree
uses -- ``from numerics_lab.linear import LinearSolveError`` -- is unaffected.
"""

from __future__ import annotations

__all__ = ['DirectSolveOutOfMemory', 'LinearSolveError']


class LinearSolveError(RuntimeError):
    """A linear backend failed to produce a finite solution (singular system / Krylov breakdown).

    Raised instead of returning silent NaNs, mirroring the MEC singular-system guard: a
    reference-free network or a Krylov breakdown is a hard failure the caller must see.
    """


class DirectSolveOutOfMemory(LinearSolveError, MemoryError):
    """The direct factorisation could not get the memory its factor (or its scratch) needs.

    A ``LinearSolveError`` so every existing handler still sees a failed solve, and a ``MemoryError`` so
    nobody reads it as a statement about the operator: it says NOTHING about the rank. Split out on
    2026-10-20, when an 806,777-unknown seed's ``OutOfMemory`` was reported as "rank-deficient -- is a
    gauge/reference set?".
    """
