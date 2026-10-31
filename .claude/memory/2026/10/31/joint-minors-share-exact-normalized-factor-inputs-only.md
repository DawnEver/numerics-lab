---
name: joint-minors-share-exact-normalized-factor-inputs-only
description: Adjacent joint minors share LU only when their actual canonical normalized CSC inputs are identical; original operators, RHS scaling, refinement and rank refusal stay separate.
metadata:
  type: project
---

# Factor the operator actually supplied to LU once

_triangular_joint_solver formerly compared raw storage/residual and state/rate
minors, then factored row-normalized operators. Distinct raw minors can supply
IDENTICAL normalized CSC input to LU, yet each retained a separate factor.

One construction-local previous operator/factor pair now compares exact shape,
dtype, canonical CSC indptr/indices and coefficient values. Only exact equality
shares the factor: no allclose, tolerance, rank estimate or physical row removal.
Canonicalization is the same sum/sort the existing SuperLU entry performs.
Every original minor gets its OWN solve/refinement closure and original RHS
normalization. Reusing the raw solve closure across differently scaled minors
would be wrong; only the actual factor is shared.

The comparison key is construction-local, not a global cache or extra retained
matrix in the returned solve closure. Weakref controls prove all key CSC owners
retire without forced GC, while real factors still solve the original system.

Canonical private TDD real6x6joint equations:3tests/2fail RED (old four factors)
->53GREEN including original joint coordinate closure/rate/lifetime/order
regressions, then54GREEN with single-ULP non-reuse control. Row-scaled exact
factor inputs use2factors, a genuinely different input uses3, one ULP ALSO
uses3, and a genuinely singular different minor still refuses via actual LU.
All RHS samples agree with full original dense6x6 solve and original rows;
no planted passing answer. Whole-repo Ruff/263formatchecks pass. Normal private
CPU1/1GiB/300s verify with exact native/source paths and all threads1.
Receipts output/logs/26/10/31/normalized-joint-factors-{red44,green44,final44}.

The motivating application's four-GiB run really passed the previous all-row
rate retry, then hit a NEW joint correction peak while retaining78842/179726
factors, before any complete initial pair. Whether its actual remaining minors
have identical normalized inputs is UNMEASURED. This source change does NOT
claim application memory fit, full initialization, integration or physical
reference accuracy. Its real new source requires a new explicit application
admission and unchanged-budget native qualification. Do not raise the cap.
