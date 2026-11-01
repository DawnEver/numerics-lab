---
name: affine-work-capacity-is-not-a-rank-acceptance-threshold
description: Share a finite 2048-column/32-MiB coefficient-array work envelope; preserve the actual pivot, error-bound, forcing and original residual criteria.
metadata:
  type: project
---

# Measured work capacity, unchanged rank authority

Actual application Schur inputs are independently captured/audited:1926rows,
1927columns,8arrays/9659816B. Its support has630components,629empty one-column
components and ONE1926x1298component. That component has2499948dense entries:
about19.1MiB per float64 coefficient array. Both the old512column and512**2entry
work guards reject it BEFORE the numerical rank algorithm runs. This is an
unsupported implementation capacity, not evidence of a singular physical model.

The bounded affine solver and joint caller now share MAX_AFFINE_COLUMNS2048
and MAX_AFFINE_ENTRIES2048**2, hence at most32MiB per coefficient array. No
per-case override or process-memory increase. Direct affine calls ALSO enforce
the entry ceiling, including very tall thin matrices; the old direct API only
bounded columns. Shape/work validation precedes copying the coefficient arrays.
Every existing pivot, entry-error reliability factor, homogeneous proof, forcing
allowance and original residual test remains unchanged. The optional homogeneous
exact-block proof retains its own512-column algorithm limit; do not silently
raise that different algorithm's work budget.

Canonical117items/5RED plus formatting ->117GREEN ->118GREEN with a genuinely
DRIVEN513-column connected border. Exact same original driven rows and unique
answer pass, not a homogeneous-zero substitute. Real connected513/1298-column
homogeneous systems pass through actual component discovery and actual affine
rank authority. Inputs above the shared column OR entry envelope still refuse.
Includes original joint cancellation, bounded elimination, projection/rate/
factor-lifetime/ordering regressions. Whole Ruff/265format checks pass.
NormalCPU1/1GiB/300s, peak262455296B, wall14.8381702s.
Receipts output/logs/26/10/31/border-capacity-{red52,green52,final52}.

These are implementation tests, not successful application rank/initialization.
Next replay the ACTUAL captured component under the unchanged4GiB limit and
independently validate its original Schur rows. Both modified numerical modules
must be explicitly admitted; never load the new caller with an old512-column
border_rank function through a stale from-import alias.
