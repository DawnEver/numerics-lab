---
name: all-four-public-kkt-orderings-are-now-characterized
description: MMD_ATA and NATURAL preserve original all-row numerical controls but do not improve the bounded FE-grid fill; no production change or physical rerun.
metadata:
  type: project
---

# All four public KKT orderings are characterized

Extend the existing bounded comparison recorder from default COLAMD and
MMD_AT_PLUS_A to MMD_ATA and NATURAL. These are the four documented public
[SuperLU column-ordering options](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html).
Only the test factory's permc_spec changes; original KKT/RHS hashes, numerical
pivoting, all five original solves/refinements, rank refusal, original row bars,
declared slack and unavailable-rate masks stay checked across EVERY ordering.
No production source, consumer overlay, native solver or installed kit changes.
initial_rate.py remains SHA0a4147d9f9906f3fd9ee3badbfd58d1ad3fdd25bdd6690bac6b8a7d75a50847f.

| Bounded 1358-coordinate KKT | COLAMD nnz/bytes | MMD_AT_PLUS_A | MMD_ATA | NATURAL |
|---|---|---|---|---|
| Independent head + grid |70350/855072|313684/3775080|71288/866328|75680/919032|
| Cross-coupled head + grid |73938/898128|335411/4035804|74364/903240|76317/926676|

Bytes are PUBLIC L/U CSC array bytes on these bounded fixtures, NOT native
factor workspace/peak. All other candidate orders are worse on both grids.
Therefore none is qualified as a blanket production replacement, nor as proof
that an unrecorded actual field-size graph fits its unchanged memory cap.
Do not blindly repeat the failed physical KKT or increase its cap from this.

TDD RED verify-20261031T035254Z.log:17 collected,2 failed/15 passed,2.85s.
The missing recorder admissions fail, NOT production numerics. GREEN
verify-20261031T035318Z.log:17 passed in2.36s; all repo Ruff/check-format pass.
Private canonical python -B -m dev_lab.dev.verify -- exact touched test file,
normal one-worker Box, no cap/queue overrides; threads/Cargo1, bytecode off,
authoritative python/ plus existing exact native portion on PYTHONPATH.
New evidence remains in output/logs/26/10/31/kkt-ordering/<worker-pid>/,
including all four operator/RHS hashes, arrays/fill and original solve counts.

An iterative replacement was considered but NOT implemented: residual success
alone is not the existing all-row KKT's full-column-rank authority. A converged
rank-deficient least-squares answer must not silently acquire unique-rate flags.
The useful next physical diagnostic is an explicitly bounded sparse ORIGINAL
rate-system capture before KKT factor entry, with all coefficients/RHS/selected
columns/original bars preserved and an exact stop before integration. That
needs separate consumer preparation/admission, not another OOM trial.
