---
name: reuse-square-lu-for-original-all-row-rate-refinement
description: Bound all-row LSQR original-defect corrections by the existing real square factor before retiring it for an unchanged KKT fallback; every original row and uniqueness authority remains unchanged.
metadata:
  type: project
---

# Reuse square LU before augmented fill

A real square choice sometimes factors but ordinary original-row refinement
leaves redundant witnesses outside their physical bar. Instead of immediately
factoring a much larger augmented system, retain that same factor as a right
preconditioner for an all-row LSQR solve. Its transpose action uses the SAME
factor's transpose solve. Initial all-row solve plus at most three corrections
of ORIGINAL physical defects, each at most128iterations, no regularization or
numerical rank/condition cutoff. LSQR stop codes/norms NEVER accept a rate.

Return a candidate ONLY if every ORIGINAL row satisfies its existing finite
tolerance/allowance bound. Failure returns None. Caller then retires the square
factor BEFORE the unchanged KKT fallback. No square factor still uses the
original KKT path. Healthy square outputs, unavailable coordinate masks,
physical contradiction refusal and original allowance retain their authority.
No caller budgets, case-specific thresholds, discarded rows or zero snapping.

Application evidence motivating the change is external, not an engine fixture:
Motronics preserved180371x179742normalized full-motor operands. Original KKT
passes at~2.24GiB offline. One preconditioned all-row solve fails one witness
and remains recorded; two original-defect corrections then pass ALL unchanged
rows at~0.99GiB. This is NOT full initial-pair or trajectory qualification.
The committed public engine must still be replayed on those actual operands
and the full application measured under its declared cap, not inferred from
this prototype or its scalar residual norm.

Canonical private verify:8missing-function RED plus formatting refusal ->
33/0GREEN across new actual numerical tests, existing square-lifetime controls,
all four KKT-order characterizations and field-size singular/contradictory
systems. Real numerical factors/solves, no planted passing answer. Lifecycle
assertion now admits a successful one-factor retry; any actual KKT still must
see the old square weakref gone. Inconsistent rows without allowance MUST
still invoke unchanged KKT and refuse. Source/fixture tiny1e-300 coefficient
remains; inputs unchanged; healthy square never enters the new retry.
Whole-repo Ruff passes and262files format-check clean.
Receipts output/logs/26/10/31/square-preconditioned-all-rows-{red41,green41}.
Normal CPU1/1GiB/300s, private source/native PYTHONPATH, all numerical threads1.
No installed/shared runtime edit or application source vendoring.
