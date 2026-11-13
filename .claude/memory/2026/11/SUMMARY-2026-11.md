---
name: SUMMARY-2026-11
description: November dependency work remains unaccepted after the first EEC smoke; the trial limiter is partial
created: 2026-11-06
accessed: 2026-11-06
metadata:
  type: summary
---

# Summary 2026-11 (numerics-lab)

## Month

October's catalogue carries no open plans. Draft PR13 proposes scoped EEC
matrix reads; it is not numerical acceptance or physical-history support.
The first remote smoke at b45b46de9cbb5478eabba2efb8296243c4c95be9
failed on Linux and Windows. The Linux record proves a missing November
catalogue and an invalid new positive fixture: march explicitly forbids
on_accept with EEC. Preserve that guard and reassess the consumer contract.
The earlier dated source note's observer-order rationale is not evidence of
a reachable physical-history/EEC path. No consumer pin has changed.
Per-row step acceptance now iterates to round-off
(`06/row-acceptance-iterates-to-round-off.md`; not a plan entry).
The November entries also describe the problem-owned trial limiter, its
retained-increment arithmetic correction and publication against the actual
integration branch -- a dated checkpoint, not a live acceptance verdict.

## Plans

### accepted-eec-linearization
- status: open
- evidence: Draft PR13; remote smoke 42213458383; source proposal in 06/accepted-eec-linearization-is-a-scoped-real-matrix-read.md. Correct the invalid fixture without claiming unsupported physical-history/EEC integration, then obtain exact-head remote acceptance.

### problem-owned-newton-trial-limits

- status: partial
- evidence: 05/newton-trial-limits-do-not-modify-the-law.md; 05/a-discarded-newton-step-is-not-roundoff-evidence.md; 05/the-trial-limiter-candidate-reaches-its-real-upstream.md; https://github.com/DawnEver/numerics-lab/pull/9

Implementation and independent source review are complete for the published
candidate. Postmerge full CI and formal review remain required; the consumer's
physical-law integration is a separate objective, not a numerical claim here.
