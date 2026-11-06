---
name: SUMMARY-2026-11
description: November dependency work remains unaccepted after the first EEC smoke
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

## Plans

### accepted-eec-linearization
- status: open
- evidence: Draft PR13; remote smoke 42213458383; source proposal in 06/accepted-eec-linearization-is-a-scoped-real-matrix-read.md. Correct the invalid fixture without claiming unsupported physical-history/EEC integration, then obtain exact-head remote acceptance.
