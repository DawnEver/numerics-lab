---
name: retain-joint-images-not-field-wide-response-columns
description: Stream original joint response/error projections and recompute only responses multiplying nonzero homogeneous directions; retain original numerical authority and validators.
metadata:
  type: project
---

# Retain the images actually consumed by the Schur solve

Application native46 constructor returns but full response accumulation exceeds
four GiB. Actual517136x1927 coupling has636active columns. All17actual joint
operands were captured before LU (94347540B) and independently audited. Even
offline48 without CAD/field owners, the old original joint response assembly
does not return: at least576responses, peak4528787456B under the same4GiB job,
CPU560.40625s/wall577.5680823s. Terminal audit verifies8540source/input files.
Do not claim the precise terminal allocation or a completed initial pair.

Component mode now computes each SAME original response and its SAME original
block-residual forward-error correction, then retains only left@response,
abs(left)@abs(response) and abs(left)@error. The Schur and gamma formulas stay
the same. No whole field-wide response/error matrix or abs-copy is retained.
The sparse-column helper accepts an explicit projected image height.

After the real Schur rank solve, only response columns multiplying ACTUALLY
nonzero homogeneous directions are needed for the physical rate lift. Those
same deterministic responses are recomputed in original sorted column order.
Only exact zero products are omitted; no epsilon, coefficient truncation,
iterative-rank proxy, changed bar or fabricated uniqueness mask. Original
physical/chart validators and noncomponent dense path remain authoritative.

Canonical62items/4numericRED plus formatting ->62GREEN, seven explicit files;
whole Ruff/264format checks pass. Two genuine1200-row systems now forbid
retaining field-sized response/error stacks while still checking the exact
same original equations/solution/uniqueness. Projected zero/nonzero widths and
1e-300 preservation pass. A real five-row joint system with513chart columns
has a NONZERO homogeneous selected-rate lift: both dependent rates remain
unavailable, proving the lift is recomputed rather than discarded.

This is software proof only. Fresh offline qualification on the captured real
operands, then whole-motor unchanged-budget qualification, are still owed.
Projected images themselves may be large; never infer memory fit from naming.
