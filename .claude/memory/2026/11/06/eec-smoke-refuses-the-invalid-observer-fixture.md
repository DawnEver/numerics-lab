# EEC smoke refuses the invalid observer fixture

Draft PR13 smoke 42213458383 at b45b46de9cbb5478eabba2efb8296243c4c95be9
failed on both base platforms. The Linux record identifies the missing monthly
catalogue and the new real-march positive fixture's forbidden on_accept/EEC
combination. These are real failures, not infrastructure failures.

Correction to the earlier source note: after_accept is an internal algorithm
stage, not proof that a physical observer can execute in an EEC trajectory.
march.py explicitly refuses accepted observers with EEC because corrections
are not a physical history. Preserve that existing guard. Native constitutive
owners requiring accepted observation cannot use this proposal to enable EEC.
Their full physical-history/correction authority remains open; no consumer
pin or numerical acceptance is implied.

The positive now uses a stateless cubic operator with a read-local clock,
exercising the actual EEC accumulation/correction and separate trial views.
It does not rely on an unavailable observer. A separate negative exercises
the actual observer/EEC admission refusal and requires no observer, trial or
matrix scope to have run. Existing numerical equivalence assertions remain.
These corrected controls are UNRUN locally; Ruff and format checks only passed.
The November catalogue was added separately at ef31b0b.
