---
name: row-acceptance-iterates-to-round-off
description: Per-row step acceptance now iterates each row to its own round-off floor (RowConvergence); row_tol loosens it, gmin composes with it; the global-norm path keeps real callers.
created: 2026-11-06
accessed: 2026-11-06
tags: [time-integrators, newton, row-acceptance, round-off, gmin]
metadata:
  type: project
---

# Per-row acceptance iterates to round-off

## Defect (measured by motronics)

`integrate_transient(row_acceptance=True)` accepted a step as soon as every row was within the FIXED
`ADMISSION_BAR` (1e-10) of its own scale. On a well-scaled circuit that is looser than the arithmetic:
the old global-norm march let Newton's quadratic last step land at round-off, and a varistor's recovered
exponent moved from rel 2e-15 to 2.9e-11 (pin 1e-12). `tol_rel` and `gmin_stages` were refused beside it.

## Rule now (`python/numerics_lab/time_integrators/_step.py`, `RowConvergence`)

- `RowAcceptance(problem, form, tol)` reads, per row, `bars = max(tol * scale, construction floor)` (noise
  rows at the system resolution, as before) and `roundoff`: the same rows at `FLOOR_SAFETY eps * scale`.
- `row_tol=None` (default): an iterate within `ADMISSION_BAR` is accepted once every row is at `roundoff`,
  OR Newton stagnated -- the last correction did not halve `max |r_i| / roundoff_i` (`_STAGNATION_RATIO`),
  or the correction was at the state's round-off. Quadratic convergence finishes; a plateau stops.
- `row_tol=x`: LOOSENS the bar for cost; an iterate within it is accepted at once.
  `row_tol=ADMISSION_BAR` is exactly the old behaviour (the planted control in the tests).
- At the `max_newton` ceiling the bar alone decides (no further iteration to wait for).
- `gmin_stages` composes: the continuation runs a fresh `RowConvergence` (solved iterates only).
- `tol_rel` beside `row_acceptance` still refuses, naming `row_tol`; `row_tol` without `row_acceptance` refuses.

Measured (`tests/unit/test_row_acceptance_iterates_to_round_off.py`): `exp(u) = 5` lands at rel 0.0 by
default vs 1.85e-12 at `row_tol=ADMISSION_BAR`; a microweber row beside a 4e8-scale volt row is driven under
`64 eps 1e-6`; a residual plateauing at 1e-12 (above round-off) stops in <= 5 iterations (14 with the
stagnation ratio disabled); `row_tol=1e-4` takes fewer iterations; gmin rescues a 3-iteration plain fail.
`test_the_row_screen_rereads_a_moved_rate.py` now pins `row_tol=ADMISSION_BAR`: to round-off its persistent
1e-18 miss takes one more iteration (3, not 2) to show stagnation; the screen is what it tests.

## Does the global-norm path still have a caller? Yes -- not deleted

`row_acceptance=False` is the DEFAULT of `integrate_transient`; every in-repo caller (`modal_transient`,
`multirate`, `periodic_march`) and every downstream march that does not opt in uses it, the problem's
`native_step` kernel is golden-equivalent only on that path, and `tol_rel` / `_at_roundoff_floor` (with a
problem's own `residual_roundoff_floor`) serve it. Retiring it needs those callers migrated and measured
(iteration cost and trajectories) first; no such measurement exists, so it stays.

## Cost to existing row-acceptance callers

Default behaviour CHANGED: a step that used to stop at 1e-10 now takes typically one or two more Newton
iterations to reach round-off or show stagnation. A consumer that wants the old cost passes
`row_tol=1e-10`; a consumer's `tol_rel` maps to `row_tol`.
