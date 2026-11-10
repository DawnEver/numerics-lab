---
name: fixed-point-strategies-hold-the-newton-field-on-deep-saturation
description: coefficient-damped holds its coefficient lag to the increment bar; anderson is safeguarded and judged on g(x)-x; modified-newton searches every step. Measured on a saturating slab before/after; picard-damped/continuation needed no change.
created: 2026-11-07
accessed: 2026-11-07
tags: [nonlinear, convergence-rule, anderson, modified-newton, coefficient-damped, port-golden]
metadata:
  type: project
---

# Fixed-point strategies hold the Newton field on deep saturation

Asked by motronics-studio (its handoff of 2026-11-06, ten xfail rows in
`test_fea_iteration_method_matches_the_nu_damped_field`). Reproduced here without motronics on a
1-D saturating slab (`tests/unit/test_a_saturating_slab_is_solved_to_the_newton_field.py`: iron
`nu` from `1/mu_r` to vacuum, an air gap, Newton at a residual stop of 1e-8 as reference; bar
1e-4 = the production `rtol_energy_pct` 0.01 %). Branch `pr/fixed-point-stop-and-globalisation`.

## Ask 1 -- the stop rule: premise half right

`_has_converged` ALREADY ANDs the relative increment into the energy stop for every fixed-point
strategy, so "stops on energy alone" is wrong. The real hole is specific to `coefficient-damped`:
it solves with the MIXED reluctivity, so under a small alpha the iterate creeps (alpha 0.1,
increment 2.6e-5) while the coefficients are still far from the ones it implies. Fix: the relative
lag `||c_fresh - c_mixed|| / ||c_mixed||` (stiffness and source) is held to the increment bar,
passed as state the iterate does not carry beside `material_change`.

| slab | before | after |
|---|---|---|
| j=20, n=120 | converged it 9, field 9.2e-4 off | converged it 16, 1.0e-5 off |
| j=6, n=40 | converged it 16, 1.09e-4 off | converged it 22, 3.1e-5 off |

## Ask 2 -- picard-damped / continuation: no change

Not reproducible: on the slab both converge inside the bar in 42-72 iterations. Downstream they
sit at field 2.5e-3..3.1e-3 at the 100 cap, i.e. 25-30x ABOVE the 1e-4 stop bar, so
"not converged" is the honest answer there and the motronics 5e-3 field bar is the looser one.
Flipping them is a cap/rate question for that case, not a declaration bug.

## Ask 3 -- divergence: confirmed and fixed

`anderson`: field error at the cap before 1.4 / 0.82 / 0.55 at j = 0.2 / 1 / 3, the last two
DECLARED CONVERGED (the test read the mixed step, not `g(x)-x`). After: judged on `g(x_k) - x_k`,
and a step that raises `||g(x)-x|| / ||x||` above the bar is rejected (mix -> history restart,
substitution -> beta halved, floor 0.1). Every j converges, field 1e-9..4.5e-5, 35-45 iterations
(picard-damped: 42-68). The equation residual `||r||` was tried as the merit first and failed:
from a zero seed it is smallest at the seed.

`modified-newton`: before 43 / 65 / 16 at j = 0.2 / 1 / 3 (the refreshed step was trusted
unsearched; from zero it lands ~180x past the root). After: Armijo backtracking on the raw
residual for every step, a held step needing any search refreshes; 10-15 iterations, field
<= 2e-6 (Newton: 9-11). New options `max_backtracks`, `shrink`, `armijo` (Newton's defaults).

Out of scope, seen on the same slab: `quasi-newton` and `bfgs` also diverge there.

## Port

Rust mirrors all three. The golden gained a `saturating` section (a diagonal `nu(x) x = b`), the
only fixture that reaches the safeguard, the search and the lag. Anderson there is held by
convergence on the Rust side, not by trajectory: SVD vs QR lstsq differ ~1e-12 on that history
and the safeguard amplifies it; five seeds tried, none inside the 1e-12 port bar.
