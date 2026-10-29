---
name: strong-wolfe-names-a-collapsed-bracket
description: strong_wolfe's zoom divided by zero in _cubic_minimizer once lo == hi (found by optimi-lab P9's lbfgsb); a bracket with no float strictly inside now raises WolfeSearchFailed naming the collapse.
metadata:
  type: project
created: 2026-10-18
accessed: 2026-10-18
---

# strong_wolfe names a collapsed bracket

- Defect: at a jump in `phi` (or any phi with no Wolfe point in the bracket) the zoom bisects until
  the midpoint equals `lo`; `f >= f_lo` then sets `hi = lo`, and `_cubic_minimizer` divides by
  `a - b == 0` -> `ZeroDivisionError`, an exception no caller could name.
- Fix (`python/numerics_lab/line_search.py`, zoom): if no float lies strictly inside the bracket,
  raise `WolfeSearchFailed('the zoom bracket collapsed ...')`. The search's contract is unchanged:
  it returns a step meeting both conditions or raises `WolfeSearchFailed`.
- Test: `test_strong_wolfe_reports_a_collapsed_bracket_as_a_failed_search` (jump at a = 0.5,
  500-evaluation budget so the collapse, not the budget, ends the search); seen red as the
  ZeroDivisionError first.
- Consumer: optimi-lab `rebuild/p13` removes `lbfgsb`'s `ZeroDivisionError` catch and its Armijo
  fallback.

## The cap is an outcome, not a failure

- Removing optimi-lab's Armijo fallback showed it caught MORE than the collapse: on HS071 through
  the augmented Lagrangian, its inner L-BFGS-B hit 21 "reached max_step without bracketing" and 16
  budget failures. The first is not a failure at all for a box-capped search: More and Thuente's
  `dcsrch` exits at `stpmax` when sufficient decrease holds there and phi still falls.
- `WolfeStep.capped` (default False) marks that exit; `strong_wolfe` returns it instead of raising.
  Tests: `test_strong_wolfe_returns_a_capped_step_where_phi_still_falls_at_max_step`, planted
  control `test_strong_wolfe_still_raises_at_a_cap_where_sufficient_decrease_fails` (it zooms to an
  uncapped Wolfe point).
