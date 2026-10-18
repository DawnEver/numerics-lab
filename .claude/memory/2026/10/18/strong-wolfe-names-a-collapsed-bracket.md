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
