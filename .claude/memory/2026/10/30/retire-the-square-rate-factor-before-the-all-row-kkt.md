# Retire the square rate factor before the all-row KKT

On `pr/storage-border-memory`, based on `96b0c27`, the large-system violated-row retry in `linear/initial_rate.py` retained its rejected square LU while `_least_squares_rate_values` constructed and factored the original all-row KKT. That LU's final numerical use was `_refined_rate_values`; after entering this large retry there is no later use. The fix is one `del factor` immediately before the existing retry call, plus its explanation. It is lexical lifetime management, not a cache clear or solver replacement.

The KKT `[I R; R.T 0]`, every original row, row allowance, rank refusal, refinement, exception, tolerance, and uniqueness logic are unchanged. Small-system retry semantics are untouched. There is no new resource qualification or increased numerical limit.

## Real numerical TDD

`tests/unit/linear/test_rate_factor_retires_before_all_row_retry.py` constructs a 601-row/600-column sparse system whose two leading nearly dependent rows REALLY factor. The rounded forcing loses a coordinate in the square solve, and the independent third original row REALLY violates its unchanged bar, triggering the existing all-row retry. No solve result, refined output, matching, or fallback is fabricated. All LU/refinement wrappers forward the original arguments/results once.

Six cases combine a genuinely unused rate column with consistent, contradictory, and explicitly admitted-slack inputs. A weakref-only real-LU proxy must become unreachable BEFORE the KKT factor entry, without forced collection. Instrumented and reference forwarding runs compare every solve RHS/result/options, refined output and call counts exactly; accepted values and uniqueness masks agree exactly, contradictions retain the same exception message, and all original rows face their original bars. Existing singular-square least-squares controls remain covered.

- RED `.verify/verify-20261030T202020Z.log`: six new numerical failures and two existing passes in 2.87s. The retired-LU ownership assertion fails on the actual violated-row path. Initial lint/format findings also failed independently; both were corrected before GREEN.
- Focused GREEN `.verify/verify-20261030T202100Z.log`:8 passed in 2.56s, exit0.
- Final GREEN `.verify/verify-20261030T202130Z.log`:188 collected/passed in 7.43s, exit0. Ruff check/format pass. Ordinary private canonical `python -m dev_lab.dev.verify --` with 18 explicit lightweight files; no part/full/heavy or handwritten pytest invocation. All five numerical thread variables and CARGO_BUILD_JOBS explicitly1; bytecode off. Environment identity `c57af406fd9f90ff`.
- Exact final source/test/config hashes before and after agree: `output/logs/26/10/30/rate-lu-lifetime/20261030T202128Z/{source-before.json,source-after.json,result.json,environment.json,targets.json}`. The source seal is separate from the measure door's tree digest.

The final selection is the prior 16-file sparse state-range/joint/particular/touching-body/slack regression plus the new lifetime file and `tests/unit/test_a_field_size_overdetermined_rate_is_solved_in_least_squares.py`. It includes both previous storage-factor lifetime and bounded storage elimination controls.

## Limits and next review

This removes demonstrated avoidable factor overlap; it does not bound the new KKT factor's own fill or prove any physical application fits a memory cap. No physical execution, install/shared-runtime mutation, consumer overlay change, resource increase, publication, or retry occurred. A consumer wishing to use this file must separately review its additional exact-module source binding; the previous four-module harness is not silently updated by this engine commit.
