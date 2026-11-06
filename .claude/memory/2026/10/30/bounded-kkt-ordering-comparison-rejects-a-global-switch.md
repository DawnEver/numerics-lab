# Bounded KKT ordering comparison rejects a global switch

Based on `6e006b1`, this round adds only `tests/unit/linear/test_kkt_ordering_comparison.py` and this note. NO production file, global default, installed package, native extension, lock or solver algorithm changed. In particular initial_rate.py remains SHA256 `0a4147d9f9906f3fd9ee3badbfd58d1ad3fdd25bdd6690bac6b8a7d75a50847f`.

## Authority and test scope

The original all-row augmented KKT is symmetric but indefinite: `[I R; R.T 0]`. The public SuperLU `splu` API documents default COLAMD and the alternative MMD_AT_PLUS_A pattern ordering. The test forwards the ORIGINAL `_least_squares_rate_values`, its original five solves/refinements, and its original KKT factorization. Only `permc_spec='MMD_AT_PLUS_A'` is added at that factory call for the candidate. Numerical pivoting, all rows/coefficients/RHS, rank handling, row allowances and uniqueness logic are unchanged. No SymmetricMode, zero diagonal-pivot threshold, normal equations, pruning, iterative solver, or response surrogate is used.

All comparison systems have more than512 columns. The bounded grids use676 grid coordinates plus a two-coordinate head, with a connected variant and retained1e-30 coupling/row controls. KKT dimensions are at most1358 and nnz below20000. Matrix/RHS hashes agree between orderings; the factorization is checked not to mutate the canonical input. Known solutions and EVERY original equation pass the existing1e-10 relative row bars. The independent and connected grid answers agree with their declared solutions and with each other at absolute1e-10. Public retry controls reuse the existing 600-column overdetermined fixture: values agree at1e-12, unused-rate uniqueness masks agree exactly, contradictory inputs retain the same exception, and declared-slack controls retain the same admission. A rank-deficient KKT still refuses with the original error. Six existing real violated-row weakref lifetime controls also run unchanged.

Public L/U extraction occurs ONLY on these bounded test factors. It is not added to any application observer. Reported array bytes count the public CSC data/indices/indptr, NOT native factor workspace or per-factor peak. The whole-test Box record is the separate peak-memory authority.

## Findings: numerical checks pass, memory remedy NOT qualified

| Fixture | KKT size | COLAMD L+U nnz | MMD L+U nnz | COLAMD public bytes | MMD public bytes |
| --- | --- | --- | --- | --- | --- |
| Grid, independent head | 1358 | 70350 | 313684 | 855072 | 3775080 |
| Grid, cross-coupled head | 1358 | 73938 | 335411 | 898128 | 4035804 |
| Public dependent-row retry | 1201 | 3611 | 3014 | 52948 | 45784 |

The six public retry variants have identical fill counts within each ordering. The candidate helps the simple dependent-row fixture but is substantially WORSE on both grid patterns. Thus it is NOT qualified as a blanket replacement or as a remedy for an unrecorded application graph. No fixture was retuned to hide that result, and no production ordering change is proposed from these findings. Exact redundant-row contraction or another solver would require a separately reviewed algebraic/certificate design.

## Receipts

- RED `.verify/verify-20261030T214603Z.log`:9 failures/6 passes in3.15s. This is a missing test-only comparison recorder, NOT a production numerical failure; initial lint/format findings also failed independently.
- `.verify/verify-20261030T214727Z.log`:INCONCLUSIVE collection error, because the private locked environment has no psutil. No install or dependency change was made; the optional process snapshot idea was removed. Public CSC bytes plus the normal Box record supply bounded memory evidence instead.
- GREEN `.verify/verify-20261030T214749Z.log`:15 selected/passed in2.39s, exit0; Ruff check/format pass. Exact canonical door: private python `-m dev_lab.dev.verify -- tests/unit/linear/test_kkt_ordering_comparison.py`. One file selects ONE worker/CPU1 through the normal door, without a manual cap/queue override. All five numerical thread variables and CARGO_BUILD_JOBS were explicitly1, bytecode off, authoritative python/ source first. Environment identity `c57af406fd9f90ff`.
- Matching `.outcome.json`:normal exit0, Box peak157962240B, CPU2.609375s, wall3.6818528s. This is the whole bounded test process, not a per-order peak.
- Nine exclusive evidence JSONs: `output/logs/26/10/30/kkt-ordering/28436/`. They preserve canonical KKT/RHS digests, source hashes, NumPy/SciPy versions, factor options/shapes/fill/array bytes and original solve counts. No full matrix or trajectory is serialized.

No application factorization, mesh, field initialization, physical replay, cap increase, consumer overlay update or remote publication occurred. A tests-only new engine HEAD does not silently update an existing exact-head consumer gate; any consumer admission remains separate.
