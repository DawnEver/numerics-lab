# Sparse component state range and original joint correction

## Scope and implementation

On `pr/storage-border-memory`, based on `78feb895`, large total boundary width no longer silently suppresses state-range discovery. `linear/index_two.py` can issue a global CSR basis from bounded actual support components. The existing dense return remains the default for direct callers; the large finite-storage path requests CSR. Boolean bipartite connectivity replaces an int8 Gram pattern, avoiding overflow when 256 rows share columns. Every nonzero coefficient, including tiny basis entries, remains represented; there is no numerical pruning.

The joint solver consumes that issued range through the original augmented equations. Its large path uses sparse response/error columns and rank work on the ACTUAL Schur plus error-support components, not the discovery partition. Cross-component mixed columns can merge those components. Local rank qualification remains 512 columns and at most 512 squared dense entries; it is not a raised global rank limit. Sparse component complements also support the index-two rate-repair path.

Original physical particular reconstruction, homogeneous anchors, tiny winding repairs, full physical and augmented chart validation, row bars, carried slack, error certificates, and rate-uniqueness checks remain. No gmin, pseudoinverse, changed coefficient, tolerance, source, or initialization policy was introduced.

## TDD receipts

- RED: `.verify/verify-20261030T192225Z.log`: 5 failed, 2 passed in 3.56 s, verify FAIL. Failures included the >512 public pair, absent CSR issuance, int8 connectivity overflow, and absent bounded-component refusal API. Initial lint/format failures were also present and were corrected separately.
- Intermediate GREEN: `verify-20261030T192453Z.log`: 11 passed; `verify-20261030T192653Z.log`: 17 passed; `verify-20261030T193128Z.log`: 53 passed in 7.19 s.
- Final GREEN: `.verify/verify-20261030T193241Z.log`: 180 collected/passed in 8.17 s, exit 0; Ruff checks and formatting passed. Ordinary private `python -m dev_lab.dev.verify --` door, 16 explicit lightweight test files, no part/full/heavy selection. Door-selected four workers; this was not a handwritten pytest invocation. Environment identity `c57af406fd9f90ff`; all five numerical thread variables and CARGO_BUILD_JOBS were explicitly 1.
- Explicit Python/test/config hashes before and after that final measure agree: `output/logs/26/10/30/component-range-green/20261030T193239Z/{source-before.json,source-after.json,result.json,environment.json}`. The door tree digest alone is not substituted for this source evidence.

Fixtures cover 513 independent index-two blocks through the real public finite-storage pair API, nonzero cross-component mixed coupling, original F/Q/G residuals and uniqueness, 65 copies of the original tiny-row particular cases, and 260 touching-body pairs (520 body candidates). Allocation guards reject global wide dense conversion/identity/zero arrays. Tiny nonzero basis entries survive; a genuinely irreducible 513-column mixed joint chain refuses rather than being split by discovery components.

Final explicit targets: `tests/unit/nonlinear/{test_storage_factor_lifetime,test_actual_rate_finite_storage_initial_pair,test_boundary_current_range_keeps_original_state_correction,test_small_winding_row_keeps_original_particular,test_large_component_state_range}.py`; `tests/unit/linear/{test_storage_border_bounded_elimination,test_sparse_initial_pair,test_storage_dependency_support,test_joint_coordinate_sparse_closure,test_a_dependent_border_row_keeps_its_own_bar,test_a_driven_rows_rounding_is_carried_through_elimination,test_a_summed_row_inherits_its_rows_evaluation_floor}.py`; `tests/unit/{test_finite_storage_right_newton,test_initial_circulating_current_keeps_exact_coordinates,test_two_touching_bodies_free_their_common_voltage,test_an_index_two_solid_voltage_is_state_authority}.py`.

## Remaining limits: not a physical closure or fit claim

An actual irreducible discovery, rate-complement, or joint Schur/error-support component beyond the unchanged local qualification still raises NotImplementedError. Large dense adjoint sharing also refuses when its actual work is unqualified; no original row is skipped to continue.

Sparse response and response-error columns are assembled with hstack. If individual field responses are globally dense, their CSR nonzeros can still approach field times border, and assembling the column list plus the final matrix can temporarily retain a copy. Sparse storage is not a bound on that density or a panel-memory algorithm. The independent-block fixtures do NOT demonstrate fit for a globally coupled physical boundary, nor close its initialization. General certified sparse elimination or bounded panels remain separate work if required.

No physical execution, install mutation, shared-runtime change, overlay update, cap increase, or publication occurred in this round. No initial pair or trajectory of any physical application is certified by these synthetic tests.
