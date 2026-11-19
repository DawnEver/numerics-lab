---
name: plan-motronics-sheds-its-generic-kernels
description: Generic kernels moved from motronics-studio into numerics (dense expm/solve/cond, symbolic fill, block condensation); the motronics-side switch is a plan that waits on a relock.
metadata:
  type: project
---
# Plan: motronics sheds its generic kernels (2026-11-19)

- User rulings: "ALL numerics code migrates to Rust; Python is only early fast-iteration and glue" (user,
  translated); numerics-lab is the family's ONE numerics engine and motronics reaches it only through its
  `motronics.gauss` door. User assigned this move 2026-11-19 ("you do it").
- Source: motronics-studio a74923ef38 (`.claude/worktrees/integ8`). Landed here on
  `lane/generic-kernels-from-motronics`, LOCAL only:

| kernel | was (motronics) | now (numerics) | Python |
|---|---|---|---|
| `scipy.linalg.expm` bit for bit to n=16, 1e-12 past | `core_num::expm` | `numerics::dense::expm` | `numerics_lab.kernels.expm` |
| `np.linalg.solve` / `cond` bit for bit | `core_num::dense` | `numerics::dense::lapack` | `dense_solve`, `dense_cond` |
| `nnz(L)` from symbolic AMD Cholesky | `core_num::fill` | `numerics::linear::direct::fill` | `cholesky_factor_len` |
| block static condensation, `block_matvec` | `hamilton_sdm` (tail of `lib.rs`) | `numerics::linear::condense` | `condense`, `CondensedCells`, `block_matvec` |

- Goldens carried byte-identical: `tests/fixtures/expm_scipy.json` (bit-for-bit, planted 1-ulp and 1e-10
  controls green after rustfmt); `fill` and condensation tests carried unchanged; new: `block_matvec` test with a
  planted short vector, and the Python seam test `tests/unit/test_the_moved_kernels_answer_through_the_public_api.py`.

## The motronics side -- NOT done; it waits on the relock

1. Land this lane (pr from it, forge PASS, merge to `integrate/main`).
2. Relock motronics to that numerics-lab sha (`pyproject.toml` git dep; `uv lock`), through the bootstrap door.
3. Python call sites switch to door calls (`motronics.gauss` re-exports from `numerics_lab.kernels`):
   - `src/motronics/hamilton/sdm/factor_budget.py` (`motronics_native.gauss.cholesky_factor_len`)
   - `src/motronics/hamilton/sdm/coupled_tangent.py` (`motronics_native.sdm.CondensedCells, condense`)
   - `src/motronics/hamilton/sdm/coupled_layout.py` (`motronics_native.sdm.block_matvec`; `CoupledCells` stays)
   - any Python caller of `motronics_native.affine_scan.expm` / `signal_read` cond (grep at the switch)
   - registries naming the old bindings move with them: `benchmarks/ffi_boundary_profile.py`,
     `native_parity.py`, `native_split_profile.py`, `port_verdicts.py`.
4. Deleted in motronics: `rust/crates/core_num/src/{expm,dense,fill}.rs`, `core_num/tests/expm_is_scipy.rs` and
   `tests/fixtures/expm_scipy.json`; `Condensed`/`CondensedCells`/`block_matvec`/`offsets_of` from
   `rust/crates/hamilton_sdm/src/lib.rs` (and its `the_condensed_step_is_the_dense_solve` test);
   `py_cholesky_factor_len` (`pybind/euclid/src/gauss_mod.rs`), `condense`/`block_matvec`/`PyCondensedCells`
   (`sdm_mod.rs`), the expm binding (`affine_scan_mod.rs`). `core_num` drops its `faer` dependency.
   The Python test that WRITES the expm fixture (`tests/unit/laplace/control/test_the_native_expm_is_scipys_bit_for_bit.py`)
   re-points at `numerics_lab.kernels.expm`; numerics' Rust fixture is then the copy it re-checks.
5. DECIDED (user, 2026-11-19): (a) first, then (b).

## (a) the door crate -- DONE locally in motronics, OWED RELOCK

- motronics `lane/numerics-door-crate/multiphysics-alignment` 5ccd02710f (off a74923ef38, worktree
  `.claude/worktrees/numerics-door`, unpushed): `rust/crates/gauss_door` is the ONLY crate naming `numerics`
  (`crate_dag.rs::only_the_door_crate_names_the_numerics_engine` + planted second holder). hamilton_fem, hamilton_pen,
  hamilton_sdm, laplace_net and pybind reach the engine through it; core_num's expm/dense/fill and hamilton_sdm's
  condensation are DELETED.
- OWED RELOCK: the cargo rev is PR #58's head 268c3b3f5e74, not its merge sha; uv.lock still resolves the older
  numerics-lab. Relock BOTH to #58's merge sha when it lands; until then the Python native build (motronics_native
  over the old numerics-lab wheel) and the sibling-address test are not expected to agree.

## (b) the march kernels -- numerics side DONE on `lane/march-kernels-from-motronics` (local, off 268c3b3)

| motronics file | decision | why |
|---|---|---|
| `laplace_net/control/affine_scan.rs` | GENERIC, moved whole -> `numerics::lti::guard_scan` | threshold arrivals of `G expm(M t) y0` + brentq: no domain name |
| `laplace_net/control/first_crossing.rs` | SPLIT: lattice, interpolants, `first_crossing`, `Dense`, `exponential`, `span_key`, `Propagators` -> `numerics::lti::first_crossing`; `augmented`, `DeviceGuard`, `ReadingRows`, `guard_rows`, `guard_band` STAY | the stayers build rows from a circuit's devices / topology / exosystem |
| `laplace_net/control/thermal_step.rs` | STAYS, calls `gauss_door::dense::expm` | a junction/package thermal network; its only numerics is the expm call |
| `laplace_net/control/signal_pass.rs` | STAYS, calls `gauss_door::dense::lapack::{solve, cond}` | the control signal pass (tasks, plant, signal laws); numerics only via those calls |
| `pybind/euclid/src/signal_read_mod.rs` (cond) | STAYS (FFI glue), through the door | a binding, not a kernel |

After the next relock (to the sha that carries `lti`), in motronics:
- `gauss_door/src/lib.rs` re-exports `lti` too.
- DELETE `laplace_net/src/control/affine_scan.rs` and the generic half of `first_crossing.rs` (no second copy, no
  Python fallback). Callers move to `gauss_door::lti::{guard_scan, first_crossing}`: `candidate_step.rs`,
  `control_event.rs`, `crossing_start.rs`, `crossing_trial.rs` (incl. its test), `graph_read.rs`, `loss_ledger.rs`,
  `storage_transition.rs`, the device half of `first_crossing.rs`, and pybind `affine_scan_mod.rs`,
  `signal_read_mod.rs`.
- Tests that move with the kernels (already carried into numerics): affine_scan's 6 inline tests, first_crossing's
  inline tests less `the_band_sums_each_guards_weights` (stays with `guard_band`). Python goldens stay in motronics
  and keep checking the native path through the door: `test_an_affine_segment_scan_is_its_golden.py`,
  `test_a_native_device_crossing_is_its_golden.py`, `test_a_native_candidate_segment_is_its_golden.py`,
  `test_a_native_control_event_is_its_golden.py`, `test_the_native_expm_is_scipys_bit_for_bit.py`,
  `test_a_wide_algebraic_loop_solves_natively_as_python.py`.
