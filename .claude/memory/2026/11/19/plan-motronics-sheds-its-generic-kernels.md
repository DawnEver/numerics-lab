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
5. OPEN DECISION (user's): motronics' RUST also calls these kernels -- `laplace_net` (`control/affine_scan.rs`,
   `first_crossing.rs`, `thermal_step.rs`: `expm`; `signal_pass.rs`: `dense::{solve, cond}`) and
   `pybind/euclid/src/signal_read_mod.rs`. A Python door cannot serve a Rust caller. Options:
   (a) a cargo git dependency on numerics-lab's `numerics` crate pinned to the relocked sha, admitted ONLY in one
   motronics door crate (e.g. a `gauss` crate re-exporting it) and in `crates/contracts/tests/crate_dag.rs`;
   (b) move the march kernels that need `expm` into numerics too. (a) is the smaller step and keeps the one-door
   rule at the crate level; (b) is where "all numerics to Rust in the engine" ends. Until decided, step 4 deletes
   only `fill` and the condensation; `expm`/`dense` stay in `core_num`, a declared second copy.
