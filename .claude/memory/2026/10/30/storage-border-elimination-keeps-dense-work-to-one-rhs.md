---
name: storage-border-elimination-keeps-dense-work-to-one-rhs
description: "Reuse the original bulk LU with vector RHS solves and sparse correction/witness products, preserving exact rank authority and tiny coefficients."
metadata:
  type: project
---

# Storage-border elimination uses bounded dense workspace

Feature branch pr/storage-border-memory, based on origin/integrate/main a69f284.
This patch is generic solver code and synthetic tests only. No downstream assets,
case settings, installed runtime or physical workload changes; no push.

storage_border_rows previously materialized the complete transposed bulk-border
coupling, its dense solved correction, and a dense null-witness product. The
replacement factors the SAME bulk matrix once, solves each original border RHS
as a vector with trans='T', and retains every nonzero solved coefficient in CSR.
CSR stacking and sparse witness multiplication replace both wide dense products.
There is no threshold, rank tolerance, regularization, fallback or dropped small
coefficient. Nonfinite solved operands and singular bulk factors still refuse.

The exact binary-float border dependency routine, independent row order, null
witness normalization, selector order, StorageRows type and CSR algebraic output
are unchanged. No premultiplication changes the original RHS before solving.
Sparse and dense accumulation need not be bit-identical in floating arithmetic;
small reference tests compare at unchanged tight bars, not relaxed tolerances.

Dense solve workspace is O(bulk), independent of whole-border width. Sparse
correction and output storage remain proportional to actual nonzeros and can be
large for dense inverse couplings; LU fill is unchanged. This does not establish
that any downstream initialization fits a particular resource cap or succeeds.

## Exact local environment, no shared installation

An older borrowed development environment was refused before tests because its
development-library revisions did not match this checkout's lock. No bypass or
lock change was used. A private .venv was created by the declared frozen sync:
uv sync --frozen --extra dev --no-install-project, using existing CPython3.13.
Validated installed development-library revisions match committed pins:
ac7db79a6e74d5d2309851422246aefe2b39a5e2 and
9ffbb21886e9bf84616615d4109386dfddec183e.

Bootstrap used normal CPU1/2GiB/600s/queue120. Initial deep-cache attempt exited1
on a Git filename-too-long error, not resource exhaustion; its evidence remains
in output/logs/26/10/30/storage-border-bootstrap/. Only cache/temp paths were
shortened, without global Git configuration changes or increased caps.
output/26/10/30/b2/: exit0, peak216416256B, wall26.6947436s, sources equal.

Although selected tests exercise sparse storage algebra, linear package import
requires the native extension eagerly. The first verify attempt therefore had
four collection errors and was INCONCLUSIVE, not RED. The declared canonical
native build was then made into the private output/26/10/30/n1/k portion, never
a shared environment. Existing pinned Rust1.99.1 was used without toolchain
download; Cargo/cache/temp/target writes stayed in owned output. Native build
exit0, peak1146167296B, wall149.7292863s, tracked source hashes unchanged.

Tests use .venv/Scripts/python.exe, python/ first on PYTHONPATH, and that private
kernel portion second through the package's existing extended __path__. A test
asserts storage_border.__file__ resolves to this checkout's authoritative source.
All five OMP/OPENBLAS/MKL/BLIS/NUMEXPR thread variables are explicitly1, as are
CARGO_BUILD_JOBS and PYTHONDONTWRITEBYTECODE. No fake kernel was provided.

## RED and GREEN evidence

- An attempted -n0 invocation ran1 failing/99 passing tests but the door refused
  its unsupported census option: INCONCLUSIVE, not accepted RED. Subsequent
  measurements use only explicit file paths and the ordinary door's4 workers.
- RED .verify/verify-20261030T171601Z.log:100 tests,1 failed/99 passed,4.36s;
  result=fail, exit1. The new allocation guard catches the original20000-by-97
  conversion before allocation. An independent hooks-installed check also
  failed; standard hooks were then installed only in this repo's private .git,
  with its principles page unchanged. No hook waiver was used.
- GREEN .verify/verify-20261030T171757Z.log:122 passed,7.19s; result=pass, exit0.
  All repository Ruff checks and formatting checks passed through the door.
  Box peak1446830080B, wall8.8663705s. No skipped or deselected tests reported.
- Explicit source-before/source-after seals and environment are in
  output/logs/26/10/30/storage-border-green/171754986-35872/: equal, exit0.
  The door's displayed tree key repeats across RED/GREEN because its address
  list names src rather than python; it is not the sole Python source proof.
  Tested storage_border.py SHA256:
  782fea2c32215986a206c32765c47d16d8ed4dd4a0ed6a1ad827f6f28deb28bb.

The eight explicit targets were:
- tests/unit/linear/test_storage_border_bounded_elimination.py
- tests/unit/linear/test_sparse_initial_pair.py
- tests/unit/linear/test_storage_dependency_support.py
- tests/unit/nonlinear/test_actual_rate_finite_storage_initial_pair.py
- tests/unit/nonlinear/test_boundary_current_range_keeps_original_state_correction.py
- tests/unit/nonlinear/test_small_winding_row_keeps_original_particular.py
- tests/unit/test_finite_storage_right_newton.py
- tests/unit/test_initial_circulating_current_keeps_exact_coordinates.py

New synthetic cases cover a100003-dimensional logical sparse operator guarded
against multi-column dense conversion/RHS, one real LU and97 real vector solves,
zero bulk/border, symmetric and nonsymmetric dense-reference equivalence,
positive/negative coefficients down to1e-300, and nonfinite/singular refusals.

## Downstream integration remains separately reviewed

Prefer an exact-module process-local source loader for
numerics_lab.linear.storage_border, installed before any consumer imports it.
Reject preloaded/stale bindings, seal source SHA and commit, verify canonical
module/function identities through consumer bindings, and retain all other
installed engine modules and native kernel. A package-wide source overlay would
also replace unrelated engine changes and is not equivalent. Such a loader needs
its own small tests before use. No downstream observation or overlay was launched.
