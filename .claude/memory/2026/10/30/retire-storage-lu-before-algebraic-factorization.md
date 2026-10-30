---
name: retire-storage-lu-before-algebraic-factorization
description: "Drop only the dead local storage LU after its final response solve; forwarding weak-reference tests preserve algebra, evaluator counts and determinacy guards."
metadata:
  type: project
---

# Retire the storage LU at its last use

Base930ecdc4ba18ad310d673d14626857c70b359f99, branch pr/storage-border-memory.
Approved scope is only the demonstrated local storage-factor lifetime, not the
algebraic factor, iteration ownership, global caches or downstream policy.

In _actual_finite_storage_state, storage_factor has its final use in the base
solve or optional response solve. A single del storage_factor follows response
construction before mapping and algebraic matrix construction. No callback uses
this local afterwards. The existing algebraic LU is still built BEFORE the
admission check; its invertibility/determinacy guard is not skipped for an
already-admitted state. Its subsequent lifetime is deliberately unchanged.

No numerical expression, solve/factor order or count, tolerance, rank authority,
charge/residual/admission criterion, failure class, fallback or regularization
changes. No global cache operation, forced GC or LU extraction is introduced.
Only this code line plus comment, a synthetic unit file and this memory change.
No downstream assets, runtime installation, private environment or caps change.
No physical workload, upstream push or permission/origin change was performed.

## TDD evidence

The tracking proxy forwards actual tiny SciPy LU and solve calls. Its spy keeps
only weak references to storage owners, and arrays/call-role metadata, never a
strong factor reference. The next algebraic factor entry requires the prior
storage proxy to be unreachable. Tests cover zero/nonzero active coupling and
both an already-admitted state and a state requiring backtracking. Expected
solutions and every solve value are compared exactly; evaluator, LU and solve
counts are explicit. A singular algebraic minor must still refuse even when
the current state already satisfies all charge/residual constraints.

- RED .verify/verify-20261030T180255Z.log:5 tests,4 failed/1 passed,3.00s;
  covered verify result=fail, exit1. All four lifetime cases fail at the retained
  storage owner. Formatting also failed independently and was corrected before
  GREEN. No claim that this first run was formatting-clean.
- GREEN .verify/verify-20261030T180447Z.log:127 passed,3.57s; result=pass, exit0.
  Ruff check and format --check pass across258 files. Box peak1440063488B,
  wall4.8682524s. The ordinary door selected4 workers; no selector override.
- Explicit source and environment seals at
  output/logs/26/10/30/storage-factor-green/180445754-29572/: before/after equal,
  exit0. The displayed verify tree key does not substitute for these Python
  source seals. Tests assert the active module is this checkout's python/ file.

The private .venv and existing private kernel portion were reused without an
install or rebuild. Source python/ precedes that kernel portion on PYTHONPATH.
All five OMP/OPENBLAS/MKL/BLIS/NUMEXPR thread variables, CARGO_BUILD_JOBS and
PYTHONDONTWRITEBYTECODE are1; caches/temp remain under existing owned output.

Nine explicit targets:
- tests/unit/nonlinear/test_storage_factor_lifetime.py
- tests/unit/linear/test_storage_border_bounded_elimination.py
- tests/unit/linear/test_sparse_initial_pair.py
- tests/unit/linear/test_storage_dependency_support.py
- tests/unit/nonlinear/test_actual_rate_finite_storage_initial_pair.py
- tests/unit/nonlinear/test_boundary_current_range_keeps_original_state_correction.py
- tests/unit/nonlinear/test_small_winding_row_keeps_original_particular.py
- tests/unit/test_finite_storage_right_newton.py
- tests/unit/test_initial_circulating_current_keeps_exact_coordinates.py

The previous122 regression cases remain green, including rank, tiny-coupling,
singular, charge and original-row admission controls. No tolerance was weakened.

Tested nonlinear/finite_storage.py SHA256:
b332f89a2e0f796c16f0a735181e25c7d3ccf9f31ba32c970087c112b482d17b.
Unchanged linear/storage_border.py SHA256:
782fea2c32215986a206c32765c47d16d8ed4dd4a0ed6a1ad827f6f28deb28bb.

## Scope and remaining proof

Weak-reference collection proves local owner retirement in the interpreter,
not exact allocator reclamation, retained native-cache bytes or a downstream
memory-budget guarantee. No application memory reduction was measured here.
Other array and algebraic-factor lifetimes remain outside this approved change.

The public released_on_exit ownership scope and DIRECT_SOLVES backend counters
were inspected read-only as a separate downstream proposal, not adopted here.
That scope removes newly analyzed cache patterns, not held-factor references;
ordinary cache-capacity eviction during its body is still possible. Backend
counter deltas are not cache occupancy/byte measurements. No live private cache
was inspected or cleared, and no downstream seed behavior was changed.

A downstream exact-module diagnostic still pinned to the old commit must refuse
this new head until its source binding is separately reviewed and updated. This
change owns nonlinear.finite_storage, not just linear.storage_border; a broad
package overlay or runtime-installed patch is not implied by these test results.
