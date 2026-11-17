---
name: the-field-circuit-admission-and-clock-refusals-were-separate-mechanisms
description: "Motronics closure diagnostics separated Q provenance, null candidates, clocks and mixed-unit floors; actual Cummins border exposed forcing-dependent rank, repaired with 427 component regressions passing."
metadata:
  type: project
---

# Field-circuit closure evidence from the Motronics consumer

Measured on 2026-10-20 in the mounted component of `motronics-studio`, before any
claim about the real Cummins 250 A operating point.

- The old consumer handoff attributed its six-wire moving-solid failures to
  nonlinear MNA authority. The actual fixture uses a terminal branch, not MNA.
  The captured mechanisms were distinct: a linkage producer supplied no arithmetic
  certificate; finite-Q increments inherited no certificate from their actual Q
  producers; and normwise SVD lost four valid coupled body-voltage/current directions.
- A captured 15-coordinate boundary block had nine nonzero original projected rows.
  Six independent directions were reconstructed from those original rows and passed
  the existing product-roundoff certificate without changing its bound; the old
  candidate search returned only the two neutral-voltage directions. A two-by-three
  mechanism plant retains the required `1e-32` component.
- The consumer's actual moving six-wire voltage callback subsequently passed both
  zero/nonzero voltage, corner landing, rollback, and independent-full-stack checks.
  The installed-engine consumer run reported **57 collected / 0 failed**, including
  all 28 re-anchored-clock tests; this is not evidence of real Cummins 250 A tracking.
- Clock diagnostics showed two-ulp nominal drift, replacement of an owner's exact
  tick by a numerical target, a `1.084e-18 s` sliver, and a restart's physical-grid
  displacement. Anchor-plus-integer clocks removed the chained additions. A declared
  window end retains authority on either side of a coincident owner tick. JSON
  checkpoint tests also cover an interior event before its pending re-anchor.
- A later, independent flux-DAE oracle exposed mixed residual units: at an accepted
  stage, shaft speed and `1/h` made the scalar floor `3.6389e-6`, hiding an algebraic
  linkage defect `1.403e-8 Wb`. Replaying the captured stage with the unchanged
  componentwise arithmetic formula gives a linkage-row floor below `1.946e-15 Wb`
  and correctly refuses it. The legacy scalar diagnostic remains byte-identical;
  declared scalar/vector owner contracts are unchanged. Gmin now consults the same
  floor authority as ordinary Newton.

The official component verification used its own lock-consistent environment and
live Python/native sources: **81 passed**, all 225 Python files passed Ruff and
format checks. Recorded verification log:
`sha256:d92540324480fdf23a52a5147ed28d8f21d0f40bbc2b6ae3b182e2fa8948c113@18`,
environment `a65ae43eb9989114`. The pure flux-DAE consumer oracle still requires its
installed-engine rerun; the source changes alone are not that result.

The Motronics shared environment is refreshed only through its sanctioned locked
installer after explicit approval. Component Python files are now declared build
cache inputs, so a same-version rebuild must not silently reuse stale Python.

## Actual native Cummins border: rank cannot depend on forcing

The recorded-voltage replay then reached the real nonlinear Cummins initial pair,
not the six-wire mechanism fixture. Official consumer capture
`measure-40444.log` reproduced its refusal on 11,945 states. The computed border
returned 83 homogeneous directions: 80 were rate-only and satisfied every original
row, while three changed physical states and violated original equations by up to
395,074. The same matrix and entry-error bounds with zero forcing returned only
the 80 valid directions. Thus the refusal correctly caught false numerical
directions; it was not evidence that those physical states lacked source authority.

`border_rank.py` had selected pivots preferentially from rows whose *evolving RHS*
was zero, after already processing the original homogeneous anchors. Removing that
preference makes pivot discovery depend on the matrix and its existing entry
bounds. Homogeneous columns are now reconstructed and refined against the same
original selected subsystem as the particular, rather than noisy eliminated
coefficients followed by uncertainty-based zeroing. No tolerance, reliability
factor, physical-state filter, or nonlinear capability guard changed.

Two small reductions of the actual border gave a clean official RED (2/31), then
GREEN (31/31). The broader component verification passed **427/427**, including
all linear and nonlinear unit tests plus the clock, event, floor and build-cache
regressions; log
`sha256:32307ce992c194940bcab4381085b3a3456be2185e6af95368f2623bd6b4221b@22`,
environment `a65ae43eb9989114`. The actual captured border now returns only rate
directions, whose lifted columns satisfy all 35,835 original equations with exact
zero defect. Its particular also satisfies each original row's own unchanged bar.
Capture arrays and source/reference hashes are retained only as ignored local
diagnostics in the consumer's `scratch/actual-native-boundary`; they are not test
fixtures or committed evidence. After the sanctioned shared refresh, all 75
installed Python files matched the source bytes. The actual native three-interval
replay then passed (consumer `measure-44236.log`, one collected, zero failed,
21.29 s call), with its report retained under the consumer's
`output/logs/26/10/20/der-native-replay-three-intervals`. This closes the numerical
admission refusal on the real case, not physical equivalence: at 3 us the current
and torque residuals reach roughly 31 A and 0.279 Nm. The separate source-declared
250 A-command / 100 us generated-drive smoke subsequently passed (consumer
`measure-6592.log`, one collected, zero failed, 182.77 s call), with no development
override, 100 field steps, finite currents/torque and the existing two-star KCL bar.
Its final phase-current peak was 101.14 A and torque -17.076 Nm. Those are finite
energized responses, not proof of 250 A tracking, steady state, or vendor parity;
`generation_accepted` correctly remains false.

The consumer's subsequent circuit census exposed a separate explicit-table clock
defect: a declared PWL corner at nextafter(t0) was present in its exact schedule,
but the engine's lattice loop skipped it through same_instant. The component
sentinel test_explicit_step_table_lands_the_representable_instant_after_start
measured one genuine failure before production changes. Explicit step_times now
skip only targets at or before the accepted time; generated parent-lattice
roundoff handling is unchanged. The sanctioned verifier then passed 439 tests,
with all 225 Python files lint/format clean (receipt log
sha256:36e88e246916df88dc2df1680578643e121791d0709b515b840d7bea3481ccd2@23).
The already-authorized consumer installation helper refreshed the shared package;
all 75 packaged Python files match source, RECORD exists and the native module
imports. Installed time_integrators/march.py SHA256 is
b954f91c541ddf634791d79465a50d88592ca569523e29ad67c1bf4cacd10d5f.
The full consumer circuit regression is pending at this entry; no waveform,
physical acceptance bar or checkpoint compatibility alias was changed.

## 2026-10-21: a computed state range is not original exact-rank authority

The consumer's actual three-phase PI admission stalled with original winding-row
defects of 121--164 while every state correction was zero. Its original linear
system was solvable. A new independent seven-coordinate rank-one field/current
model passed, but explicitly adding the original algebraic rail/channel/source
coordinates reproduced the failure for both linear and cubic fields.

Formal test `tests/unit/nonlinear/test_boundary_current_range_keeps_original_state_correction.py`
gave two passes and two genuine convergence failures, verifier log
`147a08c9cd7e4a4df5cbac3b5d92034ab75876ece046b5d889878ae855b04d3e@1581`.
An earlier dense variant refused missing bounded-range authority; that was not
the causal failure and was replaced by the actual wired sparse model.

`linear/initial_storage_border.py::_sparse_joint_border` had proved exact zero
coordinates from its appended `delta_state - D*alpha = 0` lift as though the
computed floating-point D were original exact coefficients. A rational null
direction's rounding fabricated full rank and pinned all circulating currents.
Exact anchoring now reads original rows only when a supplied state range adds
lift rows. The lift still constrains the solve; original Q/F anchors and final
physical validation remain unchanged. No tolerances or error allowances changed.

The new four tests pass (`a12d3bd96ab36d0b10d03e3622a4e33aebbb4e163d1b438f1c01936b45b1985a@17`),
and 207 affected finite-storage, joint-column, sparse-border, allowance and
constitutive-domain regressions pass
(`2ab819de714f5cd03bf9d2536839b6ab53a4021b0e5d05b98fc9385d05e71d91@19`).
Approved project-locked consumer install 10521 completed; all 75 installed Python
files byte-match. `initial_storage_border.py` SHA256 is
`d1a796ecb7f149593a3ba58239ea37c9fe0184b52dd440cdc31ee4c99646438d`.
The original consumer PI rerun is pending, not claimed closed here.

## 2026-10-21: absolute-rate forcing and fixed homogeneous coordinates

The installed computed-range repair removed the huge original F defects, but
the same PI still failed differentiated rows: F was within its original bars
(maximum about 3e-11), while off-phase G rows were about 7e-10 against bars near
7e-20. This was a separate linear-solve mechanism, not a missing physical source
or permission to enlarge an arithmetic allowance. Consumer snapshot
`output/logs/26/10/21/bounded-replay-after-range-fix/failed-bounded-differentiated-sampled-pi.npz`
has SHA256 `cda443eedd94189b415d0342d13a392648745ffb6766c2901e39535fa11ef27b`.

Two repairs were required:

1. In the bounded state-range lane, when the actual sparse rate tangent T is
   entry-for-entry J, the absolute-rate forcing is mathematically `-Y*Ft`.
   Computing it as `(Y*T)*v - Y*(J*v + Ft)` introduced a driven RHS into an
   exactly homogeneous equation. The captured PI had T == J, Ft == 0, but that
   old subtraction supplied -1.45495612e-15. The new exact-identity lane forms
   `-Y*Ft` directly and transports the actual Ft bound through the actual Y
   multiply. Missing/unknown source provenance stays unknown; the old
   cancellation bound is not reused. T != J and mixed state columns retain
   their existing path. Original F/Q/G validators are unchanged.
2. That forcing repair alone left two genuine failures among the expanded 16
   wired/bare, linear/cubic, gate-mode and nonzero-storage controls. Actual-call
   capture proved that `border_affine_space` returned exact zero rail rates,
   but weighted refinement moved coordinates already proved zero by original
   homogeneous rows to about -1.49e-8 and -1.12e-8. Resetting those coordinates
   only after reconstruction broke neighboring G rows by up to 1.599e-5.
   `_sparse_joint_border` now removes those fixed-zero remaining columns before
   Schur formation and weighted refinement. This reuses the original exact
   support witness; no rank tolerance, RHS clipping or new source probe exists.

The two captured component failures are retained in the consumer under
`output/logs/26/10/21/boundary-current-range-0a3d36d5/`, as
`failed-boundary-current-range-13.npz` and `-15.npz`, SHA256 respectively
`79b2f533dcb92906c735c3fa452c91c046e874a71cf31d2350327baef8cf9c28` and
`686503203cec3ce281e8be3fd852ac1d979f8fdda4a5324f617c4e8ad3800be6`.
The formal diagnostic run was 14 pass / 2 fail,
`0a3d36d5f0742175b10d974ffef0b580520874473a61902dbe9cf1976901b7cc@1665`.

The first expanded verification then passed 271/272 and exposed a genuine empty
Schur-equation regression in
`test_only_certified_storage_increment_roundoff_can_meet_an_exact_anchor[2e-23]`:
a free rate remained, but no equation constrained it. This now uses the exact
zero representative and full free-rate chart, not an artificial border. Both
zero-column and zero-equation branches independently replay every original
supplied row against its own unchanged bar plus original allowance; insufficient
stored allowances and driven conflicts still refuse. Unknown physical-state
directions still refuse.

Final sanctioned verification: **272/272 passed**, including all 16 new current
range controls and the exact-anchor allowance negatives; log
`9e5e8256541b3be517d7652346ff2ff17bf896b36aee4482c62a20728da4460d@20`.
The approved shared installation and original PI/moving-network consumer reruns
are pending at this entry. Component source success alone does not close them.

The subsequent approved install 57479 completed with all 75 installed Python
files byte-identical. Consumer gate `measure-53932.log` ran 22 nodes: 19 passed,
including the original actual three-phase PI accepted-state/replay test; three
prepared moving-network cases (smooth/pulse/sampled) still refused original
boundary convergence. Thus the PI's mechanism is verified closed, while those
separate moving admission failures remain open, pending their actual captures.

## 2026-10-21: a computed homogeneous range must not replace a numerical particular

The prepared moving-network smooth/pulse/sampled failures were original F, not
G: the smooth case's last winding row had -1.23192505e-17 against its unchanged
own bar about 3.018e-23. Actual line-level capture of `_sparse_joint_border`
showed that the physical solve before range reconstruction met that row exactly,
with delta current +6.15977458e-18. Replacing all state coordinates by computed
`D*alpha` changed that correction to +1.49303909e-22 and introduced the rejected
row. Original Q was exact before and after. This was not a missing arithmetic
allowance or an intrinsically inconsistent physical initial condition.

The actual consumer mechanism regression first saved its operands and then
asserted the invariant using the same original matrix/RHS, each row's existing
bar and original allowance, excluding computed lift rows. It measured genuine
RED `measure-27348.log`, newly rejected physical row 56. The retained artifact
is under consumer `output/logs/26/10/21/network-state-reconstruction-mechanism-red/`.
An earlier independent 13-coordinate reduction (four stiffness/voltage variants)
did NOT reproduce the failure; those four are positive controls, not RED proof.

The numerical particular now retains the original solved physical coordinates.
Structurally absent D rows remain fixed, and every state-range return separately
replays original F/Q/G rows against unchanged own bars plus original allowances,
not Schur-inherited slack or computed-D construction error. Homogeneous direction
reconstruction and the refusal of nonzero free physical-state directions remain
separate; particular smallness is not uniqueness authority.

The first expanded run passed 293/294. Its only failure was a positive test
requiring the computed D's last bit: original Q fixes x=1 and the actual source
row is `5*y=fl(5/3)`, while the rounded range entry is `fl(1/3)`. The actual
original solve differed from that entry by one ulp. That oracle now independently
checks exact Fraction original-row defects and range product/subtraction
discrepancy against counted binary64 arithmetic bounds. No relative tolerance
was increased, and driven fixed-coordinate / unknown-authority negatives remain.

Final sanctioned component verification: **294/294 passed**, log
`505cfe9b7a6083989cf9099d7b362b3efe1fbe568caf54570fd0546b4ed30661@21`,
source tree `cd9a394d208af0ed`. Approved install and original network/PI consumer
reruns are pending; their closure is not inferred from this component result.

Approved install 47102 subsequently completed with all 75 packaged Python files
byte-identical; installed `initial_storage_border.py` SHA256 is
`adbc0ccc4edcdd2d0a2a8101173aacbd14c3ccc7e7096599d0f4400a0f118f43`.
Consumer `measure-41692.log` then passed **23/23** (50.04 s): the complete moving
network file, including all three formerly failing prepared smooth/pulse/sampled
cases, the actual before/after original-row mechanism regression, and the actual
three-phase PI accepted-state/replay file. This verifies those numerical failures
closed on the installed engine; broader coupling recensus is still separate.

## 2026-10-21: completion retains the actual partial-time arithmetic certificate

The consumer's moving native-storage initial condition exposed a separate
completion handoff defect. Its reduced 58-coordinate pair already proved every
field/current rate; only two terminal-voltage rates remained unknown. The actual
coordinate partial `Ft + J*velocity + M*time_partial` canceled to about 1e-21,
but completion received only that residual, dropping its already-counted
performed-arithmetic certificate. An original differentiated row then had defect
1.3115987116058196e-21 against its unchanged bar 2.6231974232116395e-31.
The capture is retained in the consumer's
`output/logs/26/10/21/moving-storage-completion/rate-completion-000.{npz,json}`.
This was not evidence that the declared storage-curvature law was wrong.

`linear/initial_sparse.py::sparse_complete_initial_rate` now accepts the actual
optional `partial_time_bound` capability. Existing arithmetic owners transport
it through precisely the two differentiated RHS expressions consuming Ft,
including their actual row normalization. `initial_rate_border.py` receives
that aligned equation certificate for border formation and the final original
row check. No allowance is added to original F, independently differentiated
curvature, or preserved-rate anchors. Unknown arithmetic support stays unknown;
independent law certificates retain their existing semantics. No certificate
means the original strict behavior, not an inferred zero-error proof. Nominal
F/Q/curvature expressions, tolerance, anchors and uniqueness rules are unchanged.

The root consumer now supplies its existing `partial_time_evaluation_at` result
once, rather than discarding the bound and reevaluating a plain partial. The
evaluation aligns residual and bound shapes, selects identical reduced rows,
and the component rechecks the actual pair dimension. The only production
`complete_initial_rate` caller was migrated; other direct component callers
retain their no-certificate behavior.

The seven new controls in
`tests/unit/linear/test_completion_transports_actual_time_partial_bound.py`
first gave one pass (missing certificate refuses) and six genuine missing-keyword
failures, log
`a3d1910d631e5e4bcf72fbfc12062f722198c5ed9ee003698036ba46c08554c7@234`.
All seven then passed, including wrong-row/unknown/dimension refusal and
independent F/curvature contradictions, log
`f68fec5c5ff57d5fbaf108802439319fb61c57c0247ff470f6dc28976cd72fca@17`.
Expanded linear/nonlinear verification passed **294/294**, log
`e22bc2b901b7b840d933bc75fe40c524a757e6c57e8bfb4e8f7fc10741de899b@21`.
Four exploratory tiny-row controls also passed, but did not reproduce their
proposed separate failure and justify no production change for that mechanism.

Approved install 40684 completed with **75/75** installed Python files
byte-identical. Installed SHA256 values:

- `initial_sparse.py`: `71b8f2dba6bb481e06a2bc69c55bd555c02917bdf61d85785706f564eaee8915`.
- `initial_rate_border.py`: `e92544a95a8bf836c42fddaff6b4750d518fb228432423f7e88bd9b084d47247`.

Consumer gate 29398 is running the original moving-storage node, its diagnostic
and affected physical regressions at this entry. **Their success is pending**;
component verification and installation alone do not close the consumer defect.

2026-10-21 continuation: actual source-plane admission is distinct from Newton
projection. Tiny held source changes below solve tolerance must still satisfy
their actual counted affine-row arithmetic. Formal causal RED
`a6fd0dad6dccef9cef0629ded18a5a5f5376476b663d61e0dfa78d3e77b20fbe@59`
was 2 FAIL / 1 PASS; the steep initial linearization remains unprojected.
The new separate typed capability now guards ordinary, floor and gmin acceptance.
Initial focused3 and expanded95 PASS; approved install28952 had75/75 Python parity.
Root actual MNA owner/wrapper tests passed measure-51852.log. Original SynRM
measure-5600.log reached exact zero torque error and LUT tracking
1.50607326686182e-16, improved from recorded4.7305154962289284e-14 and
5.421863760702552e-15 respectively; remaining failures were stale-record/zero
reporting boundaries, not worsened physical bars.

Review found the native speculative fast return bypassed this owner. Formal
2 RED `e191984959d305de85f2f31ddb3acffef12da48c4cbe1f122c03cb92a7990270@84`
proved off-plane0 accepted instead5e-11 and valid native candidate never checked.
Minimal guard now declines an unadmitted converged native result and falls back
to original Newton. Actual native Rust owner copies the guess and advances no
physical history, so declining that speculative result is safe. Expanded97 PASS
`254d5d7ef9d06fe01406302097be934a28adb26a8ffeaf8b67b6fe35593e9917@18`.
Approved installer13391 succeeded;75/75 installed Python files byte-identical;
`time_integrators/_step.py` SHA256
`a2532637c2e6e1e65df6d5ddff863048a2c857f314478cb00accc627d2c14816`.
Final root consumers after this latest install remain pending.

Provenance limitation: current external lab_commons verifier hardcodes measured
targets src/tests/pyproject.toml, omitting this project's declared source_roots
python. Its unchanged RED/GREEN tree digest is NOT a production fingerprint.
Actual executions, file SHA and installed75-file parity bind the above evidence;
no cached external source was edited to disguise this tool defect.
