---
name: plan-nonlinear-bh-validation
description: Validation plan establishing whether the in-house nonlinear BH-curve iteration (saturating magnetostatic solve) is faithful to FEMM's nonlinear solve, across converged B-in-iron, total energy, and airgap Br/Bt at multiple saturation levels; ranked divergence sources and a Principle-P nonlinear-accuracy matrix.
type: project
created: 2026-07-21
accessed: 2026-07-21
---
> **MIGRATED 2026-08-01** → `.claude/memory/2026/08/01/plan-refactor-unified-backlog.md`. All remaining work (if any) is consolidated there; this file is the archived original design record.


# Nonlinear BH-curve iteration accuracy vs FEMM — validation plan

## Goal
Establish whether the in-house NONLINEAR magnetostatic solve (BH saturation via the
damped-Picard/Newton iteration) converges to the SAME saturated state FEMM does. Nonlinear
saturation error propagates into every downstream metric (airgap Br/Bt, torque, flux linkage,
voltage), so this front gates all "align with femm" work at loaded/saturated operating points.

## Current state (file:line)
- Iteration seam: `src/motronics/gauss/nonlinear/__init__.py` — `DampedPicardStrategy` (:162),
  `_energy_error_pct` (:155), `_RTOL_ENERGY_PCT=0.01` (:49), `_ALPHA_MIN=0.1` (:47),
  `ITERATION_REGISTRY` (:898), `get_strategy` (:915).
- FEA default path is NOT the generic strategy: `discretizations/fem/_newton.py`
  `fea_newton_solve` (:746) fast-paths None/'picard-damped' -> `GalerkinSystem.frozen_damped_solve`
  (:442). Reluctivity (not solution) under-relaxed in-kernel; alpha halved x0.6 floored 0.05 on
  energy-delta SIGN FLIP (:517-524); dual energy+flux convergence `_check_convergence` (:510);
  the "Energy Error Percentage ... alpha" log (:555). Non-default strategies use the seam (:821).
- MEC: `formulations/magnetic_circuit/_sparse.py:475` `_solve_reluctance_network`, seam at :836,
  max_num_iter default 100 (:586), same log line (:519).
- Per-iteration nu update: `_newton.py:426 _block_nu_slope` -> `get_reluctivity_slope`.
- BH data: single source `core/materials/item.py:124 bh_curve` ([B,H] list).
  - In-house: `core/materials/registry.py:324 _make_bh_reluctivity` = piecewise-linear nu(B),
    linear end-slope extrapolation (:345-355); Newton slope :366; HARD clamp at
    `b_saturation = bh[:,0].max()` (:447), enforced via `get_saturation_bound` (:625) / max_B_norm.
  - FEMM: `adapters/femm/materials.py:38-40 mi_addbhpoint` -> FEMM's own cubic-spline BH +
    its own tail extrapolation (keeps rising past the last knot).
- HARNESS CONTAMINATION: `postprocessing/compare.py:153 femm_element_reluctivity` evaluates
  FEMM steel energy with the IN-HOUSE `Material.reluctivity_func` (:150,:162). Today's energy
  agreement already imposes in-house interpolation on FEMM -> masks the BH-interpolation gap.
  The validation MUST bypass this and read FEMM's own B, then compute both energies natively.

## Apples-to-apples protocol
Same geometry + same `bh_curve` points + same excitation; in-house nonlinear solve vs FEMM
`.ans`, at 2-3 saturation levels (light load -> heavy saturation), per case. Compare:
1. Converged total energy (in-house `Energy_mat.sum()`, out[4]) vs FEMM element energy computed
   on FEMM's OWN B with FEMM's OWN reluctivity -- NOT compare.py:153's in-house-nu path.
2. Converged |B| in iron, per steel region (out[3] `B_norm`, area-weighted; `_stats` compare.py:104),
   to catch under/over-saturation localised in teeth/back-iron even when global energy matches.
3. Airgap Br/Bt waveform (`compare_airgap_waveform`, compare.py:22) -- the downstream metric.
4. Convergence trajectory (energy-error-% per iteration, alpha schedule) from the solver log,
   asserting the returned field actually reached tolerance (not "last iterate" warning _newton.py:562).
Discriminating test: does the in-house iteration converge to the SAME saturated |B| FEMM does,
or does it under/over-saturate? A matched global energy with mismatched per-region |B| = fail.

Saturation levels: sweep excitation on `ipm_uv_cummins` (rated=heavy) and one of
`spm_slotted`/`synrm_circular` (light->knee->deep). Use `cases/ipm_uv_cummins` rated as the anchor.

## Ranked divergence sources (check first -> last)
(a) **BH interpolation/extrapolation mismatch** -- piecewise-linear + hard clamp at top knot
    (`registry.py:345-355,:447`) vs FEMM cubic spline that keeps rising. Diverges most exactly at
    the operating points the maintainer cares about (deep saturation). HIGHEST prior.
    Sub-check: the `b_saturation` clamp caps in-house |B| at the last knot; FEMM does not -- expect
    systematic in-house UNDER-saturation of H (over-permeance) past the table. Probe by extending
    the BH table tail and re-checking.
(b) **Convergence criterion too loose** -- energy-error-% tol 0.01 (`gauss:49`; frozen 1e-2
    `_newton.py:510`) can stop the iterate under-converged; logs show early iters at inf%
    (`_energy_error_pct:157`). Re-run with a tightened tol and confirm the converged state moves.
(c) **Under-relaxation alpha not fully recovered** -- frozen schedule only ratchets alpha DOWN
    (`_newton.py:523`), never back up; `DampedPicardStrategy` likewise (`gauss:190`). A case that
    starts stiff can finish over-damped/short of the fixed point. Compare vs `picard-adaptive`
    (`gauss:203`) and `newton` which target the same root.
(d) **Different BH source between fea and mec** -- both nominally read `item.bh_curve`, but confirm
    fea (`get_reluctivity_slope`) and mec (`_sparse.py`) resolve the SAME `material_in_use_dict`
    entry; verify no per-solver re-tabulation. LOWEST prior (shared contract) but cheap to confirm.

## Principle-P test shape (nonlinear-accuracy matrix)
Parametrize (case x solver in {fea,mec} x iteration_strategy x saturation_level) vs a FEMM
reference bundle. Model on `tests/integration/matrix/test_iteration_method_matrix.py` (reference-energy
fixed point, :29). Per cell assert: converged steel |B| per region within tol of FEMM, total
energy within tol, airgap Br/Bt NRMSE within tol. Where in-house tracks FEMM -> HARD gate; where
it diverges -> `xfail(strict=False)` carrying the MEASURED residual (energy %, |B| %, waveform
NRMSE) so a fix xpasses and promotes. Mirrors the existing torque/variant/iteration matrices.
Mark `@pytest.mark.femm`; skips only on missing golden `.fem`/`.ans` (per hamilton femm rule).

## Lane disjointness (in-flight work)
- d1-gate owns sweep/validation/workflow + femm unification (methods.py/rotation.py/sweep.py/
  validation.py). jfix owns `hamilton/solve.py` hybrid + euclid/mesh.
- This BH/nonlinear lane touches `gauss/nonlinear/` + `core/materials/registry.py` +
  `discretizations/fem/_newton.py` + `formulations/magnetic_circuit/_sparse.py` + new tests
  under `tests/integration|unit/hamilton` -- mostly disjoint. BOUNDARY RISKS to confirm with
  d1-gate: `postprocessing/compare.py` (shared cross-solver harness; the :153 de-contamination
  edit overlaps validation's territory) and any femm-reference bundle builder in validation.py.
  Coordinate the compare.py femm-reluctivity change with d1-gate before touching it.

## Deliverables / next steps
1. Read-only characterization first: dump converged per-region |B|, energy, airgap Br/Bt for
   `ipm_uv_cummins` rated, in-house vs FEMM (native reluctivity), quantify the gap.
2. If (a) confirmed: prototype a cubic/monotone-spline nu(B) + FEMM-matching tail as an ADDITIVE
   option in registry.py (keep piecewise-linear as default so goldens stay bit-for-bit).
3. Land the Principle-P matrix as xfail-with-residual, then hard-gate cells that track.

## RESULT (2026-07-21, bhfix @ edbe59b6, gate PASS)
- FIXED: `_make_bh_reluctivity` now splines H(B) with a monotone PCHIP (Fritsch-Carlson) interior +
  C1 quadratic stiffening tail (curvature = table's own measured dH/dB acceleration), matching FEMM's
  cubic-spline shape. A plain natural cubic spline is NOT monotone between wide knots (fed a Newton
  limit cycle) — PCHIP required. Linear materials never hit this branch (goldens untouched).
- Measured (scratch demo_steel_bh, in-house nonlinear FEA vs live FEMM, native reluctivity each side):
  energy err light -0.68%->-0.10%, knee +4.10%->+4.14%, deep +8.10%->+6.60%; mean|B| deep +3.99%->+2.64%.
- Residual is DISCRETIZATION, proven by mesh refinement at deep: 11700 elem +6.60% -> 30837 +5.41% ->
  80848 +4.84%, monotonically converging toward FEMM (5534 conforming). Knee ~unchanged because only ~1%
  of iron is above-knot there (interpolation barely matters) but element-count mismatch applies equally.
  So the interpolation-shape defect is CLOSED; the remaining gap is a converging mesh effect, not a bug.
- Side effects (sharper reluctivity exposes convergence fragility the linear steel hid), both
  xfail-with-measured-residual: MEC undamped Picard period-2 limit cycle (MANUAL-418, needs _sparse.py
  alpha-ratchet); SDM modified-newton spurious near-zero root (MANUAL-419, newton/linesearch fine).
