---
name: report-solver-iteration-method-comparison
description: Cross-solver (fea/mec/sdm) benchmark of nonlinear iteration strategies — iteration count, accuracy, wall-time
module: maxwell
created: 2026-07-11
accessed: 2026-07-11
tags: [report, benchmark, nonlinear, iteration-strategy, picard, newton, anderson, fea, mec, sdm]
---

# Solver × iteration-method comparison (fea / mec / sdm)

Head-to-head of the 12 `ITERATION_REGISTRY` nonlinear outer strategies across the three in-house
solvers, on their own `NonlinearSystem` seam. Measured 2026-07-11 on `night` (post SDM DAY-5 JFNK
tangent, commit `963fdbb`). Driver: `tmp/bench.py` (min-of-3 timing; strategies run directly on each
solver's system so `IterationResult.iterations` is exact — the CLI/solve entry does NOT expose it).

## TL;DR
- **Accuracy is NOT the differentiator.** Every strategy that converges lands on the *same* fixed
  point (root Δ ≈ 1e-5…1e-9 vs damped-Picard). They solve the identical nonlinear system — the outer
  strategy changes **iteration count / robustness / wall-time**, never the converged field.
- **`picard-damped` is the right default**: robust, byte-exact goldens, no tangent, optimal on
  contractive (mildly-saturated) problems.
- **`anderson` = best "free" upgrade**: no Jacobian, ~halves iterations on stiff maps, same root.
- **`newton` wins on DEEP saturation** (exact tangent fea+mec, matrix-free JFNK sdm): fewest
  iterations; wins WALL-TIME only when the per-iterate assemble is expensive (real machine solves).

## ⚠️ Wall-time is NOT fair yet — read before citing the ms columns
The `time_ms` columns below are **informational only** and will be **re-measured after the Rust
rewrite**. They are NOT a fair algorithm comparison and MUST NOT gate anything, because:
1. **Pure-Python constant factors dominate.** Assemble + linear-solve are Python+numpy today, so the
   ms compares *whose Python path is shorter*, not the algorithm's floating-point work.
2. **Cases are small and contractive.** The MEC net converges in 5 iters for every strategy; the
   synthetic sdm map's assemble is so cheap that JFNK's extra matvecs make Newton (0.7 ms) *slower*
   than Picard (0.1 ms) despite fewer iters. Constant factors bury the asymptotic difference.
3. **The cost model is a RELATIVE ratio.** Newton wins iff `n_iter_newton × k_matvec < n_iter_picard`.
   Rust compresses assemble / tangent / GMRES by *different* factors, so this inequality's crossover
   **moves** — the "Newton only wins wall-time on expensive assemble" boundary shifts LEFT once per-iter
   constants are flattened and iteration-count differences dominate again.

**What is fair and portable (safe to keep, safe to GATE on):**
- **Accuracy** — all convergent strategies hit the same root (Δ≈1e-5…1e-9). A mathematical property,
  language-independent; Rust will not overturn it.
- **Iteration count** — `IterationResult.iterations` is an algorithmic quantity, invariant to language
  (Picard 66 → Newton 8 is portable). **Gates assert iteration count + root Δ, never wall-time.**

**Action after the Rust rewrite:** mark the `time_ms` tables STALE and re-run on a REAL full-machine
case (this is MANUAL-326: thread `IterationResult.iterations` to the `--iteration-strategy` multi-select
for an end-to-end benchmark), NOT the synthetic small maps. The iteration-count and accuracy tables carry over unchanged.

## Strategy families & per-solver support
| family | strategies | fea | mec | sdm |
|---|---|---|---|---|
| Picard | picard-damped*, picard, picard-adaptive | ✅ | ✅ | ✅ |
| Newton | newton, modified-newton, newton-linesearch, newton-krylov | ✅ exact tangent νK+2ν'/Δ·GGᵀ | ✅ exact dν/dB² tangent | ✅ matrix-free JFNK |
| quasi-Newton (secant) | quasi-newton (Broyden), bfgs, l-bfgs | ✅ | ✅ | ❌ raises — no secant operator |
| accel / homotopy | anderson, continuation | ✅ | ✅ | ✅ (continuation = source ramp) |
`*` default. Only gap: sdm has no secant operator → Broyden/BFGS/L-BFGS raise (Principle-P xfail).

## Measured results

### MEC — `_ReluctanceSystem`, small saturating BH network (very contractive)
| strategy | iters | conv | time_ms | root Δ |
|---|---|---|---|---|
| picard-damped | 5 | ✓ | 1.5 | 0 (ref) |
| picard / picard-adaptive | 5 | ✓ | 1.5 | 0 |
| anderson | 5 | ✓ | 1.6 | 3.6e-9 |
| newton | 5 | ✓ | 2.3 | 3.4e-9 |
| modified-newton | 5 | ✓ | 1.9 | 3.3e-9 |
| quasi-newton / bfgs / l-bfgs | 5 | ✓ | 1.6 | ~3.6e-9 |
| newton-linesearch | 5 | ✓ | 2.9 | 3.4e-9 |
| newton-krylov | 5 | ✓ | 3.1 | 3.4e-9 |
| continuation | 5 | ✓ | 4.8 | 4.7e-9 |
Read: so contractive that Picard converges in 5 iters — nothing to accelerate, so all tie on count
and Newton/continuation only add per-iter (or ramp) overhead.

### SDM — `SubdomainSaturationSystem`, synthetic saturating fixed-point (moderate stiffness)
| strategy | iters | conv | time_ms | root Δ |
|---|---|---|---|---|
| picard-damped / picard / picard-adaptive | 15 | ✓ | 0.1 | 0 (ref) |
| **anderson** | **8** | ✓ | 0.1 | 4.8e-5 |
| **newton** (JFNK) | **6** | ✓ | 0.7 | 5.1e-5 |
| modified-newton | 7 | ✓ | 0.9 | 5.1e-5 |
| newton-linesearch | 6 | ✓ | 0.7 | 5.1e-5 |
| newton-krylov | 6 | ✓ | 1.3 | 5.1e-5 |
| continuation | 15 | ✓ | 0.1 | 0 |
| quasi-newton / bfgs / l-bfgs | — | ERR | — | no secant operator |
Read: Picard 15 → anderson 8 → newton 6 (same root). BUT wall-time inverts: this map's assemble is
trivial, so JFNK's extra matvecs make Newton (0.7 ms) SLOWER than Picard (0.1 ms) despite fewer iters.

### SDM — REAL saturating steel (measured on the landed DAY-5 tangent, commit `963fdbb`)
| case | Picard iters | Newton iters | notes |
|---|---|---|---|
| jz = 3e8 | 11 | **6** | same root (Δ ≤ 5.7e-3) |
| jz = 1e9 (deep) | 28 | **7** | |
| ρ = 0.9 stiff seam map | 66 | **8** | Newton err 1e-15 vs Picard 1e-3 |
Wall-time: inexact-GMRES JFNK inner solve took the real newton solve from **~minutes → 0.2 s**
(loose inner rtol + matvec cap; a Krylov breakdown degrades to one Picard step, never a wrong dir).

### FEA — `solve('fea', …, iteration_strategy=…)`, 3×3 saturating steel
Entry hides iteration count (logs show 6 damped-Picard iters); reported as wall-time + root-match:
| strategy | time_ms | root Δ (|B|) |
|---|---|---|
| picard-damped | 2.1 | 0 (ref) |
| picard | 2.4 | 4.6e-5 |
| anderson | 2.6 | 4.5e-5 |
| newton | 2.2 | 4.5e-5 |
| newton-krylov | 2.3 | 4.5e-5 |
| modified-newton | 2.6 | 4.5e-5 |
Read: small case → all ~2 ms, same root. FEA's exact T3 tangent makes Newton land on the frozen
damped-Picard field (correctness gate `test_fea_newton`).

## Analysis / when to use what
- **Accuracy** identical across convergent strategies — pick for iterations/robustness/time, not accuracy.
- **Contractive / lightly-saturated**: keep `picard-damped`; Newton/continuation only add overhead.
- **Stiff / deeply-saturated**: Newton (or `anderson`, tangent-free) sharply cuts iterations (66→8).
  Prefer Newton when the assemble is EXPENSIVE (real solves) so fewer-but-costlier steps win wall-time;
  prefer `anderson` when assemble is cheap or no tangent is available.
- **continuation**: reserve for cold-start non-convergence (very deep saturation); no-op cost otherwise.
- **quasi-Newton**: fea/mec only; sdm lacks a secant → a future sdm secant operator flips that xfail.
- **Cost model**: total ≈ n_iter × cost_per_iterate. Picard iterate = assemble + linear solve; Newton
  iterate = assemble + tangent (fea/mec analytic; sdm = k JFNK matvecs = k assembles). Newton wins iff
  n_iter_newton × k_matvec < n_iter_picard — holds on stiff+expensive-assemble, fails on cheap/contractive.

## Caveats / extend
- Cases are small/representative; MEC network too contractive to show acceleration; the synthetic sdm
  map's cheap assemble understates Newton's real wall-time win (the real-steel table corrects this).
- The solve/CLI entry doesn't surface `IterationResult.iterations`; exposing it would let a full-machine
  `--iteration-strategy a b c …` sweep (the multi-select landed this session) report per-strategy iters
  end-to-end — recommended next step for a production benchmark (queued as a follow-up).
- Reproduce: `PYTHONPATH="$PWD/src:$PWD" python tmp/bench.py`.
