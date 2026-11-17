---
name: solver-iteration-methods-decoupling-and-comparison
description: "The 2026-07-09 mesh / equation / nonlinear-iteration decoupling design, the solver-methods roadmap (pluggable registry + permanent comparison harness), the 2026-07-11 benchmark of the 12 ITERATION_REGISTRY strategies across fea/mec/sdm, and the 2026-07-14 plan that exposed --linear-solver/--preconditioner."
created: 2026-07-09
accessed: 2026-09-27
merged_from: .claude/memory/2026/07/14/plan-wire-linear-precond-cli.md, .claude/memory/2026/07/11/report-solver-iteration-method-comparison.md, .claude/memory/2026/07/09/design-solver-iteration-decoupling.md, src/motronics/hamilton/.claude/memory/2026/07/09/roadmap-solver-methods.md
metadata:
  type: plan
---

# Solver numerical methods: iteration decoupling design, methods roadmap, cross-solver comparison, linear solver CLI

## Current conclusion

- Nonlinear iteration is decoupled behind `NonlinearSystem` (`maxwell/nonlinear.py`; `DampedPicardStrategy`/`PicardStrategy` extracted from MEC); the 12 registry strategies were benchmarked head-to-head on 2026-07-11 (`night`, post SDM DAY-5 JFNK tangent `963fdbb`).
- `--linear-solver` / `--preconditioner` are exposed on the CLI (plan CLOSED 2026-08-01 as done, `__main__.py:1149` at the time).
- The roadmap's vision stands: a pluggable registry of outer iteration, inner linear solve, preconditioning, time stepping and coupling, plus a permanent comparison harness.

## History

- Design progress of 2026-07-09: SR-008 band vocab `d3ad739`, differential reluctivity `40a94a6`, MEC loop extraction.

## Members, newest first (full text; the older ones are the History above refers to)

### Member `.claude/memory/2026/07/14/plan-wire-linear-precond-cli.md`

Original frontmatter:

```yaml
module: plan
created: 2026-07-14
priority: high
tags: [cli, linear-solver, preconditioner, sync-gate, gap]
```

> **MIGRATED 2026-08-01** → `.claude/memory/2026/08/01/plan-refactor-unified-backlog.md`. All remaining work (if any) is consolidated there; this file is the archived original design record. **CLOSED 2026-08-01 -- done (--linear-solver/--preconditioner exposed at __main__.py:1149)**


### Plan — Wire --linear-solver and --preconditioner to CLI

**Priority: HIGH.** Two registries are implemented, test-validated, and reachable programmatically
but NOT exposed at the command line — users cannot select Krylov backends or preconditioners from
`uv run motronics solve`.

#### Gap

| Registry | Module | CLI exposed? | Sync gate test? |
|---|---|---|---|
| `ITERATION_REGISTRY` | `gauss/nonlinear/__init__.py` | `--iteration-strategy` with `choices=sorted(ITERATION_REGISTRY)` | n/a (dynamic `choices` + help-string text) |
| `LINEAR_SOLVER_REGISTRY` | `gauss/linear/__init__.py` | **NO** | **NO** |
| `PRECONDITIONER_REGISTRY` | `gauss/linear/__init__.py` | **NO** | **NO** |

`preconditioner` is referenced in the sweep cache key (`__main__.py` :2414) as
`getattr(args, 'preconditioner', None)` — a dead read that always falls back to `None` because the
argument is never declared.

#### What to do

1. Add `--linear-solver` to the `run` / `case run` subparsers with `choices=sorted(LINEAR_SOLVER_REGISTRY)`.
2. Add `--preconditioner` to the same subparsers with `choices=sorted(PRECONDITIONER_REGISTRY)`.
3. Thread both through `run_workflow` → the solver call (already supported at the
   `NonlinearSystem` / `GalerkinSystem` level — the seam exists, just the CLI knob is missing).
4. Add registry-parametrized sync gate tests (same pattern as `test_every_base_solver_is_cli_reachable`):
   - `test_every_linear_solver_is_cli_reachable` — parametrize over `LINEAR_SOLVER_REGISTRY`
   - `test_every_preconditioner_is_cli_reachable` — parametrize over `PRECONDITIONER_REGISTRY`
5. Remove the dead `getattr(args, 'preconditioner', None)` fallback — it should be a real argparse dest.

#### Pattern to follow

Existing sync gate: `tests/unit/workflows/test_cli_sync_gate.py` parametrizes over
`SOLVER_SPECS` / `CANONICAL_VARIANTS` / `known_variants()` and asserts each is reachable
via `-s`. Same pattern for the linear/preconditioner registries. Existing CLI pattern:
`choices=sorted(ITERATION_REGISTRY)` on `--iteration-strategy` — the dynamic `choices` list auto-syncs
the `--help` text without a separate test.

### Member `.claude/memory/2026/07/11/report-solver-iteration-method-comparison.md`

Original frontmatter:

```yaml
name: report-solver-iteration-method-comparison
description: Cross-solver (fea/mec/sdm) benchmark of nonlinear iteration strategies — iteration count, accuracy, wall-time
module: maxwell
created: 2026-07-11
accessed: 2026-07-11
tags: [report, benchmark, nonlinear, iteration-strategy, picard, newton, anderson, fea, mec, sdm]
```

### Solver × iteration-method comparison (fea / mec / sdm)

Head-to-head of the 12 `ITERATION_REGISTRY` nonlinear outer strategies across the three in-house
solvers, on their own `NonlinearSystem` seam. Measured 2026-07-11 on `night` (post SDM DAY-5 JFNK
tangent, commit `963fdbb`). Driver: `tmp/bench.py` (min-of-3 timing; strategies run directly on each
solver's system so `IterationResult.iterations` is exact — the CLI/solve entry does NOT expose it).

#### TL;DR
- **Accuracy is NOT the differentiator.** Every strategy that converges lands on the *same* fixed
  point (root Δ ≈ 1e-5…1e-9 vs damped-Picard). They solve the identical nonlinear system — the outer
  strategy changes **iteration count / robustness / wall-time**, never the converged field.
- **`picard-damped` is the right default**: robust, byte-exact goldens, no tangent, optimal on
  contractive (mildly-saturated) problems.
- **`anderson` = best "free" upgrade**: no Jacobian, ~halves iterations on stiff maps, same root.
- **`newton` wins on DEEP saturation** (exact tangent fea+mec, matrix-free JFNK sdm): fewest
  iterations; wins WALL-TIME only when the per-iterate assemble is expensive (real machine solves).

#### ⚠️ Wall-time is NOT fair yet — read before citing the ms columns
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

#### Strategy families & per-solver support
| family | strategies | fea | mec | sdm |
|---|---|---|---|---|
| Picard | picard-damped*, picard, picard-adaptive | ✅ | ✅ | ✅ |
| Newton | newton, modified-newton, newton-linesearch, newton-krylov | ✅ exact tangent νK+2ν'/Δ·GGᵀ | ✅ exact dν/dB² tangent | ✅ matrix-free JFNK |
| quasi-Newton (secant) | quasi-newton (Broyden), bfgs, l-bfgs | ✅ | ✅ | ❌ raises — no secant operator |
| accel / homotopy | anderson, continuation | ✅ | ✅ | ✅ (continuation = source ramp) |
`*` default. Only gap: sdm has no secant operator → Broyden/BFGS/L-BFGS raise (Principle-P xfail).

#### Measured results

##### MEC — `_ReluctanceSystem`, small saturating BH network (very contractive)
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

##### SDM — `SubdomainSaturationSystem`, synthetic saturating fixed-point (moderate stiffness)
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

##### SDM — REAL saturating steel (measured on the landed DAY-5 tangent, commit `963fdbb`)
| case | Picard iters | Newton iters | notes |
|---|---|---|---|
| jz = 3e8 | 11 | **6** | same root (Δ ≤ 5.7e-3) |
| jz = 1e9 (deep) | 28 | **7** | |
| ρ = 0.9 stiff seam map | 66 | **8** | Newton err 1e-15 vs Picard 1e-3 |
Wall-time: inexact-GMRES JFNK inner solve took the real newton solve from **~minutes → 0.2 s**
(loose inner rtol + matvec cap; a Krylov breakdown degrades to one Picard step, never a wrong dir).

##### FEA — `solve('fea', …, iteration_strategy=…)`, 3×3 saturating steel
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

#### Analysis / when to use what
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

#### Caveats / extend
- Cases are small/representative; MEC network too contractive to show acceleration; the synthetic sdm
  map's cheap assemble understates Newton's real wall-time win (the real-steel table corrects this).
- The solve/CLI entry doesn't surface `IterationResult.iterations`; exposing it would let a full-machine
  `--iteration-strategy a b c …` sweep (the multi-select landed this session) report per-strategy iters
  end-to-end — recommended next step for a production benchmark (queued as a follow-up).
- Reproduce: `PYTHONPATH="$PWD/src:$PWD" python tmp/bench.py`.

### Member `.claude/memory/2026/07/09/design-solver-iteration-decoupling.md`

Original frontmatter:

```yaml
module: maxwell
severity: MEDIUM
status: open
created: 2026-07-09
accessed: 2026-07-09
tags: [design, refactor, architecture, solver, mesh-decouple, iteration, newton-raphson, picard, principle-p]
```

### DESIGN — decouple mesh / governing-equation / nonlinear-iteration; plug in every iteration method

> **PROGRESS (2026-07-09).** DONE: §7 step 1 SR-008 band-family vocab (`d3ad739`, bit-for-bit);
> §4 prereq differential reluctivity `dν/dB²` (`40a94a6`); §7 step 3 MEC loop extraction into
> `DampedPicardStrategy` + `PicardStrategy` behind `NonlinearSystem`/`maxwell/nonlinear.py`
> (`04862d5`, bit-for-bit: MEC energy unchanged to the last digit); and **§4 NewtonStrategy for MEC**
> — the exact loop-space Jacobian `J = R_loop + L_redᵀ diag(2 (l/A) B² dν/dB²) L_red`, verified to
> converge to the SAME solution as damped-Picard on a nonlinear network AND on the real synrm_circular
> / ipm_uv cases (energy identical to the last digit). `iteration_strategy` is threaded through
> `mec_solve_magnetostatic[_sparse]` and the standard `solve('mec'/'fea', iteration_strategy=)` path
> (default `None` = damped-Picard, unchanged); Principle-P iteration-method matrix done (`61df7b2`).
> **FEA Newton DONE** (`6bc5c5e`, freeze lifted): `fea/_newton.py` — the exact T3 nodal tangent
> `J_e = ν K_e^geom + (2ν'/Δ) G Gᵀ`, additive parallel path, verified BIT-FOR-BIT vs damped-Picard on
> the real synrm_circular FEA case. Newton is now in BOTH cores.
> **FEA Q4 + mixed Newton + GalerkinSystem DONE** (Task G, `7746a1e`): the T3-only hand-rolled loop was
> refactored into a `GalerkinSystem` implementing `NonlinearSystem` (assemble/residual/tangent/
> solve_linear/solve_system), so EVERY strategy (damped-Picard/Picard/Newton) drives the SAME assembled
> FEA system across T3, Q4, AND mixed. The Q4 tangent is the general `J_e = ν g_raw + ν' (g_raw a) Hᵀ`
> with the CENTROID `H = 2[(a·cby)cby + (a·cbx)cbx]` (the exact Jacobian of the frozen Q4 centroid-ν
> residual, FD-validated to 1e-12); T3's `G Gᵀ` is the special case `H = (2/Δ)G`. Verified BIT-FOR-BIT
> (rel_a = 0.0) vs damped-Picard on real synrm_circular T3/Q4/mixed BANDED meshes WITH sliding pairs.
> The default frozen damped-Picard `fea_solver` loop is untouched (byte-identical goldens; `iteration_
> strategy=None` never touches the GalerkinSystem). Matrix-gated in `test_iteration_method_matrix`.
> (The `mec:hybrid/newton` variant-suffix sugar remains an optional extension.) NB: the session's original git worktree was destroyed externally
> mid-run; work continues on branch `worktree-fea-quad-band-fix` in a re-created worktree
> (`.claude/worktrees/fea-quad-band-fix-r2`) — all commits are intact on the branch.

Future architecture + refactor. Grounded in the code as of `f71809b`. First principle applies:
IMPLEMENT every iteration method (Newton-Raphson, Picard, Picard+adaptive damping) and CHARACTERIZE
it — never exclude a method because an experiment showed it inferior. It becomes a labelled,
gated variant exactly like the mesh variants (`fea:hybrid`, `mec:tri`, ...).

#### 0. Answers to the three questions (grounded, before the design)

- **`mec:hybrid` is NOT a pure triangular mesh.** It takes the conforming triangular CDT mesh and
  *merges triangle pairs into quads* (`euclid/mesh/merge.py::merge_tris_to_quads`, driven by
  `maxwell/mec/hybrid.py::mec_solve_hybrid`), producing a MIXED-valence network (quads in the bulk,
  triangles only where a pair cannot merge). The per-vertex result is mapped back onto the ORIGINAL
  triangles so every downstream diagnostic runs on the plain tri mesh — which is why it *reads* as
  triangular. So "hybrid" = mixed valence built FROM triangles, not "rectangle+triangle laid down
  from scratch". `finding-triangular-mec.md` has the rationale (tri conforms but is shape-sensitive;
  quad is mesh-stable; merging pairs pulls energy into the stable quad regime while staying conforming).
- **"Merge small rectangles → large rectangles" IS implemented:**
  `euclid/mesh/rectangular_merge.py::aligned_merge_regions` agglomerates same-material cells onto a
  power-of-two lattice so coarse/fine interfaces NEST (clean 2:1 hanging nodes, not staggered). The
  T-junction hanging nodes (`_hanging_from_regions`) are folded into BOTH solvers: MEC via
  `mec_solve_magnetostatic(hanging_constraints=...)` (`mec/_sparse.py::_build_reduction`) and FEA via
  `fea/_solve.py::_apply_hanging_nodes`.
- **It IS wired into the `motronics case` / workflow path — opt-in, not default.**
  `run_workflow(merge_cells=, merge_max_cell_ratio=)`, CLI `motronics ... --merge --merge-ratio N`
  (`__main__.py`), plus the energy-gated auto-pick `workflow.select_merge_ratio` /
  `calibrate_merge_ratio` (walk fine→coarse, keep the coarsest merge within `rtol` energy of the
  unmerged solve). It is flagged "accuracy still being iterated" and applies to mec/sdm only
  (fea/femm unaffected). So the correct statement is "integrated but experimental/opt-in", not
  "not integrated". Making it a default with a per-case energy-gated ratio is a small follow-on.

#### 1. Where the coupling actually is today (file:line)

- **`solve.py`** already decouples mesh at the CALL boundary: `solve(solver_type, **kwargs)` →
  `SOLVER_REGISTRY[solver_type](**kwargs)` takes raw arrays (`ele_mat`, `vert_mat`,
  `group_material_list`, `conductor_current_density_list`, `bound_vert_ids_dict`, `magnet_br_list`,
  `hanging_constraints`). Good seam, but the mesh-VARIANT choice leaks in as `extras`
  (`fea_mesh`/`mec_mesh`) with the SR-008 vocabulary split (`run_workflow` uses `mec_mesh`,
  `run_rotation_sweep` uses `fea_mesh` for both solvers; MANUAL-20260709-272).
- **Governing equation and nonlinear iteration are ENTANGLED inside each solver.**
  - MEC: `mec/_sparse.py::_solve_reluctance_network` L341 — one `for i_iter in range(max_num_iter)`
    loop that (a) recomputes `ν(|B|)` per branch, (b) reassembles `R_loop`, (c) `spsolve`, (d) applies
    `phi = phi + alpha*(solve - phi)` with `alpha` halved on energy-error growth (`_ALPHA_MIN=0.1`),
    (e) energy-change convergence test (`_RTOL_ENERGY_PCT`). Picard + adaptive under-relaxation ONLY.
  - FEA (FROZEN): `fea_solver.py` L160 — same shape, `nu_relax_mat` carrying the damped per-element
    reluctivity, `alpha` ratcheted down and never raised, `_check_convergence` on energy+flux.
    Picard + adaptive under-relaxation ONLY.
  - **No Newton-Raphson, no Jacobian anywhere.** The nonlinear method is not selectable; assembly,
    damping schedule, and convergence test are hardcoded in the loop body.
- **Material layer exposes only ν(B):** `core/materials/registry.py::get_reluctivity(name, B=)`.
  There is NO differential reluctivity `dν/d(B²)` — required for a Newton tangent (see §4).

#### 2. Target architecture — three independent seams

```
DiscreteModel  ──►  GoverningEquation  ──►  IterationStrategy  ──►  SolveResult
(mesh+materials     (assembles residual     (drives A_{k+1} from     (fields+energy
 +sources, one       R(A) and, optionally,   R and J; Picard /        +accuracy label)
 value object)       tangent J(A))           Newton / damped)
```

##### Seam 1 — Mesh ↔ solve: a `DiscreteModel` value object
One frozen dataclass carrying everything a solver needs, produced by mesh builders, consumed by
every governing equation. Kills the array-tuple plumbing AND the `fea_mesh`/`mec_mesh` split.

```python
@dataclass(frozen=True)
class DiscreteModel:
    ele: list[np.ndarray]  # per-element vertex ids (valence 3/4/N -> already mixed-valence ok)
    vert: np.ndarray  # (n,2/3)
    group_of_ele: np.ndarray
    group_material: list[tuple]
    materials: MaterialSet
    sources: Sources  # conductor current density + magnet Br
    boundaries: Boundaries  # dirichlet / periodic / sliding / hanging constraints
    depth: float
    provenance: MeshVariant  # 'tri'|'quad'|'hybrid'|'tri->quad-merged'|'rect-merged' -- the LABEL
```
Mesh builders become pure producers: `build_tri`, `build_quad`, `build_hybrid` (tri-pair merge),
`build_rect_merged` (aligned_merge_regions) all return a `DiscreteModel`. The `provenance` field is
the canonical variant id (feeds `maxwell/accuracy.py`), replacing the two divergent mesh vocabularies
(SR-008/014/015 / MANUAL-20260709-272 — do this seam and that finding closes for free).

##### Seam 2 — Governing equation ↔ iteration: a `NonlinearSystem` interface
The equation knows how to ASSEMBLE; it does not know HOW to iterate.

```python
class NonlinearSystem(Protocol):
    def initial_guess(self) -> np.ndarray: ...
    def residual(self, A: np.ndarray) -> ResidualState:  # R(A), B(A), energy, ν(B) -- reused by both methods
        ...
    def secant_operator(self, st: ResidualState) -> spmatrix:  # K(ν(B)) with BC/periodic/hanging folded
        ...
    def tangent_operator(self, st: ResidualState) -> spmatrix:  # K(ν) + dν-term (None -> strategy falls back to Picard)
        ...
    def energy(self, st: ResidualState) -> float: ...
```
Two implementations, sharing `mec/_assembly.py` / `fea/_assemble*.py` element kernels:
`ReluctanceSystem` (MEC) and `GalerkinSystem` (FEA). Assembly stays byte-identical to today's frozen
kernels; only the LOOP moves out.

##### Seam 3 — the pluggable `IterationStrategy` (the heart of the directive)
```python
class IterationStrategy(Protocol):
    name: str  # 'picard' | 'picard-damped' | 'newton'

    def solve(self, sys: NonlinearSystem, *, max_iter: int, rtol: float) -> IterationResult: ...
```
Registered in a strategy registry (mirrors `SOLVER_REGISTRY`), selectable per solve:
```python
ITERATION_REGISTRY = {'picard': PicardStrategy, 'picard-damped': DampedPicardStrategy, 'newton': NewtonStrategy}
```
- **`PicardStrategy`** — plain successive substitution: `A_{k+1} = K(ν(B_k))^{-1} F`. (No damping —
  the "naive" baseline that historically diverged on saturating B-H; we KEEP it, characterized as
  `degraded/oscillatory`, so its failure mode is a labelled datum, not a hidden exclusion.)
- **`DampedPicardStrategy`** — the current production loop, extracted verbatim: `A_{k+1} = A_k +
  α(A_solve − A_k)`, α halved on energy-error growth, floored at `_ALPHA_MIN`. This is the current
  MEC/FEA behaviour, now a named strategy (regression: identical iterates → goldens bit-for-bit).
- **`NewtonStrategy`** — `J(A_k) ΔA = −R(A_k); A_{k+1} = A_k + λΔA` with an optional line-search λ.
  Needs `tangent_operator` (§4). Falls back to damped-Picard when `tangent_operator` returns `None`
  (e.g. a material with no dν slope), so wiring Newton never breaks a solver that cannot supply a Jacobian.

#### 3. Dispatch — how a solve names its (equation × mesh × iteration)

The variant label grows one axis. Today: `solver:mesh` (`mec:hybrid`). Add iteration as an optional
suffix, defaulting to today's behaviour so nothing changes unless asked:
`mec:hybrid` == `mec:hybrid/picard-damped` (default); `mec:hybrid/newton`, `mec:tri/picard`, etc.
`solve()` parses `(equation, mesh, iteration)`, builds the `DiscreteModel` (seam 1), wraps it in the
`NonlinearSystem` (seam 2), and hands it to `ITERATION_REGISTRY[iteration]` (seam 3). The
allowlist in `compare_torque_waveforms` / `_derive_solver_kwargs` extends the same way.

#### 4. Newton-Raphson for magnetostatics — the concrete Jacobian (what's missing)

Residual `R(A) = K(ν(|B|²)) A − F`, with `B = ∇×A` (2D: `Bx=∂A/∂y, By=−∂A/∂x`). The tangent is the
well-known two-term nonlinear-magnetostatic Jacobian:

```
J = ∂R/∂A = K(ν)  +  Σ_e  2 (dν/dB²)_e · (K_e A)(K_e A)ᵀ / (element geometry)
             └secant┘     └────────── differential (dν) stiffness term ──────────┘
```
i.e. secant stiffness + a rank-contribution per element proportional to `ν' = dν/d(B²)` along the
local B direction. Concretely per element: `J_e = ν K_e^geom + 2 ν'_e (K_e^geom a_e)(K_e^geom a_e)ᵀ`.

Prerequisites (all in NON-frozen layers):
1. `core/materials/registry.py`: add `reluctivity_slope(name, B) -> dν/d(B²)` (spline derivative of
   the existing B-H interpolant; `0` for linear materials). This is the ONLY physics-data addition.
   **DONE** (commit after d3ad739): `MaterialRegistry.get_reluctivity_slope` + `Material.
   reluctivity_slope_func` + `_make_bh_reluctivity_slope`; `dν/dB²=(dν/dB)/(2B)` from the
   piecewise-linear ν(B) segment slope, floored to 0 at/below `|B|=1e-9` and for linear materials.
   FD-verified in `tests/unit/core/test_material.py`.
2. `mec/_assembly.py` + `fea/_assemble*.py`: a `tangent_element(...)` alongside the existing
   `secant/stiffness` element routine (additive — the frozen secant kernel is untouched).
Expected payoff: quadratic convergence near the solution (Picard is linear and needs the damping
crutch on saturating cases); the damped-Picard stays as the robust globaliser / Newton fallback.

#### 5. The freeze boundary (invariant.md) — migration order

- **MEC is NOT frozen** → land all three seams on MEC first. `mec/_sparse.py` loop → extract
  `DampedPicardStrategy`; add `PicardStrategy` + `NewtonStrategy`; assert the extracted damped path
  reproduces today's energy iterates (characterization/golden tests bit-for-bit).
- **FEA + SDM solving logic ARE frozen** → the frozen `fea_solver.py` damped-Picard path stays
  byte-for-byte. Newton/plain-Picard for FEA is an ADDITIVE parallel path (exactly how the native Q4
  kernel `fea/_assemble_quad.py` was added), dispatched by the iteration suffix, and needs the same
  explicit in-the-moment approval the Q4 addition got. Default dispatch keeps the frozen path.

#### 6. Principle-P wiring — every method characterized, none excluded

Each new `(equation × mesh × iteration)` combo is:
1. registered in `maxwell/accuracy.py` with a measured `RegistryEntry` (convergence + torque/energy
   band): e.g. `('mec:hybrid/newton', '*')` validated if it lands, `('mec:*/picard', '*')` labelled
   `degraded` (oscillatory) — the plain-Picard divergence becomes an HONEST label, not a silent drop;
2. added to the discovery matrices — the same shape as the just-shipped
   `tests/integration/matrix/test_torque_full_matrix.py`: connect the combo, `xfail(strict=…)` a known miss
   (e.g. plain-Picard non-convergence → `xfail(strict=False)` until/unless it is fixed), never skip;
3. a convergence-characterization test (iters-to-tol, final residual) per strategy, tolerances not
   hashes. A method proven worse is KEPT and labelled, so the comparison across methods (the
   directive's "compare the various iteration methods" (user, translated)) is a first-class, permanently-run artifact, not a one-off experiment.

#### 7. First concrete steps (smallest safe increments)

1. Close SR-008 (MANUAL-20260709-272): one canonical `MeshVariant -> (equation, mesh_builder)` map;
   `run_workflow`/`run_rotation_sweep`/`compare_torque_waveforms` consume it. Prereq for seam 1.
2. Introduce `DiscreteModel` as a thin adapter over today's arrays (no behaviour change), route
   `solve()` through it.
3. Extract MEC's damped-Picard into `DampedPicardStrategy` behind `NonlinearSystem` (goldens
   bit-for-bit) — pure refactor, proves seams 2+3 with zero numeric change.
4. Add `reluctivity_slope` + `NewtonStrategy` + `PicardStrategy` on MEC; register + characterize both
   in `accuracy.py` and a new `test_iteration_method_matrix` (Principle P).
5. Only then, with approval, add the additive Newton/Picard FEA path.

Related: `handoff-remaining-work.md` R1–R3 (mec airgap accuracy) is ORTHOGONAL to this — the mortar
/ consistent-gap work is a governing-equation change; it slots into `ReluctanceSystem` unchanged by
the iteration seam.

### Member `src/motronics/hamilton/.claude/memory/2026/07/09/roadmap-solver-methods.md`

Original frontmatter:

```yaml
module: maxwell
severity: MEDIUM
status: open
created: 2026-07-09
accessed: 2026-07-09
tags: [roadmap, solver, iteration, nonlinear, linear-solver, preconditioner, continuation, anderson, newton-krylov, principle-p, comparison-experiment]
```

### ROADMAP — the solver layer supports (nearly) every numerical method, and compares them

Vision: the solver layer becomes a **pluggable registry of numerical methods** — outer nonlinear
iteration, inner linear solve, preconditioning, time-stepping, and multiphysics coupling — and a
**permanent comparison harness** measures every method on every case. First principle (`AGENTS.md`):
IMPLEMENT every method and CHARACTERIZE it; never exclude one because an experiment showed it
inferior — a method proven worse is KEPT and labelled, so the comparison is a first-class artifact.

This grounds that vision on the architecture already built this session (branch
`worktree-fea-quad-band-fix`). Read `design-solver-iteration-decoupling.md` (a179884) and
`plan-mesh-solver-iteration-layering.md` first.

**This roadmap is LAYER 3 (Iteration Strategy) + the layer-2/3 linear-solver seam of the three-layer
/ three-agent decomposition** in `plan-mesh-solver-iteration-layering.md` ("THREE-LAYER / THREE-AGENT
DECOMPOSITION"). Honour its cross-agent constraints: touch neither physics nor mesh; meet layer 2 only
through `NonlinearSystem`; keep the default (damped-Picard) path bit-for-bit; a new method is additive
+ labelled + matrix-gated, never a default numeric change.

#### 0. What already exists (the seam every method plugs into)

- **`maxwell/nonlinear.py`** — `NonlinearSystem` protocol (`initial_guess` / `assemble` ->
  `AssembledState{system_matrix, total_energy, payload}` / `solve_linear` / `residual` / `tangent`
  / `solve_system` / `on_iteration`), `IterationResult`, an `ITERATION_REGISTRY`, and `get_strategy`.
  Implemented strategies: **`picard`, `picard-damped` (default, extracted bit-for-bit), `newton`**.
- **MEC system** `mec/_sparse.py::_ReluctanceSystem` — assembles the reduced loop system, the residual
  `R_loop A - mmf`, and the exact loop-space tangent `R_loop + L_redᵀ diag(2(l/A)B² dν/dB²) L_red`.
- **FEA Newton** `fea/_newton.py::fea_newton_solve` — the exact T3 nodal tangent
  `J_e = ν K_e^geom + (2ν'/Δ) G Gᵀ` (additive parallel path; the frozen damped-Picard is untouched).
- **Material derivative** `core/materials/registry.py::get_reluctivity_slope` (`dν/dB²`).
- **Dispatch** — `iteration_strategy` flows through `solve('mec'/'fea', ...)`,
  `mec_solve_magnetostatic[_sparse]`, `fea_solve_magnetostatic`, and `run_workflow` (via
  `DiscreteModel`). Default `None` = damped-Picard (unchanged).
- **The comparison seed** `tests/integration/matrix/test_iteration_method_matrix.py` — every registered
  method on real MEC cases, Principle-P gated (converge-to-reference or xfail).

The KEY consequence: **most new nonlinear methods are just a new `IterationStrategy` class** added to
`ITERATION_REGISTRY` — the equation (assemble/residual/tangent) and the dispatch do not change.

#### 1. Nonlinear outer iteration — new strategies on the existing seam

Each is a class in `maxwell/nonlinear.py` registered in `ITERATION_REGISTRY`; it drives the iterate
using the `NonlinearSystem` methods it already has. Ordered easiest -> hardest to add:

- **`anderson` (Anderson acceleration).** Wraps the Picard fixed-point map `x <- solve_linear(assemble(x))`
  with an m-history least-squares mixing of past residuals. Uses ONLY `assemble` + `solve_linear`
  (no tangent), so it drops onto both MEC and FEA immediately. High value: accelerates the robust
  Picard where Newton is fragile. VERIFY: converges to the reference field in fewer iters than plain
  Picard on the iteration matrix.
- **`modified-newton`.** Newton but reuse the cached `tangent` for `k` steps before re-assembling it
  (a strategy-level cache). Uses `tangent`/`residual`/`solve_system` (have them). Cheaper per step.
- **`quasi-newton` (Broyden / BFGS / L-BFGS).** Build an approximate Jacobian from `residual` history
  (low-rank updates); needs `residual` + a linear apply. Positions between Picard and Newton; useful
  where the exact tangent is unavailable (a combo whose `tangent` returns `None`).
- **`newton-linesearch` / damped-Newton with line search.** Newton step scaled by `λ` from a
  backtracking line search on `||residual||`; needs `residual` (have it). The robust globaliser for
  deep saturation (today's `newton` takes the full step).
- **`continuation` (load / source stepping).** An OUTER driver that scales the source
  (`conductor_current_density_list` and magnet `Br`) through `0<λ₁<...<λₙ=1`, each sub-solve seeded by
  the previous solution, each solved by an inner strategy. Needs a source-scaling hook at the
  `DiscreteModel`/`solve` boundary (the current is already a per-solve argument, so a continuation
  driver scales it). The rescuer for strong-saturation / large-current motor operating points.
- **`arc-length`.** Simultaneously controls the solution increment and the load parameter
  (`Δaᵀ Δa + β Δλ² = Δs²`); tracks limit points / bifurcations. A specialized driver (mostly relevant
  to coupled magneto-mechanical / multistable problems) — lower priority for magnetostatics.
- **`newton-krylov` / JFNK.** Newton whose LINEAR system is solved by a Krylov iterative solver
  (GMRES/BiCGSTAB), optionally Jacobian-free via `J v ≈ (R(a+εv)-R(a))/ε` (needs only `residual`).
  REQUIRES the linear-solver seam (§2). The path to large 3D / eddy-current systems.

#### 2. Linear inner solve — a NEW pluggable seam (today it is direct `spsolve`)

Currently every solve is a DIRECT sparse factorization: MEC `_ReluctanceSystem.solve_system` and FEA
`fea/_solve.py::_solve` both call `scipy.sparse.linalg.spsolve`. To support iterative linear solvers
(and JFNK), extract a `LinearSolver` seam:

- **`LINEAR_SOLVER_REGISTRY`** in a new `maxwell/linear_solvers.py`: `solve(matrix, rhs, *, x0=None) ->
  x`, registered `direct` (default = today's spsolve, bit-for-bit) + `cg`, `minres`, `gmres`,
  `bicgstab`, `qmr` (scipy.sparse.linalg). Route `solve_system` / `_solve` through it.
- **Preconditioners** as a peer registry: `jacobi`, `ilu` (spilu), `icc`, `amg` (pyamg, optional dep),
  block, and the Maxwell-specific `hiptmair-xu` / auxiliary-space for curl-curl (edge-element future).
- The magnetostatic MEC/FEA systems are SPD-ish (CG fits); eddy-current (complex, non-symmetric) needs
  GMRES/BiCGSTAB; note the curl-curl null space + gauge for a future edge-element formulation.
- VERIFY: each linear solver reaches the same solution as `direct` within its tolerance on a real
  case (the same "converge-to-reference" contract as the nonlinear matrix).

#### 3. Beyond magnetostatics (new outer layers, orthogonal to §1/§2)

- **Time-stepping (eddy currents / transient).** `∂A/∂t` term -> Backward-Euler / Crank-Nicolson / BDF
  / Runge-Kutta drivers; each step is a (non)linear solve using §1/§2. A new transient module; the
  spatial assembly is reused.
- **Multiphysics coupling.** Monolithic vs staggered / block-Gauss-Seidel / block-Jacobi /
  partitioned, with Aitken / Anderson relaxation on the partitioned fixed point (magneto-thermal,
  magneto-mechanical). Orthogonal to the single-physics iteration.
- **Stabilisation strategies** (cross-cutting): damping, line search, trust region, load stepping —
  several already exist inside strategies; factor them into reusable globalisers.

#### 4. The comparison experiment (Principle P, permanent)

Extend `test_iteration_method_matrix.py` into a **method benchmark**: for every
`(case × equation × outer-method × inner-solver × preconditioner)`, record iters-to-tol, final
residual, and wall-clock; assert convergence-to-reference or `xfail(strict=False)` a known
non-convergence (e.g. plain Picard on a case where it diverges). Register each method's measured
convergence character in `maxwell/accuracy.py` (a new `iteration` metric band), so a regression in a
method's convergence is caught the same way a field regression is. This is the "compare the various iteration methods" (user, translated)
artifact — run permanently, never a one-off.

#### 5. Suggested order (value × safety)

1. `anderson` (pure add, both cores, big robustness win) + benchmark it.
2. The `LINEAR_SOLVER_REGISTRY` seam (`direct` default bit-for-bit; add `cg`/`gmres`) — unlocks JFNK.
3. `modified-newton`, `newton-linesearch`, `quasi-newton` (Broyden).
4. `continuation` driver (source stepping) — the saturation rescuer.
5. `newton-krylov` / JFNK on the new linear seam.
6. Transient (time-stepping) + coupling layers (larger, separate efforts).

Every step: new strategy/seam, default unchanged (bit-for-bit), verified converge-to-reference,
registered + matrix-gated. None excluded — a slower/worse method is a labelled datum, not a drop.
