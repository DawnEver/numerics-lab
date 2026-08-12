---
module: maxwell
severity: MEDIUM
status: open
created: 2026-07-09
accessed: 2026-07-09
tags: [design, refactor, architecture, solver, mesh-decouple, iteration, newton-raphson, picard, principle-p]
---

# DESIGN — decouple mesh / governing-equation / nonlinear-iteration; plug in every iteration method

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

## 0. Answers to the three questions (grounded, before the design)

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

## 1. Where the coupling actually is today (file:line)

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

## 2. Target architecture — three independent seams

```
DiscreteModel  ──►  GoverningEquation  ──►  IterationStrategy  ──►  SolveResult
(mesh+materials     (assembles residual     (drives A_{k+1} from     (fields+energy
 +sources, one       R(A) and, optionally,   R and J; Picard /        +accuracy label)
 value object)       tangent J(A))           Newton / damped)
```

### Seam 1 — Mesh ↔ solve: a `DiscreteModel` value object
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

### Seam 2 — Governing equation ↔ iteration: a `NonlinearSystem` interface
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

### Seam 3 — the pluggable `IterationStrategy` (the heart of the directive)
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

## 3. Dispatch — how a solve names its (equation × mesh × iteration)

The variant label grows one axis. Today: `solver:mesh` (`mec:hybrid`). Add iteration as an optional
suffix, defaulting to today's behaviour so nothing changes unless asked:
`mec:hybrid` == `mec:hybrid/picard-damped` (default); `mec:hybrid/newton`, `mec:tri/picard`, etc.
`solve()` parses `(equation, mesh, iteration)`, builds the `DiscreteModel` (seam 1), wraps it in the
`NonlinearSystem` (seam 2), and hands it to `ITERATION_REGISTRY[iteration]` (seam 3). The
allowlist in `compare_torque_waveforms` / `_derive_solver_kwargs` extends the same way.

## 4. Newton-Raphson for magnetostatics — the concrete Jacobian (what's missing)

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

## 5. The freeze boundary (invariant.md) — migration order

- **MEC is NOT frozen** → land all three seams on MEC first. `mec/_sparse.py` loop → extract
  `DampedPicardStrategy`; add `PicardStrategy` + `NewtonStrategy`; assert the extracted damped path
  reproduces today's energy iterates (characterization/golden tests bit-for-bit).
- **FEA + SDM solving logic ARE frozen** → the frozen `fea_solver.py` damped-Picard path stays
  byte-for-byte. Newton/plain-Picard for FEA is an ADDITIVE parallel path (exactly how the native Q4
  kernel `fea/_assemble_quad.py` was added), dispatched by the iteration suffix, and needs the same
  explicit in-the-moment approval the Q4 addition got. Default dispatch keeps the frozen path.

## 6. Principle-P wiring — every method characterized, none excluded

Each new `(equation × mesh × iteration)` combo is:
1. registered in `maxwell/accuracy.py` with a measured `RegistryEntry` (convergence + torque/energy
   band): e.g. `('mec:hybrid/newton', '*')` validated if it lands, `('mec:*/picard', '*')` labelled
   `degraded` (oscillatory) — the plain-Picard divergence becomes an HONEST label, not a silent drop;
2. added to the discovery matrices — the same shape as the just-shipped
   `tests/integration/test_torque_full_matrix.py`: connect the combo, `xfail(strict=…)` a known miss
   (e.g. plain-Picard non-convergence → `xfail(strict=False)` until/unless it is fixed), never skip;
3. a convergence-characterization test (iters-to-tol, final residual) per strategy, tolerances not
   hashes. A method proven worse is KEPT and labelled, so the comparison across methods (the
   directive's "对比各种迭代方法") is a first-class, permanently-run artifact, not a one-off experiment.

## 7. First concrete steps (smallest safe increments)

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
