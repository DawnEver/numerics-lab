---
name: hiptmair-xu-ams-preconditioner
description: "WS-Research plan 7 (AMS for curl-curl with Nedelec edge elements) and the 2026-07-10 close-out: not applicable to the 2D scalar-A formulation; defer until a 3D or H(curl) solver exists."
created: 2026-07-09
accessed: 2026-09-27
merged_from: .claude/memory/2026/07/10/finding-hiptmair-xu-assessment.md, .claude/memory/2026/07/09/plan-research-hiptmair-xu-precond.md
metadata:
  type: project
---

# Hiptmair-Xu / AMS preconditioner: research plan and applicability close-out

## Current conclusion

- NOT APPLICABLE to the current 2D scalar-A formulation; DEFER until a 3D or eddy-current / time-harmonic H(curl) edge-element solver exists. The plan is CLOSED 2026-08-01 as deferred research.

## Members, newest first (full text; the older ones are the History above refers to)

### Member `.claude/memory/2026/07/10/finding-hiptmair-xu-assessment.md`

Original frontmatter:

```yaml
module: maxwell
severity: LOW
status: closed
created: 2026-07-10
accessed: 2026-07-10
tags: [research-frontier, hiptmair-xu, ams, auxiliary-space, curl-curl, edge-element, nedelec, preconditioner, applicability, defer, playground]
```

### FINDING — Hiptmair-Xu / AMS applicability assessment (close-out of `plan-research-hiptmair-xu-precond.md`)

> Honest close-out of the overnight-flagged item. Verdict up front: **NOT APPLICABLE
> to the current 2D scalar-A formulation; DEFER until a 3D or eddy-current / time-harmonic
> H(curl) edge-element solver exists.** The plan already said this (its own §1, §6, §7);
> this finding backs it with direct evidence from the assembled matrix and a tiny prototype.

#### Verdict

Hiptmair-Xu AMS preconditions the **curl-curl** operator `curl(nu curl A) [+ j w sigma A]`
discretized with **edge (Nedelec/Whitney) elements** — an H(curl) system whose huge gradient
near-null-space stalls plain CG/ILU/AMG. **No such system exists in this codebase, and none is
on any current case's critical path.** AMS therefore has nothing to precondition today.

#### Evidence — the actual system is 2D scalar A_z nodal SPD

- **Formulation is scalar nodal, not H(curl).** `fea/_assemble.py:161` computes
  `B = curl(A z_hat) = (dA/dy, -dA/dx)` from a **scalar** node potential; DOFs are nodes
  (`T_mat = np.zeros(n_vert)`, `fea/_assemble.py:141`); the element matrix is a 3×3 nodal
  `-div(nu grad A)` Laplacian (`fea/_assemble.py:208-223`). This is `-div(nu grad A) = J`,
  SPD, Poisson-type — exactly what the plan's own §1 states.
- **Solve is a sparse direct LU by default.** `fea/_solve.py:400-401` → `solve_linear_system`,
  whose default backend is `spsolve` (`linear_solvers.py:1-19`). Krylov peers exist
  (cg/minres for SPD, gmres/bicgstab/qmr) with jacobi/ILU preconditioners only.
- **No H(curl) code anywhere.** Grep across `src/` for
  `nedelec|edge_element|whitney|curl_curl|H(curl)|discrete gradient|j*omega|omega*sigma`
  → zero hits. The only "eddy" / "time-harmonic" strings are **forward-looking comments**
  (`linear_solvers.py:7,136`) and a guard that *rejects* an imaginary flux linkage from FEMM
  (`adapters/femm/solve.py:113-117`). There is no complex / frequency-domain assembly.
- **The matrix is well-conditioned (probe `playground/hiptmair_xu_probe.py`, Part A),**
  assembled from the production `_precompute_geometry` (nu=1):

  | nodes  | nnz     | symmetric | min eig  | cond ~   | spsolve (s) |
  |--------|---------|-----------|----------|----------|-------------|
  | 256    | 1 432   | True      | 8.74e-02 | 9.05e+01 | 0.0005      |
  | 1 024  | 6 424   | True      | 2.05e-02 | 3.89e+02 | 0.0015      |
  | 4 096  | 27 160  | True      | 4.97e-03 | 1.61e+03 | 0.0077      |
  | 16 384 | 111 640 | True      | 1.22e-03 | 6.54e+03 | 0.0381      |

  Symmetric, SPD, cond grows only ~O(h⁻²) (the mild Poisson rate). Direct LU solves 16k DOF
  in 38 ms; plain CG+ILU converges. **No stalling near-null-space → no need for AMS.**

#### Smallest-prototype result — the AMS *core* ingredient IS buildable here

`playground/hiptmair_xu_probe.py` Part B, as a proof-of-concept only (no solver):
- From the repo's existing triangular mesh it builds the **discrete gradient `G`**
  (node→edge signed incidence, 85 edges × 36 nodes) and a discrete curl `C`.
- Verifies the defining identity AMS exploits: `||curl(G · phi)||_inf = 4.4e-16 ≈ 0`
  (discrete `curl · grad = 0`), i.e. a nodal scalar lifted to an edge gradient has zero curl —
  the null-space property the AMS correction is built on.

**Takeaway:** the single most important auxiliary-space object (`G`) is trivially
constructible from the mesh already produced here. So *if* the H(curl) roadmap ever lands,
AMS is not blocked by mesh/topology infrastructure — it is blocked by the missing physics.

#### What it would take IF pursued (unchanged from the plan's estimate)

The topic (AMS entry) is small; its **hard prereq is a whole new formulation**:
1. **Greenfield curl-curl EDGE-element assembly** (`maxwell/fea/_assemble_edge.py`): lowest-order
   Whitney edge elements, DOFs on edges, `curl(nu curl)` (+ mass term for eddy). A genuine new
   peer of the nodal `GalerkinSystem`. Plan estimate **~3-4 wk (2D)**, plus 3D + eddy mass later.
2. **AMS preconditioner** (`linear_solvers.py`): `PRECONDITIONER_REGISTRY['ams']` built from
   `(matrix, G, node_coords)`; inner nodal AMG V-cycles via **pyamg** (new optional dep).
   Plan estimate **~1-1.5 wk** — the `make_preconditioner` seam already fits it.
3. A curl-curl consumer: eddy-current / time-harmonic / 3D physics (`investigate-time-harmonic.md`,
   `plan-multiphysics-hybrid.md`). **None exists**, so even a finished AMS would have no caller.

The `G` proof-of-concept above collapses only a small slice of step 2's plumbing; steps 1 and 3
(≈90% of the effort and ALL of the strategic prerequisite) remain untouched, correctly outside
playground scope.

#### Recommendation

**DEFER.** Do not build edge elements or AMS now — there is no H(curl) system to precondition
and no case needs one. Re-open this plan *only when* a curl-curl formulation is scheduled
(the eddy-current / time-harmonic / 3D roadmap). At that point AMS becomes the enabling solver
and the `G` ingredient shown here plugs straight in. Until then this item is **closed as
not-applicable**, exactly as the overnight handoff flagged and the plan's own priority tier
(MID topic / LONG-TERM prereq) already recorded.

#### Artifacts
- Probe: `playground/hiptmair_xu_probe.py` (Part A conditioning table + Part B discrete-gradient PoC).
- Source plan: `.claude/memory/2026/07/09/plan-research-hiptmair-xu-precond.md` (leave `status: open`
  as a strategic roadmap item; this finding records it is intentionally deferred, not abandoned).

### Member `.claude/memory/2026/07/09/plan-research-hiptmair-xu-precond.md`

Original frontmatter:

```yaml
module: maxwell
severity: MEDIUM
status: open
created: 2026-07-09
accessed: 2026-07-09
tags: [research-frontier, hiptmair-xu, auxiliary-space, ams, curl-curl, edge-element, nedelec, preconditioner, linear-solver-registry, ws-research]
```

> **MIGRATED 2026-08-01** → `.claude/memory/2026/08/01/plan-refactor-unified-backlog.md`. All remaining work (if any) is consolidated there; this file is the archived original design record. **CLOSED 2026-08-01 -- deferred (research -> 0.6)**


### PLAN (research) — Hiptmair-Xu / auxiliary-space preconditioner for Maxwell curl-curl (edge elements)

> WS-Research plan 7 of 9. The AMS (auxiliary-space Maxwell) preconditioner for the curl-curl operator
> that arises with EDGE (Nedelec) elements. Slots into the EXISTING `PRECONDITIONER_REGISTRY`. The
> curl-curl operator + edge-element kernel are the greenfield prereq; the preconditioner seam exists.
> Fills the "curl-curl / eddy-current / 3D vector" selection-table row (plan 9).

#### 1. Motivation

The current scalar magnetostatic formulation (`-div(nu grad A) = J`, scalar A_z in 2D) is SPD and CG/
ILU solve it fine (the built linear-solver registry). But eddy-current, time-harmonic, and 3D vector-
potential problems need the **curl-curl operator** `curl(nu curl A) + j*omega*sigma A = J`, discretized
with **edge (Nedelec) elements** to avoid spurious modes. That operator has a LARGE near-null-space
(all gradients, `curl grad phi = 0`), so plain CG/ILU/AMG STALL on it — the classic hard problem of
computational electromagnetics. The **Hiptmair-Xu auxiliary-space preconditioner (AMS)** is THE
standard solver: it maps the curl-curl problem to auxiliary nodal (H1) Poisson problems (one vector, one
scalar) that AMG solves well, recombining them to precondition the edge system. This is the enabling
technology for the whole eddy-current / 3D roadmap (`plan-multiphysics-hybrid.md`: "时谐场", "3D",
"棱边法/edge element", "Krylov 子空间 CG/GMRES", "FEM-DDM"). Without it, iterative solves of the vector
problem do not converge.

#### 2. What exists in-repo to build on

- **`linear_solvers.py`** — `LINEAR_SOLVER_REGISTRY` (direct/cg/minres/gmres/bicgstab/qmr),
  `PRECONDITIONER_REGISTRY` (jacobi/ilu), `make_preconditioner(name, matrix) -> LinearOperator`, and
  `solve_linear_system(matrix, rhs, *, linear_solver, preconditioner, x0)`. **The AMS preconditioner is
  a NEW `PRECONDITIONER_REGISTRY` entry** — the seam is EXACTLY built for this.
- **`NewtonKrylovStrategy`** (`nonlinear.py`) — JFNK already routes the inner solve through the linear
  registry, so an edge-element nonlinear solve inherits AMS the moment it is registered.
- **NO curl-curl / edge-element / Nedelec code exists** (grep confirmed: no `nedelec`/`edge_element`/
  `whitney`/`curl_curl` anywhere) — the operator + element are greenfield (the real cost of this plan).
- **`accuracy.py`** + Principle-P benchmark (`test_solver_method_benchmark.py` already gates
  preconditioners: `test_linear_solver_preconditioner_reaches_reference`) — AMS drops into that harness.

#### 3. Approach mapped to the 3-layer seams

The plan has TWO parts; the preconditioner (the named topic) is small, its PREREQ (edge elements) is
the large greenfield piece.

- **PREREQ — LAYER 2 edge-element curl-curl assembly** (`maxwell/fea/_assemble_edge.py`): lowest-order
  Nedelec (Whitney) edge elements; DOFs on edges, not nodes; assemble `curl(nu curl)` (+ mass term for
  eddy). This is a genuinely new formulation (a peer of `GalerkinSystem`), needed before AMS has
  anything to precondition. 2D first (edges of the existing tri mesh), then 3D.
- **The TOPIC — LAYER 2/3 seam, AMS preconditioner** (`linear_solvers.py`): AMS needs three auxiliary
  operators from the mesh — the discrete GRADIENT `G` (node->edge incidence, trivial from mesh
  connectivity), and the nodal coordinate interpolations `Pi`. AMS applies: smooth on the edge system,
  correct via `G * (nodal-Poisson-solve) * G^T`, correct via the vector nodal Poisson. Register as
  `PRECONDITIONER_REGISTRY['ams']`, built from `(matrix, G, node_coords)`. Use pyamg (optional dep) for
  the inner nodal AMG V-cycles.
- **LAYER 3:** unchanged — `NewtonKrylovStrategy` + `solve_linear_system(preconditioner='ams')`.

#### 4. Files to touch / create

- NEW `maxwell/fea/_assemble_edge.py` — Whitney edge elements + curl-curl (+ mass) assembly (the
  greenfield prereq). Emits the edge system + the discrete gradient `G`.
- `linear_solvers.py` — `_ams_preconditioner(matrix, aux)` + `PRECONDITIONER_REGISTRY['ams']`; extend
  `make_preconditioner` to accept the auxiliary operators (G, coords) for AMS.
- `pyproject.toml` — add `pyamg` as an OPTIONAL extra (the nodal AMG for AMS's auxiliary solves).
- `maxwell/accuracy.py` — a curl-curl convergence band; the benchmark gains an AMS row.
- Tests: `tests/unit/maxwell/test_assemble_edge.py` + `test_ams_preconditioner_converges.py`.

#### 5. Verification strategy (converge-to-reference)

- **Edge-element correctness:** a curl-conforming patch test (a constant curl field reproduced exactly
  on an edge mesh; tangential continuity across edges). The defining Nedelec property.
- **AMS convergence (decisive):** on a curl-curl system, GMRES+AMS must converge in a MESH-INDEPENDENT
  number of iterations, while GMRES+ILU/jacobi STALLS (iteration count blows up with mesh size). That
  gap IS the value; assert it in `test_solver_method_benchmark`'s pattern (iters-to-tol vs mesh).
- **Reduces to reference:** the AMS-preconditioned solve reaches the SAME field as a direct solve of
  the edge system within tolerance (the registry's converge-to-reference contract).

#### 6. Priority tier + effort

- **Tier: MID (the preconditioner seam) / LONG-TERM (the edge-element prereq).** The AMS entry itself is
  ~1 wk given the built registry. But it needs edge elements, which need a curl-curl formulation, which
  is only USED once eddy-current/time-harmonic/3D lands (`investigate-time-harmonic.md`). So: high
  strategic value (unlocks the entire AC/3D roadmap), but gated behind a large greenfield element +
  formulation, and not on any current case's critical path.
- Effort: edge-element 2D curl-curl ~3-4 wk; AMS preconditioner ~1-1.5 wk; 3D + eddy mass term later.

#### 7. Dependencies / prereqs

- **HARD PREREQ: edge (Nedelec) elements** — build the curl-curl formulation FIRST; AMS is meaningless
  without it. This is the single biggest cross-plan dependency on the frontier.
- Ties to plan 8 (nodal vs loop/edge method characterization) — edge elements are the "棱边法" that plan
  8's comparison contrasts with the nodal method.
- Depends on the built `linear_solvers` registry (done) + `pyamg` (new optional dep).
- Independent of WS-Validate (different physics regime).
