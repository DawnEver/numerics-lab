---
module: maxwell
severity: MEDIUM
status: open
created: 2026-07-09
accessed: 2026-07-09
tags: [research-frontier, hiptmair-xu, auxiliary-space, ams, curl-curl, edge-element, nedelec, preconditioner, linear-solver-registry, ws-research]
---
> **MIGRATED 2026-08-01** → `.claude/memory/2026/08/01/plan-refactor-unified-backlog.md`. All remaining work (if any) is consolidated there; this file is the archived original design record. **CLOSED 2026-08-01 -- deferred (research -> 0.6)**


# PLAN (research) — Hiptmair-Xu / auxiliary-space preconditioner for Maxwell curl-curl (edge elements)

> WS-Research plan 7 of 9. The AMS (auxiliary-space Maxwell) preconditioner for the curl-curl operator
> that arises with EDGE (Nedelec) elements. Slots into the EXISTING `PRECONDITIONER_REGISTRY`. The
> curl-curl operator + edge-element kernel are the greenfield prereq; the preconditioner seam exists.
> Fills the "curl-curl / eddy-current / 3D vector" selection-table row (plan 9).

## 1. Motivation

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

## 2. What exists in-repo to build on

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

## 3. Approach mapped to the 3-layer seams

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

## 4. Files to touch / create

- NEW `maxwell/fea/_assemble_edge.py` — Whitney edge elements + curl-curl (+ mass) assembly (the
  greenfield prereq). Emits the edge system + the discrete gradient `G`.
- `linear_solvers.py` — `_ams_preconditioner(matrix, aux)` + `PRECONDITIONER_REGISTRY['ams']`; extend
  `make_preconditioner` to accept the auxiliary operators (G, coords) for AMS.
- `pyproject.toml` — add `pyamg` as an OPTIONAL extra (the nodal AMG for AMS's auxiliary solves).
- `maxwell/accuracy.py` — a curl-curl convergence band; the benchmark gains an AMS row.
- Tests: `tests/unit/maxwell/test_assemble_edge.py` + `test_ams_preconditioner_converges.py`.

## 5. Verification strategy (converge-to-reference)

- **Edge-element correctness:** a curl-conforming patch test (a constant curl field reproduced exactly
  on an edge mesh; tangential continuity across edges). The defining Nedelec property.
- **AMS convergence (decisive):** on a curl-curl system, GMRES+AMS must converge in a MESH-INDEPENDENT
  number of iterations, while GMRES+ILU/jacobi STALLS (iteration count blows up with mesh size). That
  gap IS the value; assert it in `test_solver_method_benchmark`'s pattern (iters-to-tol vs mesh).
- **Reduces to reference:** the AMS-preconditioned solve reaches the SAME field as a direct solve of
  the edge system within tolerance (the registry's converge-to-reference contract).

## 6. Priority tier + effort

- **Tier: MID (the preconditioner seam) / LONG-TERM (the edge-element prereq).** The AMS entry itself is
  ~1 wk given the built registry. But it needs edge elements, which need a curl-curl formulation, which
  is only USED once eddy-current/time-harmonic/3D lands (`investigate-time-harmonic.md`). So: high
  strategic value (unlocks the entire AC/3D roadmap), but gated behind a large greenfield element +
  formulation, and not on any current case's critical path.
- Effort: edge-element 2D curl-curl ~3-4 wk; AMS preconditioner ~1-1.5 wk; 3D + eddy mass term later.

## 7. Dependencies / prereqs

- **HARD PREREQ: edge (Nedelec) elements** — build the curl-curl formulation FIRST; AMS is meaningless
  without it. This is the single biggest cross-plan dependency on the frontier.
- Ties to plan 8 (nodal vs loop/edge method characterization) — edge elements are the "棱边法" that plan
  8's comparison contrasts with the nodal method.
- Depends on the built `linear_solvers` registry (done) + `pyamg` (new optional dep).
- Independent of WS-Validate (different physics regime).
