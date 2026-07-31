---
module: maxwell
severity: LOW
status: closed
created: 2026-07-10
accessed: 2026-07-10
tags: [research-frontier, hiptmair-xu, ams, auxiliary-space, curl-curl, edge-element, nedelec, preconditioner, applicability, defer, playground]
---

# FINDING — Hiptmair-Xu / AMS applicability assessment (close-out of `plan-research-hiptmair-xu-precond.md`)

> Honest close-out of the overnight-flagged item. Verdict up front: **NOT APPLICABLE
> to the current 2D scalar-A formulation; DEFER until a 3D or eddy-current / time-harmonic
> H(curl) edge-element solver exists.** The plan already said this (its own §1, §6, §7);
> this finding backs it with direct evidence from the assembled matrix and a tiny prototype.

## Verdict

Hiptmair-Xu AMS preconditions the **curl-curl** operator `curl(nu curl A) [+ j w sigma A]`
discretized with **edge (Nedelec/Whitney) elements** — an H(curl) system whose huge gradient
near-null-space stalls plain CG/ILU/AMG. **No such system exists in this codebase, and none is
on any current case's critical path.** AMS therefore has nothing to precondition today.

## Evidence — the actual system is 2D scalar A_z nodal SPD

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

## Smallest-prototype result — the AMS *core* ingredient IS buildable here

`playground/hiptmair_xu_probe.py` Part B, as a proof-of-concept only (no solver):
- From the repo's existing triangular mesh it builds the **discrete gradient `G`**
  (node→edge signed incidence, 85 edges × 36 nodes) and a discrete curl `C`.
- Verifies the defining identity AMS exploits: `||curl(G · phi)||_inf = 4.4e-16 ≈ 0`
  (discrete `curl · grad = 0`), i.e. a nodal scalar lifted to an edge gradient has zero curl —
  the null-space property the AMS correction is built on.

**Takeaway:** the single most important auxiliary-space object (`G`) is trivially
constructible from the mesh already produced here. So *if* the H(curl) roadmap ever lands,
AMS is not blocked by mesh/topology infrastructure — it is blocked by the missing physics.

## What it would take IF pursued (unchanged from the plan's estimate)

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

## Recommendation

**DEFER.** Do not build edge elements or AMS now — there is no H(curl) system to precondition
and no case needs one. Re-open this plan *only when* a curl-curl formulation is scheduled
(the eddy-current / time-harmonic / 3D roadmap). At that point AMS becomes the enabling solver
and the `G` ingredient shown here plugs straight in. Until then this item is **closed as
not-applicable**, exactly as the overnight handoff flagged and the plan's own priority tier
(MID topic / LONG-TERM prereq) already recorded.

## Artifacts
- Probe: `playground/hiptmair_xu_probe.py` (Part A conditioning table + Part B discrete-gradient PoC).
- Source plan: `src/motronics/gauss/.claude/memory/2026/07/09/plan-research-hiptmair-xu-precond.md` (leave `status: open`
  as a strategic roadmap item; this finding records it is intentionally deferred, not abandoned).
