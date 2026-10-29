---
name: b3-public-api-and-the-clean-ups
description: 2026-10-18 Track B3 of PLAN-engines-and-one-outcome-vocabulary -- the public numerics API optimi-lab P9 consumes (line search, trust region, quasi-Newton, Krylov, jittered Cholesky, KKT, shooting/collocation, Stream), the strategies rewired onto it, scipy.optimize removed, Rust constrained modules renamed, the checkpoint upgrade path deleted, b27f6b6 re-verified
created: 2026-10-18
accessed: 2026-10-18
tags: [numerics-lab, track-b, b3, api, line-search, quasi-newton, trust-region, kkt, retired]
metadata:
  type: project
---

# B3 -- the public API, and the clean-ups

Branch `b3-public-api` (local, no remote). Plan: a downstream engines-and-outcome-vocabulary plan, II.4.

## Where each II.4 item lives

| item | module |
|---|---|
| jittered Cholesky (reports jitter) | `linear/cholesky.py` `jittered_cholesky` |
| dense and sparse KKT | `linear/kkt.py` `solve_kkt` |
| CG, MINRES, Steihaug-CG | `linear/krylov.py` (matrix-free; the registry's `cg`/`minres` backends are NOT rewired: the linear goldens characterise them) |
| Armijo, strong Wolfe, backtracking | `line_search.py`; `directional_line_search` moved here from `time_integrators._step` |
| trust-region update, dogleg | `trust_region.py` |
| BFGS, L-BFGS two-loop, SR1, Broyden | `quasi_newton.py` (matrix-free and dense) |
| shooting/collocation residuals | `time_integrators/shooting.py` `shooting_defects`; `time_integrators/collocation.py` |
| `Stream(seed, path)` | `random_streams.py` |

Rewired: `NewtonStrategy._search`, `finite_storage._backtracked_state`, `_step`'s halving search
(bit-identical step lengths: powers of two), BFGS/L-BFGS/Broyden strategies. The Armijo test is now
`phi <= phi0 + c1 lam slope0` with `slope0 = -||r||`, not `(1 - c lam) ||r||`: identical but for
an ulp at an exact tie. Goldens unchanged and green.

## Findings

- `periodic_shooting` imported `scipy.optimize.newton_krylov` and the R1 guard did not see it (it
  checked top-level `scipy` only). The guard now admits scipy only via linalg/sparse/special, and
  shooting runs its own Jacobian-free Newton-GMRES with Armijo backtracking.
- The bare registry `direct` backend (`get_linear_solver('direct')`) answers a singular symmetric
  KKT with finite garbage in 40 of 300 random draws; only `solve_linear_system` applies the
  residual guard. `solve_kkt` goes through the guarded call (test sweeps the 300 draws). Any other
  caller of the bare backend has the same hole -- NOT fixed here.

## Clean-ups

- Rust `linear::{dirichlet, constraints}` -> `linear::constrained::{fixed_values, pin_rows, master_slave}`.
  The file moves were swept into commit 610571d (staged `git mv`), so 610571d alone does not build;
  fd34d39 completes it.
- Checkpoint: no upgrade table; any envelope but `numerics-lab-transient-checkpoint/v1` is refused
  naming both schemas, "never upgraded".
- Retired-name registry: `tests/architecture/test_retired_names_stay_retired.py` (`RETIRED`).

## Verification

- b27f6b6 re-verified: `lab_commons.dev.verify -- tests` PASS, 1130 passed (tree = b27f6b6 + a
  lab-commons lock bump).
- Tip: verify PASS 1184 passed; `cargo test --workspace` 154 + 6 + 4 green.
