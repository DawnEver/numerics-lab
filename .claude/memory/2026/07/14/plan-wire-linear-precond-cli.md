---
module: plan
created: 2026-07-14
priority: high
tags: [cli, linear-solver, preconditioner, sync-gate, gap]
---

# Plan — Wire --linear-solver and --preconditioner to CLI

**Priority: HIGH.** Two registries are implemented, test-validated, and reachable programmatically
but NOT exposed at the command line — users cannot select Krylov backends or preconditioners from
`uv run motronics solve`.

## Gap

| Registry | Module | CLI exposed? | Sync gate test? |
|---|---|---|---|
| `ITERATION_REGISTRY` | `gauss/nonlinear/__init__.py` | `--iteration-strategy` with `choices=sorted(ITERATION_REGISTRY)` | n/a (dynamic `choices` + help-string text) |
| `LINEAR_SOLVER_REGISTRY` | `gauss/linear/__init__.py` | **NO** | **NO** |
| `PRECONDITIONER_REGISTRY` | `gauss/linear/__init__.py` | **NO** | **NO** |

`preconditioner` is referenced in the sweep cache key (`__main__.py` :2414) as
`getattr(args, 'preconditioner', None)` — a dead read that always falls back to `None` because the
argument is never declared.

## What to do

1. Add `--linear-solver` to the `run` / `case run` subparsers with `choices=sorted(LINEAR_SOLVER_REGISTRY)`.
2. Add `--preconditioner` to the same subparsers with `choices=sorted(PRECONDITIONER_REGISTRY)`.
3. Thread both through `run_workflow` → the solver call (already supported at the
   `NonlinearSystem` / `GalerkinSystem` level — the seam exists, just the CLI knob is missing).
4. Add registry-parametrized sync gate tests (same pattern as `test_every_base_solver_is_cli_reachable`):
   - `test_every_linear_solver_is_cli_reachable` — parametrize over `LINEAR_SOLVER_REGISTRY`
   - `test_every_preconditioner_is_cli_reachable` — parametrize over `PRECONDITIONER_REGISTRY`
5. Remove the dead `getattr(args, 'preconditioner', None)` fallback — it should be a real argparse dest.

## Pattern to follow

Existing sync gate: `tests/unit/workflows/test_cli_sync_gate.py` parametrizes over
`SOLVER_SPECS` / `CANONICAL_VARIANTS` / `known_variants()` and asserts each is reachable
via `-s`. Same pattern for the linear/preconditioner registries. Existing CLI pattern:
`choices=sorted(ITERATION_REGISTRY)` on `--iteration-strategy` — the dynamic `choices` list auto-syncs
the `--help` text without a separate test.
