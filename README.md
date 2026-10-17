# numerics-lab

Equation-solving numerics, imported as `numerics_lab`: sparse linear solves (direct, Krylov,
preconditioners, refinement), constrained linear systems, nonlinear iteration strategies, DAE
initial conditions and time integration. Python with a Rust core (`crates/numerics`, exposed as
`numerics_lab._native`).

Runtime dependencies are numpy and scipy only. Extracted from `motronics.gauss` with history.

- Install for development: `uv venv` then `uv pip install -e ".[dev]"` (builds the Rust extension).
- Python tests: `.venv/Scripts/python.exe -m lab_commons.dev.verify -- tests`.
- Rust tests: `cargo test --workspace`.
