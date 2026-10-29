# numerics-lab

Equation-solving numerics, imported as `numerics_lab`: sparse linear solves (direct, Krylov,
preconditioners, refinement), constrained linear systems, nonlinear iteration strategies, DAE
initial conditions and time integration. Python with a Rust core (`crates/numerics`, exposed as
`numerics_lab._native`).

Runtime dependencies are numpy and scipy only. Extracted with history from a downstream solver package.

- Install for development: `uv venv` then `uv pip install -e ".[dev]"` (builds the Rust extension).
- Python tests: `.venv/Scripts/python.exe -m dev_lab.dev.verify -- tests`.
- Rust tests: `cargo test --workspace`.

## Mounting in a product

A product mounts this engine as the component `numerics-lab` (path derived as
`components/numerics-lab`), providing the Python distribution `numerics-lab` and the Cargo crate
`numerics` at `crates/numerics`:

    [tool.dev_lab.components.numerics-lab]
    url = "<the engine's repository>"
    branch = "main"
    parts = [...]
    package = "numerics-lab"
    crates = { numerics = "crates/numerics" }
