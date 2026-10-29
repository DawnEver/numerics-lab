# numerics-lab

Equation-solving numerics, imported as `numerics_lab`: sparse linear solves (direct, Krylov,
preconditioners, refinement), constrained linear systems, nonlinear iteration strategies, DAE
initial conditions and time integration. Python with a Rust core (`crates/numerics`, exposed as
`numerics_lab._native`).

Runtime dependencies are numpy and scipy only. MIT licensed.

## Using it

A consumer declares it as a plain git dependency; building it needs a Rust toolchain (maturin is the
build backend):

    dependencies = ["numerics-lab @ git+https://github.com/DawnEver/numerics-lab.git"]

The Rust crate is consumed the same way, as a Cargo git dependency on `crates/numerics`.

## Developing it

- Install: `uv venv`, then `uv pip install -e ".[dev]"` (builds the Rust extension).
- Python tests: `.venv/Scripts/python.exe -m dev_lab.dev.verify -- tests` (`.venv/bin/python` on POSIX).
- Rust tests: `cargo test --workspace`.
- How a change lands: `AGENTS.md`.
