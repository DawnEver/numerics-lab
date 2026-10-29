# Changelog

The version is the Cargo workspace's (`Cargo.toml` `[workspace.package] version`); maturin reads it for the
Python distribution, and a release tag `v<version>` must match it.

## 0.1.0 -- 2026-10-18

First release, extracted with its history from a downstream solver package.

- `numerics_lab`: sparse linear solves (direct, Krylov, preconditioners, refinement), constrained linear
  systems, nonlinear iteration strategies and line searches, DAE initial conditions and time integration.
- `numerics_lab._native`: the Rust core (`crates/numerics`) through pyo3, built by maturin.
- The engine imports only numpy and scipy's `linalg`/`sparse`/`special`; every other algorithm is its own.
- Licensed MIT.
- Onboarded to the family tooling: rendered project files, TOML git hooks, the committed `uv.lock` under
  PIN + FRESHNESS, the branch set (`main` plus `integrate/main`) and the forge part declaration.
