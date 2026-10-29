# numerics-lab

Equation-solving numerics for the family: linear and nonlinear solves, time integration (Python + Rust).

## How a change lands

The family flow, stated once in [the integration branch](https://github.com/DawnEver/lab-commons/blob/integrate/main/docs-src/dev/integration.md):

- Cut `pr/<topic>` from `origin/integrate/main`, measure what you touched, commit, push, open a pull request.
- A producer claims the pushed head and posts `tests/<part>`; the FORGE merges the PR once every part
  stands PASS, the head merges cleanly and the merge passes the audit.
- `main` moves only by `integrate promote <sha>` on a full PASS.
- `ephemeral = ['pr/*']` and `pr_merges_since` are declared in `pyproject.toml`, and
  `tests/architecture/test_this_repo_declares_the_one_pr_flow.py` refuses this repo without them.

The mechanisms are lab-commons's (`docs-src/dev/`); this repo declares the names and runs the assertions.
