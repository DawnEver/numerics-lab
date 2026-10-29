---
name: sparse-initial-storage-keeps-original-coordinate-authority
description: Remove redundant approximate discovery-basis equations from the full sparse initial-pair solve; preserve original storage and state/rate witnesses. Fifty component controls pass; dependency publication remains pending.
created: 2026-10-23
accessed: 2026-10-23
tags: [finite-storage, initial-pair, sparse, exact-authority, dependency-baseline]
metadata:
  type: project
---

# Original physical equations own the sparse initial correction

Latest source checkpoint: the PR branch merged current `origin/integrate/main` (`cf2c321`) through
`dev_lab.dev.forge sync`, preserving the initial-state fix. Configured verification on that actual
synced source passes 72 controls (50 numerical/lifecycle and 22 upstream architecture controls),
environment `8d1ee944e34f4d2b`; the resolver only normalized the lock header revision. This is not a
main-project or machine-accuracy verdict, and publication is verified separately from this record.

## Defect and minimal correction

A downstream field-owned zero-inductance network regression exposed a false initial-pair refusal:
`joint stored state contradicts actual state/rate equations at exact homogeneous anchors`.
Coordinates were field potential, voltage, source current, winding current and linkage. The actual
equations admit state `[0.4, 0, -0.24, 0.24, 0.4]`, with field/linkage rates `-0.48`, retaining the
original finite storage exactly.

`certified_null_directions` discovered the circulating-current state freedom using an SVD basis.
Its opposite entries were `0.7071067811865475` and `-0.7071067811865476`. Passing that discovery
basis as `state_directions` appended equations which the downstream exact homogeneous-anchor
check correctly treated as exact; combined with the original current-balance row, those unequal
normalization coefficients falsely pinned the circulating current to zero.

In `python/numerics_lab/nonlinear/finite_storage.py`, the full sparse call to
`joint_coordinate_sparse_correction` now omits only the redundant `state_directions` argument.
Its original residual/algebraic rows, original stored-coordinate correction witnesses and
differentiated physical rows already define the allowed correction range. No SVD threshold,
anchor check, residual bar or physical equation was weakened. The generic explicitly owned
`state_directions` API remains unchanged for its other callers; the dense branch is unchanged.

Regression: `tests/unit/test_initial_circulating_current_keeps_exact_coordinates.py` expresses
the five-coordinate system directly in engine J/M/Q data, with no Hamilton dependency. It checks
the actual state/rate, original residual closure and exact retained storage.

## Coherent source and environment

The parent checkout originally pinned component `e623f3c`, but its running wheel already exposed
the later `released_on_exit` API. A guarded reinstall of that old source surfaced the mismatch as
an import failure. The component was locally fast-forwarded to its direct descendant `874811b`
(`fix(linear): name faer OutOfMemory and hold one direct factor kind at a time`), restoring the
actual intended memory-lifetime feature, its Rust allocation guards and its tests. This was not
replaced with a compatibility shim or an installed-package patch.

The parent performed user-approved, guarded installation from the coherent local source with the
minimal finite-storage fix. Component source and installed `finite_storage.py` hashes matched.
The component's own configured verification initially refused the parent's shared environment:
the parent had newer lab-commons than this component's locked `4a9...` version. An isolated
component `.venv` was bootstrapped through its own frozen dependency door; the shared parent
environment was not downgraded, and freshness checks were not bypassed.

## Evidence and remaining work

- Component configured verification: **50 tests pass**, including the new circulating-current
  regression, native OOM/scoped-factor lifetime, weak/tiny border constraints, index-two,
  initial field/rate and right-Newton controls. All 224 component source Ruff/format checks pass.
- Parent `measure-26604.log`: **600 collected, 4 failed**. All 596 previously selected parent
  controls pass on the coherent engine, including the actual field-owned network and the two
  moving-master controls. The four failures are newly planted thin-wall geometry extraction
  regressions, not initial-storage failures.
- Earlier nested-root aggregate measurement `measure-17740` was **incomplete** because its
  component tail did not emit the parent census. Do not quote it as a valid aggregate result.
- Final component fix commit, parent dependency pin update and remote publication remain pending
  the parent's approval/verification workflow. No sub-agent committed or pushed this work.
- This entry establishes this engine correction only. It does **not** establish completion of
  the parent's entire SDM wall-model scope; thin-wall geometry and broader acceptance remain
  separately tracked in the parent project's gap audit.

## Split development kit and PR publication (later in this session)

The numerical feature was locally committed as `2b88c75` after the initial verification. Publishing
then surfaced a second stale declaration: upgrading lab-commons to `2618779` removed its former
`dev` extra and `lab_commons.dev` namespace. The authoritative replacement is the separate
`dev-lab` package (`2235c51c5eeead0b1e8e57413d9b24682bd9d1e4`, `integrate/main`), using
`dev_lab.dev.*`. Component dev dependencies, configured producer/doors, generated family files
and architecture imports now use it; `tool.lab_commons` configuration namespaces remain unchanged.

The isolated component environment was resolved/synchronized through the guarded dependency door;
the parent environment was untouched. Three old generated hook shims were preserved byte-for-byte
in this checkout's private Git hook backup directory, then the new kit's normal installer installed
and checked all four actual stages, including `post-commit`. No hook was bypassed.

The actual PR policy now explicitly declares ephemeral `pr/*` heads alongside integration/trunk.
Tests pin that named set and actual hook commands. Four newly registered rules have real hook
mechanisms; five new rules without local proof remain explicitly named as absent. The absence
ceiling is an explicit registry migration (`47 - 1 retired ID + 5 new unproved rules = 51`), not a
computed exemption or a numerical tolerance change.

New-kit configured verification: **75 tests pass** (the same 50 numerical/lifetime controls plus
25 affected architecture controls), all 224 source files Ruff/format clean. Verdict:
`tree=sha256:96d379be45c0292b`, `env=1f624090778013c5`,
`log=sha256:2161ffa20c7ecd12dcd14a1360fd0b9c1e03846d018f7166cd33e5fafde2428d@28`.
Publication, resulting final component SHA and parent resolver/pin update remain parent-owned;
no sub-agent merged integration or pushed.

### Upstream supersedes the local tooling migration

A fresh authoritative fetch later showed `origin/integrate/main` at `cf2c321`, already containing
both the historical memory guard and the upstream dev-lab/PR-policy adoption. The local uncommitted
tooling migration described above was therefore reversed by minimal patches to its eleven
configuration/generated/architecture files, preserving this numerical fix and regression. Its
75-test result remains historical evidence for that local migration, not a claim about the newly
fetched upstream tree. The parent owns synchronization with current integration, fresh verification
and publication of the numerical feature; no duplicated framework migration should be published.
