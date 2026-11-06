---
name: junction-limiter-syncs-the-staged-review-door
description: Resolve the integration conflict using its coherent generated workflows and staged dependency pin without changing numerical code
created: 2026-11-06
metadata:
  type: project
---

# Junction limiter synchronizes the staged integration prerequisite

Execution client: Codex. PR 9 remained open at bc91bc5a0f8b when checked.
The integration comment 6685903561 reports a real merge conflict with
integrate/main e2069c924730f74f37a1883c49a00d542962ad07. Its standing base
receipts report 1517 tests on Linux and Windows; these are the old head's
receipts, not evidence for this new merge. Review was reported successful with
nine nonblocking findings; merge-ready still refused its posted provenance.
Do not confuse that receipt with approval to bypass the review mechanism.

The upstream integration includes the staged dev-lab 6f6b042a pin and a relock.
Resolve lock and workflow conflicts with that coherent upstream set. The
automatically merged ci.yml would otherwise retain the lane's newer key-bundle
renderer while taking the staged dependency. Take the upstream generated
ci.yml as well; do not hand-author a hybrid workflow or weaken trusted-base
review isolation. No numerical Python or Rust source changes in this merge.

MERGE-RESOLUTION-DECLARED:
.github/workflows/merge-tick.yml: newer-structure -- integration's generated workflow matches its staged review-door dependency
.github/workflows/review-claims.yml: newer-structure -- integration's generated workflow matches its staged review-door dependency
uv.lock: newer-structure -- integration owns the staged exact dev-lab source and complete resolution

No tests, project imports, native compilation or environment mutation occurred
locally. Check JSON/TOML and source equivalence only. Publish normally and
obtain fresh exact-head remote receipts. The consumer's numerics pin remains
unchanged, and pnjlim/full-circuit continuation acceptance remains open.
