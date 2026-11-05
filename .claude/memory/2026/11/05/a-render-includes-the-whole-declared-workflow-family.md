---
name: a-render-includes-the-whole-declared-workflow-family
description: "Publish all owner-rendered lifecycle files and distinguish draft execution policy from remote acceptance."
metadata:
  type: project
---

# The generated family includes its lifecycle companions

The dependency renderer also creates merge-tick.yml and review-claims.yml.
The earlier commit staged only the existing ci.yml: its diff stat did not
include untracked generated companions. Include both owner-rendered files;
do not hand-write replacements or omit declared capabilities.

The current owner explicitly runs no PR jobs for drafts. Run 42128578596 at
2b1a112 is skipped, not passed. PR 9 was marked ready for review to request
fresh remote validation. This is not acceptance of the limiter or its still
draft motronics consumer. No auto-merge was enabled manually. Actual new
run 42128800968 started; numerical and review results remain pending.

The preceding 761609e run completed both base parts and absent-parts
successfully. Its review failure remains recorded. A generic status-list
answer of 1467 passing tests is not evidence that the new head's jobs ran;
actual Actions runs and head commit statuses must also be inspected.

The lock-upgrade touched rust-toolchain.toml line endings only. Its normalized
Git content is unchanged; refresh the index without creating a content change.
No local test was run and no acceptance threshold changed.
