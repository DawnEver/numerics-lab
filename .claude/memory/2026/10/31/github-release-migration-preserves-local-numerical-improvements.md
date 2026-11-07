---
metadata:
  type: project
name: github-release-migration-preserves-local-numerical-improvements
description: Preserve the eleven numerical improvements on GitHub release history, verify them and record publication authentication separately.
---

# Move the local numerical changes onto the public release history

The user identifies GitHub as the current upstream. The old local branch
pr/storage-border-memory remains intact at a38cb095. Its all-ref bundle52 was
independently restored and checked before this migration. No old history was
force-pushed or joined through an unrelated-history merge.

The old integration a69f284 and GitHub initial release02cb9d2 have identical
python, rust, tests and pyproject.toml trees (git diff --quiet returns zero).
Create pr/storage-border-github from origin/integrate/main26106e5 and replay
only the eleven numerical improvement commits following a69f284. Preserve the
upstream rank_bar function while adding the independent bounded-work constants;
capacity is not a rank tolerance. All eleven changes are now represented.

Bootstrap through dev_lab.dev.dep --bootstrap with the committed frozen lock.
The refreshed dev-lab9e1fb475 and lab-commons9e9236ee come from GitHub, and the
local native extension is rebuilt. The first bounded canonical measurement
passes all118 numerical tests but fails the formatter on the combined export
list; expand that list without a semantic change and repeat verification.
The repeated canonical measurement adds upstream sparse-initial-pair tests:
148 pass, Ruff passes and all266 files pass formatting. Evidence resides in
the enclosing checkout's output/logs/26/10/31/github-integration-sync/engine-measure3.

No full-motor success follows from this migration. The original physical
validators, memory ceiling and numerical acceptance bars remain unchanged.

## Publication outcome

GitHub push is NOT complete. A normal push waits in the editor askpass path;
interrupt only the owned command rather than alter a credential helper. The
bounded canonical netverb retry then attempts the same own PR ref three times
with20-second walls. All three time out and report: Cannot prompt because
user interactivity has been disabled. Exit124 and the structured report are
preserved at the enclosing output/logs/26/10/31/github-integration-sync/engine-push.
The refreshed GitHub kit already passed local verification. The observed
noninteractive publication failure does NOT prove that the user must log in,
provide a token or change account permissions. It is not a numerical/test failure. Leave
credentials, editor authentication and remote permissions untouched. The local
branch retains the changes for publication once authentication is available.

## Later upstream lock synchronization

A fresh canonical forge sync discovers one further integration commit,
fb253334cede01a7779ff5e876c4d980f0eac990. It changes ONLY uv.lock; the numerical
source is unchanged. The clean merge is074b684fea3d4deea075a7600c55499495324620.
Canonical dev_lab.dev.dep --sync refreshes the kit again to dev-lab73b56c74,
updates the generated lock and changes env_key to dc17667732d42bd4. Verify
the same148 numerical tests against this actual updated lock/runtime before
committing its lock receipt. The actual repeated canonical run passes all148
tests, Ruff and266-file formatting in env dc17667732d42bd4; evidence is in
the enclosing github-integration-sync/engine-after-lock-sync directory.
No extra GitHub push is attempted while the
recorded authentication condition remains unchanged.

User now requests a handoff and questions the repeated GitHub demand. Stop
publication retries; consuming upstream and preserving this local branch need
no user account action. Carry forward the observed error without prescribing
re-login. All numerical features remain local, tested and backed up.
