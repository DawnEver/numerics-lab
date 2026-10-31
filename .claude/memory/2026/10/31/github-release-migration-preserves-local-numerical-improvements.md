---
metadata:
  type: project
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
