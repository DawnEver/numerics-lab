---
name: numerics-pushes-to-github-like-every-toolchain-repo
description: The disabled origin pushurl was a pre-row stopgap; numerics-lab publishes pr/* to GitHub like every toolchain repo, guarded by the public-root pre-push row.
metadata:
  type: project
---
# numerics-lab pushes to GitHub like every toolchain repo (2026-11-08)

- User, 2026-11-08: "numerics-lab 等等 工具链仓库都在 github 都是通过 pr 合并的逻辑啊？！" -- every toolchain repo
  (dev-lab, lab-commons, ai-lab, numerics-lab, optimi-lab, learn-lab) publishes `pr/*` to GitHub DawnEver and lands by PR.
- Misleading state found: the main checkout's machine-local `.git/config` carried
  `remote.origin.pushurl = DISABLED-push-from-the-private-history-repo` (and `remote.gitea-private.pushurl =
  DISABLED-private-history`). It was set by hand in the D78/D86 era (2026-10-29), when this checkout still sat on
  the private Gitea history. No tracked file declares it, so nothing explains it to a reader, and every
  `with-retry.sh push origin pr/x` fails with "does not appear to be a git repository".
- Why it is baggage now: the checkout was switched to the public history on 2026-11-07 (HEAD bdd43e5 descends from
  the public root), and the security control is the DECLARED `public-root` pre-push row (`scripts/public_root.py`,
  9a035ff, #22): a GitHub push must descend from `PUBLIC_ROOT` and reach no second root; with no url it fails closed.
  `tests/architecture/test_only_the_public_root_reaches_github.py` proves it. A second, undeclared, machine-local
  copy of the control is a declaration nobody reads.
- Remedy (machine-local, not tracked): back up `.git/config` to `.git/config.bak-20261108`, then
  `git config --unset remote.origin.pushurl`. KEEP `remote.gitea-private.pushurl` disabled: the private history is
  read-only and is never pushed anywhere, nor deleted, nor rewritten.
- No other toolchain repo carries a disabled pushurl (dev-lab, lab-commons, ai-lab, optimi-lab checked 2026-11-08).
