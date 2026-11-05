---
name: the-new-kit-refreshes-the-declared-guard-not-its-tests
description: "Repair remotely observed generated-guard drift after the dependency upgrade without weakening checks or running local tests."
metadata:
  type: project
---

# Generated adoption must match the installed owner

The Linux base job 127600396604 in run 42128800968 reports 1488 passing
and four failing tests. Two concern missing generated workflow companions,
already repaired in local a008fea. Two concern generated guard drift: the
installed owner adds a close verb in refusal text and requires disabling
the superseded sharp-review plugin in project wiring.

Ran the repository's scripts/deny_rules.py renderer and the declared
dev_lab.dev.agent_guard installer. Only the expected generated refusal text
and enabledPlugins wiring changed; the engine and all 21 guard rules remain.
The hook installer additionally creates the normal issues-backed rem config.
No test, expected test outcome, acceptance bar, or credential was changed.
No local runtime tests were run. The owner installer reports live wiring;
that static installation report is not a remote test verdict.

The same job also could not publish its failed result asset after three
remote API attempts. Git push of a008fea and owner issue creation likewise
failed three times remotely. Preserve local repairs and retry only once the
external forge state changes; do not bypass a hook, drop files, or call this
an acceptance result.
