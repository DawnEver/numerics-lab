---
name: full-ci-found-the-missing-month-catalogue
description: "Both hosted full suites pass all numerical tests but fail the missing November catalogue; review separately fails an ai-lab/lab-commons import, not a missing credential."
metadata:
  type: project
---

# Full CI found the missing month catalogue

At f96a354, GitHub run 42125986125 executed both base platforms. Linux and
Windows each reported 1491 passed, one failed in the first group, plus four
passed in the second group: 1495 passing of 1496 executed tests per platform.
The ONLY failed test is test_every_month_carries_its_plans, naming missing
2026/11/SUMMARY-2026-11.md. The numerical controls passed, but the whole part
remains FAIL. The new catalogue records the actual partial feature; the test
is neither skipped nor weakened. Fresh CI is still needed on the repair.

The separate review job failed before reviewing code. Its masked log shows
ai-lab 23f2bd88e importing API_KEY_TASKS from lab-commons ed24ab516, where it
does not exist. This is dependency incoherence, NOT evidence of a missing
Anthropic key. No credential was exposed, replaced or re-registered. The
review dependency must be corrected through its owner; it cannot be claimed
green from the passing numerical controls.

Logs were read from completed hosted jobs through the existing authenticated
GitHub CLI; control escape bytes were stripped before displaying excerpts.
No local test or alternative producer was run. PR 9 stays draft pending both
the repaired full test parts and actual review readiness.
