---
name: the-lifecycle-ratchet-names-all-three-owner-workflows
description: "Update the exact lifecycle-name assertion after the owner split its generated workflow family; preserve drift checks."
metadata:
  type: project
---

# A stricter named set follows the actual owner structure

The local generated-workflow and guard repairs reached origin at 88bfb73
after a subsequent forge comment write demonstrated the remote had recovered.
No hook was bypassed. Actual run 42129782029 now reports Linux 1491 passing
and one failing test. The remaining failure is the repository's explicit
one-file lifecycle assertion, which still expects only ci.yml while the
adopted owner declares three files.

Change the exact expected set to ci.yml, merge-tick.yml and review-claims.yml.
This is an intentional architecture migration, not removal of an assertion:
missing or unexpected lifecycle files still fail, and the separate drift
assertion remains unchanged. Both generated companion files are committed.

The same run's review still fails at the trusted-base CLI mismatch recorded
previously; passing numerical parts cannot replace review. Windows was still
running when this entry was written. The new assertion needs fresh remote
execution; no local test was run, skipped or weakened.
