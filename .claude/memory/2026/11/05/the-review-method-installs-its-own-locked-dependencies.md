---
name: the-review-method-installs-its-own-locked-dependencies
description: "Adopt the existing dev-lab reproducible review installation fix without changing credentials or bypassing review."
metadata:
  type: project
---

# Reproduce the method environment, not only its top-level revision

Run 42127792419 at 761609e passes the Linux base part after the November
catalogue repair. The Windows part was still running when this entry was
written. Review still fails: its pinned ai-lab imports API_KEY_TASKS, but the
floating installation resolves lab-commons beyond ai-lab's own lock, where
that name has been retired. This is not evidence of a missing credential.

The owner already fixed the cause in dev-lab 0ad7c9b542 (#107): export the
selected ai-lab checkout's frozen lock, install the exported graph, and install
the project with --no-deps. Do not duplicate the implementation or restore a
compatibility constant. No key was replaced, disclosed, or written here.

The declared dependency door's lock-upgrade follows the project's unqualified
Git requirements to their current tips. Its actual result is dev-lab
f359db349e804e6037bc40263a4b990d5d8dd51f, not an exact 0ad7 pin.
The forge compare confirms it is 42 commits ahead and zero behind 0ad7;
the rendered workflow contains the frozen ai-lab export and no-deps installs.
lab-commons is ed24ab516a140b62b77a89a1b546d5fee8903d22. The tool also updates
pydantic, pydantic-core and Ruff, and adds the declared cryptography/ops
dependencies. These are new-environment changes requiring fresh remote tests.

The environment changed through dev_lab.dev.dep, with no verdict in flight
locally and no local tests. Its env_key is ecc9f8835d9e6554. The CI file was
rendered by the installed owner, not edited manually. Shared dev-lab and ai-lab
checkouts remain untouched. This changes no numerical law or acceptance bar.
Both the limiter and its consumer remain draft/unaccepted until actual remote
test parts and review pass; a green security job alone cannot establish that.
