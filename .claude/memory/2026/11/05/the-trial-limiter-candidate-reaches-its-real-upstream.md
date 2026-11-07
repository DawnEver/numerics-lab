---
name: the-trial-limiter-candidate-reaches-its-real-upstream
description: "Prepare publication against real numerics-lab integration, preserving the limiter/floor fixes and requiring fresh remote CI rather than reusing premerge test results."
metadata:
  type: project
---

# The limiter candidate reaches its real upstream

The owned clone was seeded from a local sibling checkout, not from a publication
remote. That old origin is now named seed. The canonical origin is the project's
declared https://github.com/DawnEver/numerics-lab.git. Nothing was pushed into
the sibling checkout, and its unrelated dirty state was never changed.

The local 8fed605 and 070c9dd fixes are based on the product's locked 4064fef.
The real origin/integrate/main is e185afc4409ba9b0a781d5d95d2aeced3ab031d2;
its merge into the candidate has no conflicts. A read-only independent source
review verified the formulation trait, physical trial mapping, directional ray,
retained-increment arithmetic evidence and limiter tests remain unchanged.

Upstream adds cancellation-aware storage-border rank detection and its grounded
JFET regression. That can affect initial-state decomposition despite not
overlapping the limiter, so the earlier 103-pass measurement is NOT validation
of the merged candidate. Upstream also changes family pins, environment hooks
and hosted test coverage. No local runtime tests are run: the user now requires
remote validation. The publication is a draft until actual postmerge CI proves
the engine feature, and it does not represent completion of the consumer's
pnjlim/raw-law or charged-initialization work.

The full consumer objective remains active. In particular the remaining
third-derivative audit identifies real discontinuities at the current Ikf gate
and BV substitution: an outgoing rate cannot invent ordinary derivatives at
a value jump. The trial limiter alone does not certify those physical seams.
