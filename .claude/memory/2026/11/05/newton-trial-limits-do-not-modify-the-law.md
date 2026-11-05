---
name: newton-trial-limits-do-not-modify-the-law
description: "A problem-owned Newton trial limit is required to remove permanent constitutive clipping downstream; 90 selected engine tests pass, including affine/gmin/directional composition. Local candidate only, not published."
metadata:
  type: project
---

# Bound the iterate, not the equation

## Why this change exists

Motronics' PLECS alignment found that a diode law permanently continued by its
tangent changes a converged high-current operating point. Its product-side
declared Newton now limits intrinsic voltage, but the engine's implicit-step
Newton had no owner limit. `TrialNotEvaluable` already rejects uncomputable line
search probes; affine projection is NOT a substitute for nonlinear iterate
limiting. Removing the law continuation first would leave other consumers
unprotected.

Prior art: formulation optional protocols, ProblemCapabilities, affine trial
projection/admission, TrialNotEvaluable, directional line search, gmin continuation,
native-step admission. This checkout is a LOCAL independent clone inside the
motronics workspace, based on its exact installed/locked engine 4064fefaa831.
The sibling source checkout was read, not edited. No dependency pin was changed.

## Contract and review decisions

NewtonTrialLimitProblem receives private copies of the previous and proposed
PHYSICAL stage states at the actual stage time. It does not change the residual,
acceptance bar or accepted history. Native kernels lacking this contract decline
to the Python solver when the capability is present. Unrestricted Newton steps
remain available to existing arithmetic-floor checks.

The first implementation passed nine tests but review found two genuine gaps:
gmin skipped affine projection, and directional search could start from an
unprojected/off-plane guess. New tests reproduced BOTH (nine pass/two fail).
Now one shared trial operation composes projection BEFORE limiting for normal,
backtracking, fallback and gmin paths. The first linearization stays unprojected;
directional search restores a feasible iterate before searching its straight
bounded ray. Without an admission certificate, projector fixed-point identity is
a conservative trial-ray eligibility check, NOT a convergence criterion.

Physical-state results are mapped through the stage's real nonzero alpha0;
both the owner result and reconstructed endpoint must be finite. A finite
physical result with an unrepresentable endpoint is an error, not a hidden
overflow. The owner must preserve owned affine constraints and ensure its
bounded directional segment is evaluable; the engine does not invent a device
limit or a different law along that segment.

## Measurements

Used this repository's declared `dev_lab.dev.verify -- <paths>` door, not a bare
pytest invocation or a producer/forge part. The separate frozen environment was
bootstrapped with `dev_lab.dev.dep --bootstrap . --extra dev`, because the product
venv correctly failed this engine lock's dev-tool identity check. Local hooks
were installed through `dev_lab.dev.hook_install --install`. Tests assert that
the imported Python package comes from this checkout's python/, not the installed
baseline wheel. These are local measurements, not forge acceptance.

* Initial TDD: nine collected/nine failed (unlimited proposals reached the law).
* First implementation: nine passed; review regressions then nine passed/two failed.
* Fixed composition: **39 passed**, selector covers new limits, capability caching,
  affine admission, directional energy search and native-step admission.
  CAS log: b2609660bf613ee302b56a433c1949bac2d1c8e8ee27627cbd62ba1d9131054b.
* Adjacent integrator/factor/local-convergence/resume checks: **51 passed**.
  CAS log: 5266143956d64cf6531c5b037fba6cc19b4da6869bacd701aa0e4ce68435598e.
* Ruff check and format pass. No full-suite or publication verdict claimed.

The exact-exponential control overflows at the unlimited first proposal, while
the limited solve reaches log(1001) without a permanent law cap. Other tests
cover invalid owner returns, physical-vs-endpoint coordinates, endpoint overflow,
native bypass refusal and both line-search choices.

Still required downstream: wire this protocol to the actual intrinsic junction
coordinates across MNA/OP/continuation, update exporters and remove the permanent
tangent together, update the real dependency through its publication workflow,
and run the physical circuit/vendor regressions. This engine seam alone is NOT
completion of Motronics' pnjlim or PLECS-alignment objective. No push or PR.
