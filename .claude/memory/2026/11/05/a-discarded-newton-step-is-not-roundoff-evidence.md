---
name: a-discarded-newton-step-is-not-roundoff-evidence
description: "Post-8fed605 controls found false floor convergence at residual sqrt(2). Applied-increment provenance, stable stage inversion and explicit bounded endpoints fix it; 103 selected tests pass."
corrects:
  - 2026/11/05/newton-trial-limits-do-not-modify-the-law.md
metadata:
  type: project
---

# Do not certify an iterate with an unexecuted solve

The 90 passing tests recorded for 8fed605 did NOT establish safe arithmetic-floor
composition. A new physical linear-system control proved the missing case:
A=[[1,1],[1,1+1e-15]], b=[1,-1], with a trial limiter restricting increments to
1e-20. The solver reported converged=True, on_floor=True, residual=sqrt(2),
borrowing the enormous UNEXECUTED Newton correction in its generic floor.
One collected/one failed; CAS log
51fc58e6c52b509c8f3db31fea50c090a232ffec639d256a98096d807b10003b.

## Retained-state provenance

If a limiter changed the proposed physical state, the full solve's increment
cannot bound cancellation in the retained state. The generic floor omits that
discarded term; actual-state arithmetic and owner-declared residual floors remain
available. An identity limiter retains the original solve and its evidence.

Review then caught a second provenance boundary. A rejected probe's metadata
cannot replace the retained iterate's metadata. The unevaluable-last-probe
control reproduced false convergence at the SAME residual sqrt(2), CAS
1760789619f05702302931787c94f9fd85977c280ce09fb79cddab353e140ba5.

Even retaining the old limiter flag is insufficient: when NO trial is retained,
the loop's step variable holds a NEW discarded solve. A paired control with and
without a limiter reproduced this existing bug (both failed). Increment
eligibility now follows actual iterate assignment, and is explicitly cleared
after an all-unevaluable search. Probe metadata travels with the probe until it
is actually retained, not in a shared last-probe flag.

## Preserve the real bounded point

`endpoint + (bounded - argument)/alpha0` can cancel a small bound against a huge
discarded proposal. Changed physical states now use the affine inverse
`(bounded - h_u)/alpha0`; unchanged physical states take the identity fast path,
so unrestricted existing solves do not acquire roundtrip perturbations.

A finite physical target can still be unrepresentable with a huge history
offset. This is documented, not hidden by a new tolerance: residual/admission
checks use the RECONSTRUCTED physical state, never the requested target as a
side-channel verdict. Finite endpoint checks remain mandatory.

Directional search also retains the explicit bounded endpoint and interpolates
the segment as a convex combination. Subtracting a previous 1e20 state and then
re-adding the direction had lost a bounded endpoint of 1. The public endpoint
must declare the same finite canonical direction; mismatched/nonfinite
declarations now fail before slope evaluation. Default callers without an
explicit endpoint retain their old ray expression.

## Measurements

Used the repo's declared verify door and checkout source binding throughout.
Shared-resource refusals measured NOTHING and were not counted as test failures
or passes. No holder belonging to another task was stopped; longer queue waits
were used without changing resource demands.

* Initial false-floor control: one failed, as above.
* Intermediate 74 runtime tests passed, but the invocation failed its formatter;
  it was NOT called green.
* Retained-probe control: one failed, as above.
* 98 selected tests then passed, CAS
  00b13a64a399aa36f88b9b393b12aeaa4ef8990cdbd26363467a22d938561dd2.
* Additional no-update/endpoint-declaration controls: five collected/five failed,
  CAS de23efd1c7d3696704e316e58185763fdfd2165e4de020fc5ba4e8d0b3a39aa7.
* Final combined selector: **103 passed**, with Ruff check and format passing.
  CAS c3d1ec25dacab64938820276459c982d0d897f30eaf111c0754685570af4c441.
  Covers new limits, capability caching, affine/native admission, directional
  energy controls, integrators, factors, local convergence and resume.

Still a local dependency candidate, not forge acceptance or publication. Product
MNA wiring, exact-law removal, exporter changes and physical/vendor acceptance
remain open as recorded in Motronics G49. Do not publish 8fed605 alone as a safe
trial-limit implementation; this corrective commit is required too.
