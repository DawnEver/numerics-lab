# Accepted EEC linearization is a scoped real matrix read

LOCAL UNVERIFIED dependency candidate, based exactly on integration/main
e185afc4409ba9b0a781d5d95d2aeced3ab031d2, separate from PR9 and its WIP history.
All controls UNRUN. No numerical/model/native/compiler execution or push.

Prior art: nonlinear first-order EEC's after_accept directly read the endpoint
Jacobian after accepted observers/projection, outside StepEvaluationProblem.
The actual periodic math owner also reads endpoint stiffness/mass for affine
correction and endpoint mass for nonlinear correction. No accepted scope
protocol exists in this base. A constitutive owner may correctly refuse its
unowned pointwise jet at such a corner; this is source evidence, not a verified
runtime failure or a claim about any vendor algorithm.

AcceptedLinearizationProblem now declares one read-only outgoing matrix scope
at the actual accepted clock. periodic.accepted_jacobian grants it around ONLY
the existing problem.jacobian call. All state/rate/acceleration objects,
DerivativeWeights and return identity remain unchanged. Missing capability
uses nullcontext; no shim/default method is installed on ordinary problems.
The four actual EEC Jacobian reads use this single helper. The correction
solve, ordinary trial Newton, observers, projection, checkpoint and the rest
of after_accept are not inside the scope. No solver/options are disabled or
duplicated and the engine imports no product/physics module.

Authored controls cover argument/weight/result identity, immediate cleanup,
ordinary absence, actual error/typed-stop object and existing-cause retention,
correction solve outside scope and the genuine march's accumulation/correction
versus separate trial views. Existing EEC controls remain unchanged and UNRUN.

The preexisting march.py is 1636 lines. This candidate changes only its import
and actual matrix-read call; it does not claim that old file is under800 or
perform an unrelated whole-march extraction. New cohesive code/tests are small.
The Motronics installed consumer remains numerics4064fef unchanged. No consumer
pin or automatic PR9 mixture is implied; root reviews this separate dependency
and real consumer implementations before any external publication.
