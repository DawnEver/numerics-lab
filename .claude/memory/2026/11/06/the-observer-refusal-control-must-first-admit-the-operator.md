# The observer refusal control must first admit the operator

Smoke 42215858308 at 054b69b856b974ab5b845431e9defdb4f9e5c47d
has an observed Linux failure record with 1477 cells: 1476 OK and one rejected.
The stateless actual-march positive and monthly catalogue now pass on Linux.
The remaining rejection is the new observer-refusal negative, at its expected
exception-message assertion. Windows was still running when this was read.

Source diagnosis: the negative omitted eec_nonlinear_first_order=True, so
_admit_eec refused the missing operator assertion before reaching the intended
observer/EEC guard. Add the same truthful cubic-operator assertion already used
by the positive. Keep the exact observer refusal message, no-callback/no-trial/
no-read checks, and every production guard unchanged. This corrects the control's
prerequisite, not a numerical engine defect or relaxed expectation.

Ruff and formatting checks passed. The corrected negative remains locally
UNRUN and requires a new exact-head remote receipt. Physical-history/EEC and
consumer dependency adoption remain open; PR13 stays Draft.
