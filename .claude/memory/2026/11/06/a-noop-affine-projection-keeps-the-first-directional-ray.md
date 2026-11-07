---
name: a-noop-affine-projection-keeps-the-first-directional-ray
description: Address first-iteration directional recovery without moving the original guess or making idempotence a convergence gate
created: 2026-11-06
metadata:
  type: project
---

# A sufficient no-op fact preserves the first directional search

Execution client: Codex. Prior art: _step explicitly keeps the caller's initial
guess unprojected to preserve its first linearization; tested recovery uses the
existing merit/admission rules and retained-trial provenance. The near-idempotent
projector controls forbid demanding exact idempotence on every subsequent
iteration. Keep both requirements.

F-6da21b84ad3a identifies a separate actual source gap: absent an explicit
affine admission certificate, projected_iterate starts false even when the
first guess is an exact no-op of its declared projector. The first directional
search is then unnecessarily replaced by residual-merit recovery.

After computing the unchanged first residual/Jacobian, check this one sufficient
fact once, only for directional+limiter+projection without declared admission.
Read the actual projector on a copy of the actual physical stage point, with
the existing finite/shape contract. An exact unchanged projection permits the
original first ray. A changed or explicitly unevaluable projection supplies no
certificate: keep evaluated recovery and its original acceptance policy.
Never install the probe as a new guess, apply the limiter in this probe, loosen
a tolerance, or require bit-exact idempotence again after retained recovery.
Ordinary callback failures and stop signals still propagate.

The new actual take_step/directional_line_search spy control asserts an exact
no-op projector lacking admission searches from the original zero point on
iteration one and converges. Existing near-idempotence, rejected recovery,
off-plane first-linearization and typed-stop controls remain unchanged. All
runtime controls are UNRUN locally; source checks do not prove convergence.
