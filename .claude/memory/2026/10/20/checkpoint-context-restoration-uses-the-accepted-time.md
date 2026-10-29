---
name: checkpoint-context-restoration-uses-the-accepted-time
description: "Selected-time physical-context restoration fixes a real moving signed-PM portable restart without rebasing its physical origin."
metadata:
  type: project
---

# Accepted numerical time owns checkpoint-context restoration

The consumer's synthetic conducting signed-PM ring crossed its actual motion corner, but portable BDF1/BDF2 restarts refused the saved physical context. The checkpoint time and saved context were correct; the freshly reconstructed moving owner correctly retained its original physical origin. The engine dispatched only the no-time context-restoration method, which validated against the constructor's initial time instead of the checkpoint's accepted time.

`python/numerics_lab/time_integrators/checkpoint.py` now declares optional `CheckpointContextAt`. After the existing checkpoint/context validations, restoration passes `checkpoint.time` to owners implementing that capability. Other owners retain their actual no-time contract. Metadata cannot choose the numerical clock, and neither the physical origin nor admission tolerances were changed.

`tests/unit/test_checkpoint_context_uses_accepted_time.py` formally measured RED: two failures and one pass. It checks accepted-time dispatch, rejection of a forged metadata time, and preservation of the no-time contract. Targeted verification of sixteen affected checkpoint, context, history, landing, projection, nominal-clock and restart files then measured **221 collected, zero failures** (receipt `ec56edb1...@20`). The full-suite command was explicitly refused because that verdict belongs to the forge producer after a push; no push was authorized or performed.

The approved supported installer refreshed the consumer environment under its resource lock. All 75 installed Python files matched component source bytes; native and `CheckpointContextAt` imports succeeded. Installed `checkpoint.py` SHA256: `bae6bcec52b3d5a03fd16003ed06e4f32f7ab383442e841761cafbb8b1e283eb`.

Post-install consumer verification measured **33 collected, zero failures** (session 5094), including actual corner crossing and portable BDF1/BDF2 restart, public compiler, material-domain and accepted-observer checks. This is a synthetic-ring trajectory/restart verdict, not full downstream trajectory parity or a forge full-suite verdict.
