---
name: retired-trial-helper-is-detected-in-real-python-spellings
description: Address review F-3f217755d9ab without weakening the retired-name scan
created: 2026-11-06
metadata:
  type: project
---

# Retired helper scan covers actual Python use

Execution client: Codex. Review comment 6681210759 on PR 9 identifies
F-3f217755d9ab: the retirement registry's dotted text does not occur in normal
absolute/relative imports, module calls or the old function definition.
Independent source inspection confirms live_spellings only searches literal
substrings. This finding remains valid at 7c42e510d30e despite its numerical
base passes; a no-match scan is not proof of effective retirement enforcement.

Register the removed helper's unique identifier instead of its fully qualified
path. Keep the existing complete Python/Rust/tests scan unchanged. Add a planted
control for all four actual Python spellings, deriving the identifier from the
one registry entry rather than maintaining another retirement fact.

This changes no numerical source or public capability. Ruff/format/AST and a
read-only source occurrence audit are allowed locally; runtime controls remain
UNRUN and need new exact-head remote verification. The 1517-per-platform passes
belong to parent 7c42, not this follow-up. Other review findings and the trusted
review-auth prerequisite are not closed by this correction.
