"""ONE-BRANCH-PER-SESSION, MERGE-OR-DELETE: origin and this box carry only `[tool.lab_commons.branchset]`.

The measurement and the refusals are `lab_commons.dev.famtests.branchset`; this file points them at
this checkout. The declared set is the trunk `main` plus the integration branch `integrate/main`; a
lane is local, merges into integration and goes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from lab_commons.dev.branchset import declared_branchset
from lab_commons.dev.famtests import branchset

_ROOT: Final = Path(__file__).resolve().parents[2]


def test_the_declared_set_is_the_trunk_and_integration() -> None:
    declared = declared_branchset(_ROOT)
    assert declared.trunk == 'main'
    assert set(declared.sessions) == {'integrate/main'}


def test_origin_carries_only_the_declared_branches() -> None:
    branchset.assert_origin_branches_declared(root=_ROOT, branchset=declared_branchset(_ROOT))


def test_this_box_carries_only_the_declared_branches() -> None:
    branchset.assert_local_branches_declared(root=_ROOT, branchset=declared_branchset(_ROOT))


def test_the_branchset_guard_fires(tmp_path: Path) -> None:
    branchset.assert_the_planted_branchset_is_policed(tmp_path, trunk='main')
