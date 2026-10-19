"""A famtests body that reads the developer checkout runs only where one exists: its test carries the marker."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from lab_commons.dev.famtests.boxneeds import assert_box_bodies_are_marked

_ROOT: Final = Path(__file__).resolve().parents[2]


def test_every_box_body_carries_its_resource_marker() -> None:
    pyproject = (_ROOT / 'pyproject.toml').read_text(encoding='utf-8')
    assert_box_bodies_are_marked(tests=_ROOT / 'tests', pyproject=pyproject, floor=1)
