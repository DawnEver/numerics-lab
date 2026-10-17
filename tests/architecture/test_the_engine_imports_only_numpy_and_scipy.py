"""The engine imports nothing but itself, the standard library, numpy and scipy (plan R1, Track B4.5).

In particular it never imports `motronics`, the product it was extracted from: a consumer depends
on the engine, never the reverse.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parents[2] / 'python' / 'numerics_lab'
_ALLOWED = frozenset({'numerics_lab', 'numpy', 'scipy'}) | frozenset(sys.stdlib_module_names)


def foreign_roots(source: str) -> set[str]:
    """Top-level packages ``source`` imports that are not the engine, numpy, scipy or stdlib."""
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split('.')[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split('.')[0])
    return roots - _ALLOWED


def test_the_scan_sees_a_planted_product_import() -> None:
    planted = 'from motronics.contracts import X\nimport numpy as np\nfrom scipy import sparse\n'
    assert foreign_roots(planted) == {'motronics'}


def test_the_engine_imports_only_numpy_scipy_and_itself() -> None:
    files = sorted(_PACKAGE.rglob('*.py'))
    assert len(files) > 50, f'the scan reached only {len(files)} modules'
    offenders = {
        path.relative_to(_PACKAGE).as_posix(): found
        for path in files
        if (found := foreign_roots(path.read_text(encoding='utf-8')))
    }
    assert offenders == {}
