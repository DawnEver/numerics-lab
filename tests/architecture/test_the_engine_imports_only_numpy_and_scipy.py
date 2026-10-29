"""The engine imports nothing but itself, the standard library, numpy and scipy's linalg/sparse/special.

Plan rulings R1 and Track B4.5: every algorithm is self-implemented, so `scipy.optimize`,
`scipy.integrate`, `scipy.stats` and the rest appear only in tests, as oracles. In particular the
engine never imports `downstream`, a stand-in for the product it was extracted from.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parents[2] / 'python' / 'numerics_lab'
_ALLOWED_ROOTS = frozenset({'numerics_lab', 'numpy'}) | frozenset(sys.stdlib_module_names)
#: scipy is allowed only through these subpackages (and their own submodules).
_ALLOWED_SCIPY = ('scipy.linalg', 'scipy.sparse', 'scipy.special')


def _allowed(module: str) -> bool:
    if module.split('.', maxsplit=1)[0] in _ALLOWED_ROOTS:
        return True
    return any(module == allowed or module.startswith(f'{allowed}.') for allowed in _ALLOWED_SCIPY)


def foreign_imports(source: str) -> set[str]:
    """Modules ``source`` imports that are not the engine, numpy, scipy's allowed subpackages or stdlib."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == 'scipy':
                found |= {f'scipy.{alias.name}' for alias in node.names}
            else:
                found.add(node.module)
    return {module for module in found if not _allowed(module)}


def test_the_scan_sees_planted_foreign_imports() -> None:
    planted = (
        'from downstream.contracts import X\nimport numpy as np\nfrom scipy import sparse\n'
        'from scipy.sparse.csgraph import connected_components\nfrom scipy.optimize import newton_krylov\n'
        'from scipy import stats\nimport scipy\n'
    )
    assert foreign_imports(planted) == {'downstream.contracts', 'scipy.optimize', 'scipy.stats', 'scipy'}


def test_the_engine_imports_only_numpy_scipy_and_itself() -> None:
    files = sorted(_PACKAGE.rglob('*.py'))
    assert len(files) > 50, f'the scan reached only {len(files)} modules'
    offenders = {
        path.relative_to(_PACKAGE).as_posix(): found
        for path in files
        if (found := foreign_imports(path.read_text(encoding='utf-8')))
    }
    assert offenders == {}
