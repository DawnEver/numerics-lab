"""A `numerics_lab/` portion LATER on `sys.path` is part of the package, so a separately built kernel is reachable.

The cluster puts each sha's `python/` on `PYTHONPATH` and the native build's output (`$LAB_CI_NATIVE`,
`[tool.lab_commons.hpc] native`) after it. Python resolves `numerics_lab._native` through
`numerics_lab.__path__` alone, so without the package extending its path the regular package in
`python/` hides the built extension and every Rust-backed test runs a stale kernel or none.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_SOURCE = Path(__file__).resolve().parents[2] / 'python'


def test_a_later_portion_supplies_a_submodule(tmp_path: Path) -> None:
    portion = tmp_path / 'numerics_lab'
    portion.mkdir()
    (portion / '_portion_probe.py').write_text('MARK = 7\n', encoding='utf-8')
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(_SOURCE), str(tmp_path)])}
    probe = 'import numerics_lab._portion_probe as m; print(m.MARK)'
    out = subprocess.run(
        [sys.executable, '-c', probe], env=env, capture_output=True, text=True, check=False, timeout=120
    )
    assert (out.returncode, out.stdout.strip()) == (0, '7'), out.stderr
