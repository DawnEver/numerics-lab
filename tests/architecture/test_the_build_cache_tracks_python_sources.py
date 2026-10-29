"""A Python-only engine edit must invalidate the mixed Python/Rust wheel."""

import tomllib
from pathlib import Path


def test_every_packaged_python_source_is_a_build_cache_input():
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / 'pyproject.toml').read_text(encoding='utf-8'))
    source = root / project['tool']['maturin']['python-source']
    packaged = set(source.rglob('*.py'))
    tracked = {path for key in project['tool']['uv']['cache-keys'] for path in root.glob(key['file'])}
    assert packaged, 'the mixed package must expose its Python sources'
    missing = sorted(str(path.relative_to(root)) for path in packaged - tracked)
    assert not missing, f'Python-only edits would reuse a stale wheel: {missing}'
