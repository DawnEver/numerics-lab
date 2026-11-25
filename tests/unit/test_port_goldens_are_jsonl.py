"""The port goldens and the expm fixture are JSONL records (ruling R3), holding the values the JSON held.

Each pin is the sha256 of the canonical value (sorted keys, compact) measured on the ``.json`` file this
``.jsonl`` replaced (2026-11-24): bit-identical floats or the hash moves. A deliberate regeneration of a
golden updates its pin in the same commit.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from tests.port_goldens import GOLDENS, load_golden

_FIXTURE = GOLDENS.parent / 'fixtures' / 'expm_scipy.jsonl'

_PINS = {
    GOLDENS / 'interp.jsonl': '99836debc2451a72fe0daa2a822b81e4c2a23274204338c0f4e46fe961309026',
    GOLDENS / 'linear.jsonl': 'a6b1ae6dabeb5a97073687c9ff267c39581fec2869e15962348d3d2cb55108d6',
    GOLDENS / 'nnls.jsonl': '689b375758e6b60490d72185a36657119778ce222d39f1a14387fb75b6d239a1',
    GOLDENS / 'nonlinear.jsonl': '694648ad615c7635f26e2064b3e7678adbfb15b90a83bdcd002684870ce95726',
    _FIXTURE: '0869ba6fba4233f3f74c57d3d918868915e173a5ce494f4e76b54b7adb6ed928',
}


def _canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _value(path):
    if path == _FIXTURE:
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    return load_golden(path)


@pytest.mark.parametrize('path', sorted(_PINS), ids=lambda p: p.name)
def test_the_jsonl_holds_the_value_the_json_held(path):
    assert _canonical(_value(path)) == _PINS[path]


def test_no_json_golden_is_left_beside_them():
    assert not list(GOLDENS.glob('*.json'))
    assert not list(_FIXTURE.parent.glob('*.json'))


def test_a_golden_of_two_lines_is_refused_by_name(tmp_path):
    """Planted control: the one-record reader says no."""
    two = tmp_path / 'two.jsonl'
    two.write_text('{}\n{}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='ONE JSONL record'):
        load_golden(two)
