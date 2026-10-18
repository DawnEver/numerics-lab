"""A random stream is identified by its seed and path, never by creation or draw order."""

from __future__ import annotations

import numpy as np
import pytest

from numerics_lab.random_streams import Stream


def _draw(stream: Stream) -> np.ndarray:
    return stream.generator().random(4)


def test_the_same_seed_and_path_give_the_same_numbers() -> None:
    np.testing.assert_array_equal(_draw(Stream(5, ('a', 'b'))), _draw(Stream(5).child('a').child('b')))


def test_draws_elsewhere_do_not_shift_a_stream() -> None:
    before = _draw(Stream(5).child('mutation'))
    Stream(5).child('crossover').generator().random(1000)
    np.testing.assert_array_equal(_draw(Stream(5).child('mutation')), before)


@pytest.mark.parametrize(
    ('left', 'right'),
    [
        (Stream(5, ('a',)), Stream(5, ('b',))),
        (Stream(5, ('ab',)), Stream(5, ('a', 'b'))),
        (Stream(5), Stream(5, ('a',))),
        (Stream(5, ('a',)), Stream(6, ('a',))),
    ],
)
def test_different_streams_differ(left, right) -> None:
    assert not np.array_equal(_draw(left), _draw(right))


@pytest.mark.parametrize(('seed', 'path'), [(-1, ()), (True, ()), (1.5, ()), (1, ('',)), (1, ['a'])])
def test_a_malformed_stream_is_refused(seed, path) -> None:
    with pytest.raises(ValueError, match='stream'):
        Stream(seed, path)
