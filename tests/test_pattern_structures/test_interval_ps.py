import pytest

from paspailleur.pattern_structures.interval_ps import IntervalPS
from bitarray import frozenbitarray as fbarray
import math


def test_intersect_patterns():
    ips = IntervalPS()
    assert ips.join_patterns((0, 1), (2, 3)) == (0, 3)
    assert ips.join_patterns((1.5, 2), (1.1, 1.9)) == (1.1, 2)

    assert ips.join_patterns(ips.max_pattern, (2, 3)) == (2, 3)
    assert ips.join_patterns((2, 3), ips.max_pattern) == (2, 3)


def test_bin_attributes():
    data = [(0, 1), (2, 3), (1.5, 2)]
    patterns_true = ((0, 3), (1.5, 3), (2, 3), (0, 2), (0, 1), (math.inf, -math.inf))
    flags_true = (
        '111',  # (0, 3)
        '011',  # (1.5, 3)
        '010',  # (2, 3)
        '101',  # (0, 2)
        '100',  # (0, 1)
        '000',  # None
    )
    flags_true = tuple([fbarray(flag) for flag in flags_true])

    ips = IntervalPS()
    patterns, flags = list(zip(*list(ips.iter_attributes(data))))
    assert patterns == patterns_true
    assert flags == flags_true


def test_is_subpattern():
    ips = IntervalPS()

    assert ips.is_less_precise((0, 1), (0.5, 0.7))
    assert not ips.is_less_precise((0, 6), (3, 10))


def test_n_bin_attributes():
    data = [(0, 1), (2, 3), (1.5, 2)]

    ips = IntervalPS()
    assert ips.n_attributes(data) == 6


def test_intent():
    data = [(0, 1), (2, 3), (1.5, 2)]

    ips = IntervalPS()
    assert ips.intent(data) == (0, 3)

    for i, x in enumerate(data):
        assert ips.intent(data, [i]) == x


def test_extent():
    data = [(0, 1), (2, 3), (1.5, 2)]

    ips = IntervalPS()
    assert list(ips.extent(data, (1.5, 3))) == [1, 2]


def test_preprocess_data():
    data = [(0, 1), [1, 2], range(5), 10.2, 1]

    ips = IntervalPS()
    assert list(ips.preprocess_data(data)) == [(0., 1.), (1., 2.), (0., 4.), (10.2, 10.2), (1., 1.)]

    with pytest.raises(ValueError):
        next(ips.preprocess_data(['x']))


def test_verbalize():
    ips = IntervalPS()
    assert ips.verbalize([1, 2]) == "[1.00, 2.00]"
    assert ips.verbalize([0.2, 0.7], number_format='.0%') == '[20%, 70%]'
    assert ips.verbalize([0.1, math.inf]) == '>= 0.10'
    assert ips.verbalize([-math.inf, 0.265]) == '<= 0.27'
    assert ips.verbalize([math.inf, -math.inf]) == '∅'


def test_closest_less_precise():
    ips = IntervalPS(ndigits=2)
    assert list(ips.closest_less_precise((1, 2))) == [(1, 2.01), (0.99, 2)]
    assert list(ips.closest_less_precise((1, 2), use_lectic_order=True)) == [(1, 2.01)]
    assert list(ips.closest_less_precise((-math.inf, math.inf))) == []


def test_closest_more_precise():
    ips = IntervalPS(ndigits=2)
    assert list(ips.closest_more_precise((1, 2))) == [(1, 1.99), (1.01, 2)]
    assert list(ips.closest_more_precise((1, 2), use_lectic_order=True)) == [(1, 1.99)]
    assert list(ips.closest_more_precise((1, 1))) == [ips.max_pattern]
    assert list(ips.closest_more_precise(ips.max_pattern)) == []


def test_keys():
    ips = IntervalPS(ndigits=2)
    data = [(1, 3), (1, 2)]
    assert ips.keys((1, 3), data) == [(-math.inf, math.inf)]
    assert ips.keys((1, 2), data) == [(-math.inf, 2.99)]
