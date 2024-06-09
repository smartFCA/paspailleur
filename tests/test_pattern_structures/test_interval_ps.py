import pytest

from paspailleur.pattern_structures.interval_ps import IntervalPS, BoundStatus as BS
from bitarray import frozenbitarray as fbarray
import math


def test_init():
    ips = IntervalPS()
    assert ips.min_bounds is None
    assert ips.max_bounds is None

    ips = IntervalPS(values=[3, 1, 2, 0, 10])
    assert ips.min_bounds == (0, 1, 2, 3, 10)
    assert ips.max_bounds == (0, 1, 2, 3, 10)

    ips = IntervalPS(min_bounds=[10, 10, 2], max_bounds=[10, 2, 2])
    assert ips.min_bounds == (2, 10)
    assert ips.max_bounds == (2, 10)


def test_intersect_patterns():
    ips = IntervalPS()
    assert ips.join_patterns((0, 1, BS.CLOSED), (2, 3, BS.CLOSED)) == (0, 3, BS.CLOSED)
    assert ips.join_patterns((1.5, 2, BS.CLOSED), (1.1, 1.9, BS.CLOSED)) == (1.1, 2, BS.CLOSED)

    assert ips.join_patterns(ips.max_pattern, (2, 3, BS.CLOSED)) == (2, 3, BS.CLOSED)
    assert ips.join_patterns((2, 3, BS.CLOSED), ips.max_pattern) == (2, 3, BS.CLOSED)


def test_meet_patterns():
    ips = IntervalPS()

    assert ips.meet_patterns((0, 5, BS.CLOSED), (2, 10, BS.CLOSED)) == (2, 5, BS.CLOSED)
    assert ips.meet_patterns((0, 5, BS.RCLOSED), (2, 10, BS.LCLOSED)) == (2, 5, BS.CLOSED)
    assert ips.meet_patterns((0, 5, BS.LCLOSED), (2, 10, BS.RCLOSED)) == (2, 5, BS.OPEN)
    assert ips.meet_patterns((0, 5, BS.LCLOSED), (5, 10, BS.LCLOSED)) == ips.max_pattern
    assert ips.meet_patterns(ips.min_pattern, (0, 5, BS.CLOSED)) == (0, 5, BS.CLOSED)
    assert ips.meet_patterns((0, 5, BS.CLOSED), ips.min_pattern) == (0, 5, BS.CLOSED)


def test_bin_attributes():
    data = [(0, 1, BS.CLOSED), (2, 3, BS.CLOSED), (1.5, 2, BS.CLOSED)]
    patterns_true = ((0, 3, BS.CLOSED), (1.5, 3, BS.CLOSED), (2, 3, BS.CLOSED),
                     (0, 2, BS.CLOSED), (0, 1, BS.CLOSED), (None, None, BS.CLOSED))
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

    assert ips.is_less_precise((0, 1, BS.CLOSED), (0.5, 0.7, BS.CLOSED))
    assert not ips.is_less_precise((0, 6, BS.CLOSED), (3, 10, BS.CLOSED))


def test_n_bin_attributes():
    data = [(0, 1, BS.CLOSED), (2, 3, BS.CLOSED), (1.5, 2, BS.CLOSED)]

    ips = IntervalPS()
    assert ips.n_attributes(data) == 6


def test_intent():
    data = [(0, 1, BS.CLOSED), (2, 3, BS.CLOSED), (1.5, 2, BS.CLOSED)]

    ips = IntervalPS()
    assert ips.intent(data) == (0, 3, BS.CLOSED)

    for i, x in enumerate(data):
        assert ips.intent(data, [i]) == x


def test_extent():
    data = [(0, 1, BS.CLOSED), (2, 3, BS.CLOSED), (1.5, 2, BS.CLOSED)]

    ips = IntervalPS()
    assert list(ips.extent(data, (1.5, 3, BS.CLOSED))) == [1, 2]


def test_preprocess_data():
    data = [(0, 1), [1, 2], range(5), 10.2, 1]

    ips = IntervalPS()
    assert list(ips.preprocess_data(data)) == [(0., 1., BS.CLOSED), (1., 2., BS.CLOSED), (0., 4., BS.CLOSED),
                                               (10.2, 10.2, BS.CLOSED), (1., 1., BS.CLOSED)]

    with pytest.raises(ValueError):
        next(ips.preprocess_data(['x']))

    data = [(0, 1), (1, 2), (5, 9)]
    ips = IntervalPS(values=[0, 5, 10])
    assert list(ips.preprocess_data(data)) == [(0., 5., BS.CLOSED), (0., 5., BS.CLOSED), (5, 10, BS.CLOSED)]


def test_verbalize():
    ips = IntervalPS()
    assert ips.verbalize([1, 2, BS.CLOSED]) == "[1.00, 2.00]"
    assert ips.verbalize([0.2, 0.7, BS.CLOSED], number_format='.0%') == '[20%, 70%]'
    assert ips.verbalize([0.1, math.inf, BS.LCLOSED]) == '[0.10, ∞)'
    assert ips.verbalize([-math.inf, 0.265, BS.RCLOSED]) == '(-∞, 0.27]'
    assert ips.verbalize([None, None, BS.CLOSED]) == '∅'


def test_precision():
    ips = IntervalPS(ndigits=2)
    assert ips.precision == 0.01

    ips = IntervalPS(ndigits=-2)
    assert ips.precision == 100


def test_closest_less_precise():
    ips = IntervalPS(ndigits=2)
    assert list(ips.closest_less_precise((1, 2, BS.CLOSED))) == [(1, 2.01, BS.LCLOSED), (0.99, 2, BS.RCLOSED)]
    assert list(ips.closest_less_precise((1, 2, BS.OPEN))) == [(1, 2, BS.RCLOSED), (1, 2, BS.LCLOSED)]
    #assert list(ips.closest_less_precise((1, 2, BS.CLOSED), use_lectic_order=True)) == [(1, 2.01)]
    assert list(ips.closest_less_precise((-math.inf, math.inf, BS.OPEN))) == []
    assert list(ips.closest_less_precise((0, 0.06, BS.CLOSED))) == [(0, 0.07, BS.LCLOSED), (-0.01, 0.06, BS.RCLOSED)]

    ips = IntervalPS(ndigits=2, values=[0, 1, 2, 3, 4, 5])
    assert list(ips.closest_less_precise((1, 2, BS.CLOSED))) == [(1, 3, BS.LCLOSED), (0, 2, BS.RCLOSED)]
    assert list(ips.closest_less_precise((1, 2, BS.LCLOSED))) == [(1, 2, BS.CLOSED), (0, 2, BS.OPEN)]
    #assert list(ips.closest_less_precise((1, 2), use_lectic_order=True)) == [(1, 2.99)]
    #assert list(ips.closest_less_precise((1, 2.99), use_lectic_order=True)) == [(1, 3)]
    assert list(ips.closest_less_precise((0, 5, BS.CLOSED))) == [(0, math.inf, BS.LCLOSED), (-math.inf, 5, BS.RCLOSED)]


def test_closest_more_precise():
    # TODO: Setup tests for lectic order
    ips = IntervalPS(ndigits=2)
    assert list(ips.closest_more_precise((1, 2, BS.CLOSED))) == [(1, 2, BS.LCLOSED), (1, 2, BS.RCLOSED)]
    #assert list(ips.closest_more_precise((1, 2, BS.OPEN), use_lectic_order=True, intent=(1, 2., BS.CLOSED))) == [
    #    (1, 1.99, BS.RCLOSED)]
    #assert list(ips.closest_more_precise((1, 2, BS.OPEN), use_lectic_order=True, intent=(1, 2, BS.CLOSED))) == [
    #    (1, 1.99, BS.RCLOSED), (1.01, 2, BS.LCLOSED)]
    assert list(ips.closest_more_precise((1, 1, BS.CLOSED))) == [ips.max_pattern]
    assert list(ips.closest_more_precise(ips.max_pattern)) == []
    assert list(ips.closest_more_precise((0, 0.03, BS.OPEN))) == [(0, 0.02, BS.RCLOSED), (0.01, 0.03, BS.LCLOSED)]

    ips = IntervalPS(ndigits=2, values=[0, 1, 2, 3, 4, 5])
    assert list(ips.closest_more_precise((1, 2, BS.OPEN))) == [(1, 1, BS.RCLOSED), (2, 2, BS.LCLOSED)]
    #assert list(ips.closest_more_precise((1, 2, BS.OPEN), use_lectic_order=True, intent=(1, 3, BS.CLOSED))) == [
    #    (1, 1, BS.RCLOSED)]
    #assert list(ips.closest_more_precise((1, 2, BS.OPEN), use_lectic_order=True, intent=(1, 2, BS.CLOSED))) == [
    #    (1, 1, BS.RCLOSED), (2, 2, BS.LCLOSED)]
    assert list(ips.closest_more_precise((0, 0, BS.CLOSED))) == [ips.max_pattern]
    assert list(ips.closest_more_precise((-math.inf, math.inf, BS.OPEN))) == [
        (-math.inf, 5, BS.RCLOSED), (0, math.inf, BS.LCLOSED)]


def test_keys():
    ips = IntervalPS(ndigits=2)
    data = [(1, 3, BS.CLOSED), (1, 2, BS.CLOSED)]
    assert ips.keys((1, 3, BS.CLOSED), data) == [(-math.inf, math.inf, BS.OPEN)]
    assert ips.keys((1, 2, BS.CLOSED), data) == [(-math.inf, 3, BS.OPEN)]

    ips = IntervalPS(ndigits=2, values=[1, 2, 3])
    assert ips.keys((1, 3, BS.CLOSED), data) == [(-math.inf, math.inf, BS.OPEN)]
    assert ips.keys((1, 2, BS.CLOSED), data) == [(-math.inf, 3, BS.OPEN)]
    assert ips.keys((2, 3, BS.CLOSED), data) == [(1, math.inf, BS.OPEN)]

