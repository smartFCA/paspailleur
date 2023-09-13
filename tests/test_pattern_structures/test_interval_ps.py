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
    patterns, flags = list(zip(*list(ips.iter_bin_attributes(data))))
    assert patterns == patterns_true
    assert flags == flags_true


def test_is_subpattern():
    ips = IntervalPS()

    assert ips.is_less_precise((0, 1), (0.5, 0.7))
    assert not ips.is_less_precise((0, 6), (3, 10))


def test_n_bin_attributes():
    data = [(0, 1), (2, 3), (1.5, 2)]

    ips = IntervalPS()
    assert ips.n_bin_attributes(data) == 6


def test_intent():
    data = [(0, 1), (2, 3), (1.5, 2)]

    ips = IntervalPS()
    assert ips.intent(data) == (0, 3)


def test_extent():
    data = [(0, 1), (2, 3), (1.5, 2)]

    ips = IntervalPS()
    assert list(ips.extent(data, (1.5, 3))) == [1, 2]
