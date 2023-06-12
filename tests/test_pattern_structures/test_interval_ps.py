from paspailleur.pattern_structures.interval_ps import IntervalPS
from bitarray import frozenbitarray as fbarray


def test_intersect_patterns():
    ips = IntervalPS()
    assert ips.intersect_patterns((0, 1), (2, 3)) == (0, 3)
    assert ips.intersect_patterns((1.5, 2), (1.1, 1.9)) == (1.1, 2)


def test_bin_attributes():
    data = [(0, 1), (2, 3), (1.5, 2)]
    patterns_true = ((0, 3), (1.5, 3), (2, 3), (0, 2), (0, 1), None)
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
    patterns, flags = list(zip(*list(ips.bin_attributes(data))))
    assert patterns == patterns_true
    assert flags == flags_true


def test_is_subpattern():
    ips = IntervalPS()

    assert ips.is_subpattern((0, 1), (0.5, 0.7))
    assert not ips.is_subpattern((0, 6), (3, 10))


def test_n_bin_attributes():
    data = [(0, 1), (2, 3), (1.5, 2)]

    ips = IntervalPS()
    assert ips.n_bin_attributes(data) == 6


