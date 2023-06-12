from paspailleur.pattern_structures import CartesianPS, IntervalPS
from bitarray import frozenbitarray as fbarray


def test_intersect_patterns():
    cps = CartesianPS(basic_structures=[IntervalPS(), IntervalPS()])
    assert cps.intersect_patterns([(1, 2), (3, 5)], [(0, 2), (3, 5)]) == [(0, 2), (3, 5)]


def test_bin_attributes():
    data = [
        [(0, 1), (10, 20)],
        [(1, 2), (10, 20)]
    ]
    patterns_true = (
        (0, (0, 2)), (0, (1, 2)), (0, (0, 1)), (0, None),
        (1, (10, 20)), (1, None),
    )
    flags_true = (
        '11',  # (0, (0, 2))
        '01',  # (0, (1, 2))
        '10',  # (0, (0, 1))
        '00',  # (0, None)
        '11',  # (1, (10, 20)
        '00',  # (1, None))
    )
    flags_true = tuple([fbarray(flag) for flag in flags_true])

    cps = CartesianPS(basic_structures=[IntervalPS(), IntervalPS()])
    patterns, flags = list(zip(*list(cps.bin_attributes(data))))
    assert patterns == patterns_true
    assert flags == flags_true


def test_is_subpattern():
    cps = CartesianPS(basic_structures=[IntervalPS(), IntervalPS()])

    assert cps.is_subpattern([(0, 1), (3, 5)], [(1, 1), (3, 4)])
    assert not cps.is_subpattern([(0, 1), (3, 5)], [(1, 1), (2, 4)])


def test_n_bin_attributes():
    data = [
        [(0, 1), (10, 20)],
        [(1, 2), (10, 20)]
    ]

    cps = CartesianPS(basic_structures=[IntervalPS(), IntervalPS()])
    assert cps.n_bin_attributes(data) == 6


def test_binarize():
    data = [
        [(0, 1), (10, 20)],
        [(1, 2), (10, 20)]
    ]
    patterns_true = [
        (0, (0, 2)), (0, (1, 2)), (0, (0, 1)), (0, None),
        (1, (10, 20)), (1, None),
    ]
    itemsets_true = [
        '101010',
        '110010',
    ]
    itemsets_true = [fbarray(itemset) for itemset in itemsets_true]

    cps = CartesianPS(basic_structures=[IntervalPS(), IntervalPS()])
    patterns, itemsets = cps.binarize(data)
    assert patterns == patterns_true
    assert itemsets == itemsets_true
