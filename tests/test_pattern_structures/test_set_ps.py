from paspailleur.pattern_structures.set_ps import SetPS
from bitarray import frozenbitarray as fbarray


def test_intersect_patterns():
    sps = SetPS()
    assert sps.intersect_patterns({'a', 'b'}, {'c', 'b'}) == {'a', 'b', 'c'}
    assert sps.intersect_patterns(set(), {'a'}) == {'a'}


def test_bin_attributes():
    data = [{'a'}, {'b'}, {'a', 'c'}]
    patterns_true = ({'a', 'b', 'c'}, {'a', 'b'}, {'a', 'c'}, {'b', 'c'}, {'a'}, {'b'}, {'c'}, set())
    flags_true = (
        '111',  # {'a', 'b', 'c'},
        '110',  # {'a', 'b'},
        '101',  # {'a', 'c'},
        '010',  # {'b', 'c'},
        '100',  # {'a'},
        '010',  # {'b'},
        '000',  # {'c'},
        '000',  # set(),
    )
    flags_true = tuple([fbarray(flag) for flag in flags_true])

    sps = SetPS()
    patterns, flags = list(zip(*list(sps.bin_attributes(data))))
    assert patterns == patterns_true
    assert flags == flags_true


def test_is_subpattern():
    sps = SetPS()

    assert sps.is_subpattern({'a', 'b', 'c'}, {'a'})
    assert not sps.is_subpattern({'a'}, {'c'})


def test_n_bin_attributes():
    data = [{'a'}, {'b'}, {'a', 'c'}]

    sps = SetPS()
    assert sps.n_bin_attributes(data) == 8


def test_intent():
    data = [{'a'}, {'b'}, {'a', 'c'}]

    sps = SetPS()
    assert sps.intent(data) == {'a', 'b', 'c'}


def test_extent():
    data = [{'a'}, {'b'}, {'a', 'c'}]

    sps = SetPS()
    assert list(sps.extent({'a', 'c'}, data)) == [0, 2]
