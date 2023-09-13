from paspailleur.pattern_structures.set_ps import SuperSetPS, SubSetPS
from bitarray import frozenbitarray as fbarray


def test_intersect_patterns():
    sps = SuperSetPS()
    assert sps.join_patterns({'a', 'b'}, {'c', 'b'}) == {'a', 'b', 'c'}
    assert sps.join_patterns(set(), {'a'}) == {'a'}

    sps = SubSetPS()
    assert sps.join_patterns({'a', 'b'}, {'c', 'd'}) == set()
    assert sps.join_patterns({'a', 'b'}, {'a', 'd'}) == {'a'}
    assert sps.join_patterns(set(), {'a'}) == set()

    assert sps.join_patterns(sps.max_pattern, {'a'}) == {'a'}
    assert sps.join_patterns({'a'}, sps.max_pattern) == {'a'}


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

    sps = SuperSetPS()
    patterns, flags = list(zip(*list(sps.iter_bin_attributes(data))))
    assert patterns == patterns_true
    assert flags == flags_true

    # SubsetPS
    patterns_true = (set(), {'a'}, {'b'}, {'c'}, {'a', 'b', 'c'})
    flags_true = (
        '111',  # set(),
        '101',  # {'a'}
        '010',  # {'b'}
        '001',  # {'c'}
        '000',  # {'a','b','c'}
    )
    flags_true = tuple([fbarray(flag) for flag in flags_true])

    sps = SubSetPS()
    patterns, flags = list(zip(*list(sps.iter_bin_attributes(data))))
    assert patterns == patterns_true
    assert flags == flags_true


def test_is_subpattern():
    sps = SuperSetPS()

    assert sps.is_less_precise({'a', 'b', 'c'}, {'a'})
    assert not sps.is_less_precise({'a'}, {'c'})

    sps = SubSetPS()
    assert sps.is_less_precise({'a'}, {'a', 'b', 'c'})
    assert not sps.is_less_precise({'a'}, {'c'})


def test_n_bin_attributes():
    data = [{'a'}, {'b'}, {'a', 'c'}]

    sps = SuperSetPS()
    assert sps.n_bin_attributes(data) == 8

    sps = SubSetPS()
    assert sps.n_bin_attributes(data) == 5


def test_intent():
    data = [{'a'}, {'b'}, {'a', 'c'}]

    sps = SuperSetPS()
    assert sps.intent(data) == {'a', 'b', 'c'}

    sps = SubSetPS()
    assert sps.intent(data) == set()


def test_extent():
    data = [{'a'}, {'b'}, {'a', 'c'}]

    sps = SuperSetPS()
    assert list(sps.extent(data, {'a', 'c'})) == [0, 2]

    sps = SubSetPS()
    assert list(sps.extent(data, {'a'})) == [0, 2]
