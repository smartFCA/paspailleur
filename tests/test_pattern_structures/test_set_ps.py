import pytest

from paspailleur.pattern_structures.set_ps import DisjunctiveSetPS, ConjunctiveSetPS, ValuesUniverseUndefined
from bitarray import frozenbitarray as fbarray


def test_intersect_patterns():
    sps = DisjunctiveSetPS()
    assert sps.join_patterns({'a', 'b'}, {'c', 'b'}) == {'a', 'b', 'c'}
    assert sps.join_patterns(set(), {'a'}) == {'a'}

    sps = ConjunctiveSetPS()
    assert sps.join_patterns({'a', 'b'}, {'c', 'd'}) == set()
    assert sps.join_patterns({'a', 'b'}, {'a', 'd'}) == {'a'}
    assert sps.join_patterns(set(), {'a'}) == set()

    assert sps.join_patterns(sps.max_pattern, {'a'}) == {'a'}
    assert sps.join_patterns({'a'}, sps.max_pattern) == {'a'}


def test_bin_attributes():
    data = [{'a'}, {'b'}, {'a', 'c'}]

    sps = DisjunctiveSetPS()
    patterns_true = ({'a', 'b'}, {'a', 'c'}, {'b', 'c'})
    flags_true = (
        '111',  # {'a', 'b'},
        '101',  # {'a', 'c'},
        '011',  # {'b', 'c'},
    )
    flags_true = tuple([fbarray(flag) for flag in flags_true])

    patterns, flags = list(zip(*list(sps.iter_attributes(data))))
    assert patterns == patterns_true
    assert flags == flags_true

    patterns, flags = list(zip(*list(sps.iter_attributes(data, min_support=0.5))))
    assert set(flags) == {flg for flg in flags_true if flg.count() >= 2}

    patterns, flags = list(zip(*list(sps.iter_attributes(data, min_support=2))))
    assert set(flags) == {flg for flg in flags_true if flg.count() >= 2}

    # SubsetPS
    sps = ConjunctiveSetPS()
    patterns_true = ({'a'}, {'b'}, {'c'})
    flags_true = (
        '101',  # {'a'},
        '010',  # {'b'},
        '001',  # {'c'},
    )
    flags_true = tuple([fbarray(flag) for flag in flags_true])

    patterns, flags = list(zip(*list(sps.iter_attributes(data))))
    assert patterns == patterns_true
    assert flags == flags_true

    patterns, flags = list(zip(*list(sps.iter_attributes(data, min_support=0.5))))
    assert set(flags) == {flg for flg in flags_true if flg.count() >= 2}

    patterns, flags = list(zip(*list(sps.iter_attributes(data, min_support=2))))
    assert set(flags) == {flg for flg in flags_true if flg.count() >= 2}


def test_is_subpattern():
    sps = DisjunctiveSetPS()

    assert sps.is_less_precise({'a', 'b', 'c'}, {'a'})
    assert not sps.is_less_precise({'a'}, {'c'})

    sps = ConjunctiveSetPS()
    assert sps.is_less_precise({'a'}, {'a', 'b', 'c'})
    assert not sps.is_less_precise({'a'}, {'c'})


def test_n_bin_attributes():
    data = [{'a'}, {'b'}, {'a', 'c'}]

    sps = DisjunctiveSetPS()
    assert sps.n_attributes(data) == 3

    sps = ConjunctiveSetPS()
    assert sps.n_attributes(data) == 3


def test_intent():
    data = [{'a'}, {'b'}, {'a', 'c'}]

    sps = DisjunctiveSetPS()
    assert sps.intent(data) == {'a', 'b', 'c'}

    sps = ConjunctiveSetPS()
    assert sps.intent(data) == set()


def test_extent():
    data = [{'a'}, {'b'}, {'a', 'c'}]

    sps = DisjunctiveSetPS()
    assert list(sps.extent(data, {'a', 'c'})) == [0, 2]

    sps = ConjunctiveSetPS()
    assert list(sps.extent(data, {'a'})) == [0, 2]


def test_preprocess_data():
    data = [{'a'},  1, [3, 'x']]

    sps = DisjunctiveSetPS()
    assert list(sps.preprocess_data(data)) == [frozenset({'a'}), frozenset({1}), frozenset({3, 'x'})]

    sps = ConjunctiveSetPS()
    assert list(sps.preprocess_data(data)) == [frozenset({'a'}), frozenset({1}), frozenset({3, 'x'})]


def test_verbalize():
    sps = ConjunctiveSetPS()
    assert sps.verbalize({'a', 'b', 'c'}) == 'a, b, c'
    assert sps.verbalize({'a', 'bcde', 'f'}, add_curly_braces=True) == '{a, bcde, f}'
    assert sps.verbalize(set()) == '∅'
    assert sps.verbalize(set(), add_curly_braces=True) == '{}'

    sps = DisjunctiveSetPS()
    assert sps.verbalize({'a', 'b', 'c'}) == 'a, b, c'
    assert sps.verbalize({'a', 'bcde', 'f'}, add_curly_braces=True) == '{a, bcde, f}'
    assert sps.verbalize(set()) == '∅'
    assert sps.verbalize(set(), add_curly_braces=True) == '{}'


def test_closest_more_precise():
    dps = DisjunctiveSetPS(all_values={'a', 'b', 'c'})
    assert set(dps.closest_more_precise(frozenset({'a', 'b'}))) == {frozenset({'a'}), frozenset({'b'})}
    assert set(dps.closest_more_precise(frozenset({'a', 'b'}), use_lectic_order=True)) == set()
    assert set(dps.closest_more_precise(frozenset({'a', 'c'}), use_lectic_order=True, intent=frozenset({'a', 'b', 'c'})))\
           == {frozenset({'a'})}
    assert set(dps.closest_more_precise(frozenset())) == set()

    dps = DisjunctiveSetPS()
    with pytest.raises(ValuesUniverseUndefined):
        list(dps.closest_more_precise(frozenset({'a', 'b'}), use_lectic_order=True))

    cps = ConjunctiveSetPS(all_values={'a', 'b', 'c'})
    assert set(cps.closest_more_precise(frozenset({'a'}))) == {frozenset({'a', 'b'}), frozenset({'a', 'c'})}
    assert set(cps.closest_more_precise(frozenset({'b'}), use_lectic_order=True)) == {frozenset({'b', 'c'})}
    assert set(cps.closest_more_precise(frozenset({'a', 'b', 'c'}))) == set()

    cps = ConjunctiveSetPS()
    with pytest.raises(ValuesUniverseUndefined):
        list(cps.closest_more_precise(frozenset({'a'})))


def test_closest_less_precise():
    dps = DisjunctiveSetPS(all_values={'a', 'b', 'c'})
    assert set(dps.closest_less_precise(frozenset({'a', 'b'}))) == {frozenset({'a', 'b', 'c'})}
    assert set(dps.closest_less_precise(frozenset({'b'}))) == {frozenset({'a', 'b'}), frozenset({'b', 'c'})}
    assert set(dps.closest_less_precise(frozenset({'b'}), use_lectic_order=True)) == {frozenset({'b', 'c'})}

    dps = DisjunctiveSetPS()
    with pytest.raises(ValuesUniverseUndefined):
        list(dps.closest_less_precise(frozenset({'b'})))
    with pytest.raises(ValuesUniverseUndefined):
        list(dps.closest_more_precise(frozenset({'a', 'b'}), use_lectic_order=True))

    cps = ConjunctiveSetPS(all_values={'a', 'b', 'c'})
    assert set(cps.closest_less_precise(frozenset({'a', 'b'}))) == {frozenset({'a'}), frozenset({'b'})}
    assert set(cps.closest_less_precise(frozenset({'a', 'b'}), use_lectic_order=True))\
           == set()
    assert set(cps.closest_less_precise(frozenset({'a', 'c'}), use_lectic_order=True, intent=frozenset({'a', 'b', 'c'})))\
           == {frozenset({'a'})}
    assert set(cps.closest_less_precise(frozenset())) == set()

    cps = ConjunctiveSetPS()
    with pytest.raises(ValuesUniverseUndefined):
        list(cps.closest_less_precise(frozenset({'a', 'b'}), use_lectic_order=True))


def test_keys():
    dps = DisjunctiveSetPS(all_values={'a', 'b', 'c'})
    data = [frozenset({'a', 'b'}), frozenset({'b'})]

    assert dps.keys(frozenset({'a', 'b'}), data) == [frozenset({'a', 'b', 'c'})]
    assert dps.keys(frozenset({'b'}), data) == [frozenset({'b', 'c'})]

    cps = ConjunctiveSetPS(all_values={'a', 'b', 'c'})
    assert cps.keys(frozenset({'a', 'b'}), data) == [frozenset({'a'})]
    assert cps.keys(frozenset({'b'}), data) == [frozenset()]
    assert cps.keys(frozenset({'a', 'b', 'c'}), data) == [frozenset({'c'})]
