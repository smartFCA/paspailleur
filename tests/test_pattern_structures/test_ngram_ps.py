import pytest

from paspailleur.pattern_structures.ngram_ps import NgramPS

from bitarray import frozenbitarray as fbarray

##################
#  Test NgramPS  #
##################


def test_ngram_init():
    ps = NgramPS()
    assert ps.min_n == 1

    assert ps.max_pattern == {('<MAX_NGRAM>',)}

    ps = NgramPS(10)
    assert ps.min_n == 10


def test_ngram_preprocess():
    ps = NgramPS()
    texts = ['hello world', 'hello there', 'hi', '']
    patterns = [{('hello', 'world')}, {('hello', 'there')}, {('hi',)}, set()]
    assert list(ps.preprocess_data(texts)) == patterns

    ps.min_n = 10
    assert list(ps.preprocess_data(texts)) == [set(), set(), set(), set()]


def test_ngram_is_less_precise():
    a = {('hello', 'world'), ('who', 'is', 'there')}
    b = {('hello', 'there')}
    join = {('hello',), ('there',)}

    ps = NgramPS()
    assert ps.is_less_precise(set(), a) is True
    assert ps.is_less_precise(join, a) is True
    assert ps.is_less_precise(join, b) is True
    assert ps.is_less_precise(a, b) is False
    assert ps.is_less_precise(b, a) is False

    assert ps.is_less_precise({('word',)}, {('word_suffix',)}) is False

    assert ps.is_less_precise({('hello', 'world'), ('who', 'is', 'there')}, {('hello', 'world', 'who', 'is', 'there')})


def test_ngram_join_patterns():
    a = {('hello', 'world'), ('who', 'is', 'there')}
    b = {('hello', 'there')}
    join = {('hello',), ('there',)}

    ps = NgramPS()
    assert ps.join_patterns(a, b) == join

    ps = NgramPS(min_n=2)
    assert ps.join_patterns(a, b) == set()

    ps = NgramPS()
    join = ps.join_patterns({tuple('abcd')}, {tuple('cxaz')})
    assert join == {tuple('a'), tuple('c')}

    join = ps.join_patterns({tuple('ab'), tuple('bc')}, {tuple('abc')})
    assert join == {tuple('ab'), tuple('bc')}

    assert ps.join_patterns(ps.max_pattern, a) == a
    assert ps.join_patterns(a, ps.max_pattern) == a

    assert ps.join_patterns({('a', '_', 'ab', 'c')}, {('a', '__', 'ab', 'c')}) == frozenset({('a',), ('ab', 'c')})

    assert ps.join_patterns({('a', 'b')}, {('a',)}) == {('a',)}


def test_ngram_iter_bin_attributes():
    patterns = [{('hello', 'world')}, {('hello', 'there')}, {('hi',)}, set()]

    ps = NgramPS()
    subpatterns, extents = zip(*ps.iter_attributes(patterns))
    assert subpatterns[0] == set()
    assert extents[0].all()
    assert not extents[-1].any()

    extents_true = {fbarray('1111'), fbarray('1100'),
                    fbarray('1000'), fbarray('0100'), fbarray('0010'), fbarray('0000')}
    assert set(extents) == extents_true

    subpatterns, extents = zip(*ps.iter_attributes(patterns, min_support=2))
    assert subpatterns == (set(), ('hello',))
    assert extents == (fbarray('1111'), fbarray('1100'))


def test_ngram_n_bin_attributes():
    patterns = [{('hello', 'world')}, {('hello', 'there')}, {('hi',)}, set()]
    ps = NgramPS()
    bin_attrs = list(ps.iter_attributes(patterns))
    assert ps.n_attributes(patterns) == len(bin_attrs)


def test_verbalize():
    ps = NgramPS()
    assert ps.verbalize({('hello', 'world'), ('hi',)}) == 'hello world; hi'
    assert ps.verbalize({('hello', 'world'), ('hi',)}, ngram_separator='\n') == 'hello world\nhi'
    assert ps.verbalize(set()) == '∅'


def test_closest_less_precise():
    ps = NgramPS()
    assert set(ps.closest_less_precise(frozenset())) == set()
    assert set(ps.closest_less_precise(frozenset({tuple('a')}))) == {frozenset()}
    assert set(ps.closest_less_precise(frozenset({tuple('ab'), tuple('x')}))) == {
        frozenset({tuple('ab')}), frozenset({tuple('a'), tuple('b'), tuple('x')}),
    }
    assert set(ps.closest_less_precise(frozenset({tuple('ab'), tuple('ac')}), use_lectic_order=True)) == {
        frozenset({tuple('ab'), tuple('c')}), frozenset({tuple('ac'), tuple('b')})
    }


def test_closest_more_precise():
    ps = NgramPS()
    assert set(ps.closest_more_precise(frozenset())) == set()

    result = set(ps.closest_more_precise(frozenset({('a',), ('b',)})))
    result_true = {
        frozenset({('a', 'a'), ('b',)}), frozenset({('a', ), ('b', 'b')}),
        frozenset({('a', 'b')}), frozenset({('b', 'a')})
    }
    assert result == result_true

    result = set(ps.closest_more_precise(frozenset(), vocabulary={'a', 'b', 'c'}))
    result_true = {frozenset({('a',)}), frozenset({('b',)}), frozenset({('c',)})}
    assert result == result_true

    result = set(ps.closest_more_precise(frozenset({('a', 'b'), ('c',)})))
    result_true = {
        frozenset({('a', 'b', 'a'), ('c',)}), frozenset({('a', 'a', 'b'), ('c',)}),
        frozenset({('a', 'b'), ('c', 'a')}), frozenset({('a', 'b'), ('a', 'c',)}),
        frozenset({('a', 'b', 'b'), ('c',)}), frozenset({('b', 'a', 'b'), ('c',)}),
        frozenset({('a', 'b'), ('c', 'b')}), frozenset({('a', 'b'), ('b', 'c')}),
        frozenset({('a', 'b', 'c')}), frozenset({('c', 'a', 'b')}),
        frozenset({('a', 'b'), ('c', 'c')}),
    }
    assert result == result_true

    result = list(ps.closest_more_precise(frozenset({('a', 'b'), ('c',)}), use_lectic_order=True))
    result_true = {
        frozenset({('a', 'b', 'a'), ('c',)}), frozenset({('a', 'b'), ('c', 'a')}),
        frozenset({('a', 'b', 'b'), ('c',)}), frozenset({('a', 'b'), ('c', 'b')}),
        frozenset({('a', 'b', 'c')}), frozenset({('a', 'b'), ('c', 'c')}),
    }
    assert len(result) == len(result_true)
    assert set(result) == result_true


def test_keys():
    ps = NgramPS()
    data = [
        frozenset({('a', 'b'), ('c',)}),
        frozenset({('c', 'a')}),
        frozenset({('a',)})
    ]
    assert ps.keys(frozenset({('a',)}), data) == [frozenset()]
    assert set(ps.keys(frozenset({('a', 'b'), ('c',)}), data)) == {frozenset({('b',)})}
    assert ps.keys(frozenset({('c', 'a')}), data) == [frozenset({('c', 'a')})]
    assert ps.keys(frozenset({('a',), ('c',)}), data) == [frozenset({('c',)})]
