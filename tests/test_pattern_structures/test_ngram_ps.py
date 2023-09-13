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


def test_ngram_iter_bin_attributes():
    patterns = [{('hello', 'world')}, {('hello', 'there')}, {('hi',)}, set()]

    ps = NgramPS()
    subpatterns, extents = zip(*ps.iter_bin_attributes(patterns))
    assert subpatterns[0] == set()
    assert extents[0].all()
    assert not extents[-1].any()

    extents_true = {fbarray('1111'), fbarray('1100'),
                    fbarray('1000'), fbarray('0100'), fbarray('0010'), fbarray('0000')}
    assert set(extents) == extents_true

    subpatterns, extents = zip(*ps.iter_bin_attributes(patterns, min_support=2))
    assert subpatterns == (set(), ('hello',))
    assert extents == (fbarray('1111'), fbarray('1100'))


def test_ngram_n_bin_attributes():
    patterns = [{('hello', 'world')}, {('hello', 'there')}, {('hi',)}, set()]
    ps = NgramPS()
    bin_attrs = list(ps.iter_bin_attributes(patterns))
    assert ps.n_bin_attributes(patterns) == len(bin_attrs)
