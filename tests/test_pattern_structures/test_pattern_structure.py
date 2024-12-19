from paspailleur.pattern_structures.pattern_structure import PatternStructure
from paspailleur.pattern_structures.pattern import Pattern

from bitarray import frozenbitarray as fbarray


def test_init():
    ps = PatternStructure()

    ptrn = ps.PatternType({1, 2, 3})
    assert type(ptrn) == Pattern
    assert ptrn.value == {1, 2, 3}


def test_fit():
    patterns = [Pattern(frozenset({1, 2, 3})), Pattern(frozenset({0, 4})), Pattern(frozenset({1, 2, 4}))]
    context = {'a': patterns[0], 'b': patterns[1], 'c': patterns[2]}

    ps = PatternStructure()
    ps.fit(context)
    assert ps._object_names == ['a', 'b', 'c']
    object_irreducibles = {patterns[0]: fbarray('100'), patterns[1]: fbarray('010'), patterns[2]: fbarray('001')}
    assert ps._object_irreducibles == object_irreducibles


def test_extent():
    patterns = [Pattern(frozenset({1, 2, 3})), Pattern(frozenset({0, 4})), Pattern(frozenset({1, 2, 4}))]
    context = {'a': patterns[0], 'b': patterns[1], 'c': patterns[2]}

    ps = PatternStructure()
    ps.fit(context)
    assert ps.extent(patterns[0]) == {'a'}
    assert ps.extent(patterns[1]) == {'b'}
    assert ps.extent(patterns[2]) == {'c'}
    assert ps.extent(Pattern(frozenset({4}))) == {'b', 'c'}
    assert ps.extent(Pattern(frozenset())) == {'a', 'b', 'c'}
    assert ps.extent(Pattern(frozenset({1, 2, 3, 4}))) == set()

    assert ps.extent(patterns[0], return_bitarray=True) == fbarray('100')
    assert ps.extent(patterns[1], return_bitarray=True) == fbarray('010')
    assert ps.extent(patterns[2], return_bitarray=True) == fbarray('001')
    assert ps.extent(Pattern(frozenset({4})), return_bitarray=True) == fbarray('011')
    assert ps.extent(Pattern(frozenset()), return_bitarray=True) == fbarray('111')
    assert ps.extent(Pattern(frozenset({1, 2, 3, 4})), return_bitarray=True) == fbarray('000')


def test_intent():
    patterns = [Pattern(frozenset({1, 2, 3})), Pattern(frozenset({0, 4})), Pattern(frozenset({1, 2, 4}))]
    context = {'a': patterns[0], 'b': patterns[1], 'c': patterns[2]}

    ps = PatternStructure()
    ps.fit(context)
    assert ps.intent({'a'}) == patterns[0]
    assert ps.intent(['b']) == patterns[1]
    assert ps.intent(fbarray('001')) == patterns[2]
    assert ps.intent({'a', 'b'}) == Pattern(frozenset())
    assert ps.intent({'a', 'c'}) == Pattern(frozenset({1, 2}))
    assert ps.intent({'b', 'c'}) == Pattern(frozenset({4}))
    assert ps.intent([]) == Pattern(frozenset({0, 1, 2, 3, 4}))
