from collections import OrderedDict
from collections.abc import Iterator
from typing import Literal, Self

import pytest

from paspailleur.pattern_structures.pattern_structure import PatternStructure
from paspailleur.pattern_structures.pattern import Pattern
from paspailleur.pattern_structures import built_in_patterns as bip
from paspailleur.algorithms import mine_equivalence_classes as mec

from bitarray import frozenbitarray as fbarray, bitarray


def test_init():
    ps = PatternStructure()
    assert ps.PatternType is None

    ps = PatternStructure(Pattern)
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
    assert ps._atomic_patterns is None

    class APattern(Pattern):  # short for atomised pattern
        def atomise(self, atoms_configuration: Literal['min', 'max'] = 'min') -> set[Self]:
            return {self.__class__(frozenset([v])) for v in self.value}

    patterns = [APattern(p.value) for p in patterns]
    context = {'a': patterns[0], 'b': patterns[1], 'c': patterns[2]}
    ps.fit(context)
    assert ps._atomic_patterns is not None


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
    context = dict(zip('abc', patterns))

    ps = PatternStructure()
    ps.fit(context)
    assert ps.intent({'a'}) == patterns[0]
    assert ps.intent(['b']) == patterns[1]
    assert ps.intent(fbarray('001')) == patterns[2]
    assert ps.intent({'a', 'b'}) == Pattern(frozenset())
    assert ps.intent({'a', 'c'}) == Pattern(frozenset({1, 2}))
    assert ps.intent({'b', 'c'}) == Pattern(frozenset({4}))
    assert ps.intent([]) == Pattern(frozenset({0, 1, 2, 3, 4}))

    context = {
        'Stewart Island': {'Hiking', 'Observing Nature', 'Sightseeing Flights'},
        'Fjordland NP': {'Hiking', 'Observing Nature', 'Sightseeing Flights'},
        'Invercargill': {'Hiking', 'Observing Nature', 'Sightseeing Flights'},
        'Milford Sound': {'Hiking', 'Observing Nature', 'Sightseeing Flights'},
        'MT. Aspiring NP': {'Hiking', 'Observing Nature', 'Sightseeing Flights'},
        'Te Anau': {'Hiking', 'Jet Boating', 'Observing Nature', 'Sightseeing Flights'},
        'Dunedin': {'Hiking', 'Observing Nature', 'Sightseeing Flights'},
        'Oamaru': {'Hiking', 'Observing Nature'},
        'Queenstown': {'Bungee Jumping', 'Hiking', 'Jet Boating', 'Parachute Gliding', 'Sightseeing Flights', 'Skiing',
                       'Wildwater Rafting'},
        'Wanaka': {'Bungee Jumping', 'Hiking', 'Jet Boating', 'Parachute Gliding', 'Sightseeing Flights',
                   'Skiing', 'Wildwater Rafting'},
        'Otago Peninsula': {'Hiking', 'Observing Nature'},
        'Haast': {'Hiking', 'Observing Nature'},
        'Catlins': {'Hiking', 'Observing Nature'}
    }
    context = {obj: Pattern(frozenset(descr)) for obj, descr in context.items()}

    ps.fit(context)
    assert ps.intent(set(context)) == Pattern(frozenset({'Hiking'}))
    intent = ps.intent({'Fjordland NP'})
    assert intent == Pattern(frozenset({'Hiking', 'Observing Nature', 'Sightseeing Flights'}))


def test_min_pattern():
    patterns = [Pattern(frozenset({1, 2, 3})), Pattern(frozenset({0, 4})), Pattern(frozenset({1, 2, 4}))]
    context = dict(zip('abc', patterns))

    ps = PatternStructure()
    ps.fit(context)
    assert ps.min_pattern == Pattern(frozenset())

    patterns = [Pattern(frozenset({1, 2, 3})), Pattern(frozenset({0, 1, 4})), Pattern(frozenset({1, 2, 4}))]
    context = dict(zip('abc', patterns))
    ps.fit(context)
    assert ps.min_pattern == Pattern(frozenset({1}))

    class BPattern(Pattern):  # short for bounded pattern
        @property
        def min_pattern(self):
            return self.__class__(frozenset())

    patterns = [BPattern(frozenset({1, 2, 3})), BPattern(frozenset({0, 1, 4})), BPattern(frozenset({1, 2, 4}))]
    context = dict(zip('abc', patterns))
    ps.fit(context)
    assert ps.min_pattern == BPattern(frozenset())


def test_max_pattern():
    patterns = [Pattern(frozenset({1, 2, 3})), Pattern(frozenset({0, 4})), Pattern(frozenset({1, 2, 4}))]
    context = dict(zip('abc', patterns))

    ps = PatternStructure()
    ps.fit(context)
    assert ps.max_pattern == Pattern(frozenset(range(5)))

    class BPattern(Pattern):  # short for bounded pattern
        @property
        def max_pattern(self):
            return self.__class__(frozenset(range(10)))

    patterns = [BPattern(frozenset({1, 2, 3})), BPattern(frozenset({0, 1, 4})), BPattern(frozenset({1, 2, 4}))]
    context = dict(zip('abc', patterns))
    ps.fit(context)
    assert ps.max_pattern == BPattern(frozenset(range(10)))


def test_atomic_patterns():
    class APattern(Pattern):  # short for atomised pattern
        def atomise(self, atoms_configuration: Literal['min', 'max'] = 'min') -> set[Self]:
            return {self.__class__(frozenset([v])) for v in self.value}

    patterns = [APattern(frozenset({1, 2, 3})), APattern(frozenset({0, 4})), APattern(frozenset({1, 2, 4}))]
    atomic_patterns_true = OrderedDict([
        (APattern(frozenset({2})), fbarray('101')),  # supp: 2
        (APattern(frozenset({1})), fbarray('101')),  # supp: 2
        (APattern(frozenset({4})), fbarray('011')),  # supp: 2
        (APattern(frozenset({3})), fbarray('100')),  # supp: 1
        (APattern(frozenset({0})), fbarray('010')),  # supp: 1
    ])
    context = dict(zip('abc', patterns))

    ps = PatternStructure()
    assert ps._atomic_patterns is None

    ps.fit(context, compute_atomic_patterns=False)
    ps.init_atomic_patterns()
    assert ps._atomic_patterns == atomic_patterns_true
    assert ps._atomic_patterns_order == [fbarray('0'*len(atomic_patterns_true))] * len(atomic_patterns_true)

    atomic_patterns_true_verb = OrderedDict([(k, {'abc'[g] for g in ext_ba.search(True)})
                                             for k, ext_ba in atomic_patterns_true.items()])
    assert ps.atomic_patterns == atomic_patterns_true_verb


def test_iter_premaximal_patterns():
    patterns = [Pattern(frozenset({1, 2, 3})), Pattern(frozenset({4})), Pattern(frozenset({1, 2, 4}))]
    context = dict(zip('abc', patterns))

    premaximal_true = [patterns[0], patterns[2]]
    premaximal_true_set = {patterns[0]: set('a'), patterns[2]: set('c')}
    premaximal_true_ba = {patterns[0]: fbarray('100'), patterns[2]: fbarray('001')}
    ps = PatternStructure()
    ps.fit(context)
    assert dict(ps.iter_premaximal_patterns()) == premaximal_true_set
    assert list(ps.iter_premaximal_patterns(return_extents=False)) == premaximal_true
    assert dict(ps.iter_premaximal_patterns(return_extents=True, return_bitarrays=True)) == premaximal_true_ba


def test_premaximal_patterns():
    patterns = [Pattern(frozenset({1, 2, 3})), Pattern(frozenset({4})), Pattern(frozenset({1, 2, 4}))]
    context = dict(zip('abc', patterns))

    premaximal_true = {patterns[0]: set('a'), patterns[2]: set('c')}

    ps = PatternStructure()
    ps.fit(context)
    assert ps.premaximal_patterns == premaximal_true


def test_builtin_atomic_patterns():
    ps = PatternStructure()

    patterns = [bip.ItemSetPattern({1, 2, 3}), bip.ItemSetPattern({0, 4}), bip.ItemSetPattern({1, 2, 4})]
    context = dict(zip('abc', patterns))
    atomic_patterns_true_verb = [
        ({2}, 'ac'),  # supp: 2
        ({1}, 'ac'),  # supp: 2
        ({4}, 'bc'),  # supp: 2
        ({3}, 'a'),  # supp: 1
        ({0}, 'b'),  # supp: 1
    ]
    atomic_patterns_true_verb = OrderedDict([(bip.ItemSetPattern(ptrn), set(ext))
                                             for ptrn, ext in atomic_patterns_true_verb])
    ps.fit(context)
    assert ps.atomic_patterns == atomic_patterns_true_verb

    atomic_patterns_order_true = {(2,): [], (1,): [], (4,): [], (3,): [], (0,): []}
    atomic_patterns_order_true = {bip.ItemSetPattern(k): {bip.ItemSetPattern(v) for v in vs}
                                  for k, vs in atomic_patterns_order_true.items()}
    assert ps.atomic_patterns_order == atomic_patterns_order_true

    patterns = [bip.IntervalPattern('[0, 10]'), bip.IntervalPattern('(2, 11]'), bip.IntervalPattern('[5, 10]')]
    context = dict(zip('abc', patterns))
    atomic_patterns_true_verb = [
        ('>= 0.0', 'abc'), ('> 0.0', 'bc'),
        ('<= 11.0', 'abc'), ('< 11.0', 'ac'),
        ('<= 10.0', 'ac'),
         ('>= 2.0', 'bc'), ('> 2.0', 'bc'),
        ('>= 5.0', 'c'),
    ]

    atomic_patterns_order_true = {
        '>= 0.0': ['> 0.0', '>= 5.0', '> 2.0', '>= 2.0'],
        '<= 11.0': ['<= 10.0', '< 11.0'], '< 11.0': ['<= 10.0'],
        '<= 10.0': [],
        '> 0.0': ['>= 2.0','>= 5.0', '> 2.0'],
        '>= 2.0': ['>= 5.0', '> 2.0'], '> 2.0': ['>= 5.0'],
        '>= 5.0': [],
    }

    atomic_patterns_true_verb = OrderedDict([(bip.IntervalPattern(ptrn), set(ext))
                                             for ptrn, ext in atomic_patterns_true_verb])
    atomic_patterns_order_true = {bip.IntervalPattern(k): {bip.IntervalPattern(v) for v in vs}
                                  for k, vs in atomic_patterns_order_true.items()}

    ps = PatternStructure()
    ps.fit(context)
    with pytest.raises(AssertionError):
        ps.atomic_patterns  # because the basic IntervalPattern is not atomisable

    class BoundedIntervalPattern(bip.IntervalPattern):
        BoundsUniverse = (0, 2, 5, 10, 11)
    atomic_patterns_true_verb = OrderedDict([(BoundedIntervalPattern(ptrn), set(ext))
                                             for ptrn, ext in atomic_patterns_true_verb.items()])
    atomic_patterns_order_true = {BoundedIntervalPattern(k): {BoundedIntervalPattern(v) for v in vs}
                                  for k, vs in atomic_patterns_order_true.items()}
    context = dict(zip('abc', [BoundedIntervalPattern(p) for p in patterns]))
    ps = PatternStructure()
    ps.fit(context)
    assert set(ps.atomic_patterns) == set(atomic_patterns_true_verb)
    assert all([ps.atomic_patterns[k] == atomic_patterns_true_verb[k] for k in atomic_patterns_true_verb])
    assert all([ps.atomic_patterns_order[k] == atomic_patterns_order_true[k] for k in atomic_patterns_true_verb])
    assert all([
        len(ps.atomic_patterns[a]) >= len(ps.atomic_patterns[b])
        for a, b in zip(list(ps.atomic_patterns), list(ps.atomic_patterns)[1:])
    ])


    patterns = [['hello world', 'who is there'], ['hello world'], ['world is there']]
    patterns = [bip.NgramSetPattern(ngram) for ngram in patterns]
    context = dict(zip('abc', patterns))
    atomic_patterns_true_verb = [
        ('world', 'abc'),
        ('hello', 'ab'), ('hello world', 'ab'),
        ('is', 'ac'), ('there', 'ac'), ('is there', 'ac'),
        ('who', 'a'), ('who is', 'a'), ('who is there', 'a'),
        ('world is', 'c'), ('world is there', 'c'),
    ]
    atomic_patterns_true_verb = OrderedDict([(bip.NgramSetPattern([ptrn]), set(ext))
                                             for ptrn, ext in atomic_patterns_true_verb])

    ps = PatternStructure()
    ps.fit(context)
    assert set(ps.atomic_patterns) == set(atomic_patterns_true_verb)
    assert all(len(ps.atomic_patterns[prev]) >= len(ps.atomic_patterns[next])
               for prev, next in zip(ps.atomic_patterns, list(ps.atomic_patterns)[1:]))

    atomic_patterns_order_true = {
        'world': ['hello world', 'world is', 'world is there'],
        'hello': ['hello world'],
        'hello world': [],
        'is': ['is there', 'who is', 'who is there', 'world is', 'world is there'],
        'there': ['is there', 'who is there', 'world is there'],
        'is there': ['who is there', 'world is there'],
        'who': ['who is', 'who is there'],
        'who is': ['who is there'],
        'who is there': [],
        'world is': ['world is there'],
        'world is there': [],
    }
    atomic_patterns_order_true = {bip.NgramSetPattern([k]): {bip.NgramSetPattern([v]) for v in vs}
                                  for k, vs in atomic_patterns_order_true.items()}
    assert ps.atomic_patterns_order == atomic_patterns_order_true


def test_builtin_premaximal_patterns():
    patterns = [bip.ItemSetPattern({1, 2, 3}), bip.ItemSetPattern({4}), bip.ItemSetPattern({1, 2, 4})]
    context = dict(zip('abc', patterns))
    ps = PatternStructure()
    ps.fit(context)
    assert ps.premaximal_patterns == {patterns[0]: {'a'}, patterns[2]: {'c'}}

    patterns = [bip.IntervalPattern('[0, 10]'), bip.IntervalPattern('(2, 11]'), bip.IntervalPattern('[5, 10]')]
    context = dict(zip('abc', patterns))
    ps = PatternStructure()
    ps.fit(context)
    assert ps.premaximal_patterns == {patterns[2]: {'c'}}

    patterns = [['hello world', 'who is there'], ['hello world'], ['world is there']]
    patterns = [bip.NgramSetPattern(ngram) for ngram in patterns]
    context = dict(zip('abc', patterns))
    ps = PatternStructure()
    ps.fit(context)
    assert ps.premaximal_patterns == {patterns[0]: {'a'}, patterns[2]: {'c'}}


def test_iter_atomic_patterns():
    #####################################################################
    # Test Atomised patterns where all atomic patterns are incomparable #
    #####################################################################
    class APattern(Pattern):  # short for atomised pattern
        def atomise(self, atoms_configuration: Literal['min', 'max'] = 'min') -> set[Self]:
            return {self.__class__(frozenset([v])) for v in self.value}

    patterns = [APattern(frozenset({1, 2, 3})), APattern(frozenset({0, 4})), APattern(frozenset({1, 2, 4}))]
    atomic_patterns_true = OrderedDict([
        (APattern(frozenset({2})), fbarray('101')),  # supp: 2
        (APattern(frozenset({1})), fbarray('101')),  # supp: 2
        (APattern(frozenset({4})), fbarray('011')),  # supp: 2
        (APattern(frozenset({3})), fbarray('100')),  # supp: 1
        (APattern(frozenset({0})), fbarray('010')),  # supp: 1
    ])
    context = dict(zip('abc', patterns))

    ps = PatternStructure()
    ps.fit(context, compute_atomic_patterns=False)
    ps._atomic_patterns = atomic_patterns_true

    atomic_patterns = ps.iter_atomic_patterns(return_bitarrays=False)
    assert isinstance(atomic_patterns, Iterator)
    assert list(dict(atomic_patterns)) == list(atomic_patterns_true)

    atomic_patterns_true_verb = OrderedDict([(k, {'abc'[g] for g in ext_ba.search(True)})
                                             for k, ext_ba in atomic_patterns_true.items()])
    atomic_patterns = ps.iter_atomic_patterns(return_bitarrays=False)
    assert isinstance(atomic_patterns, Iterator)
    assert list(atomic_patterns) == list(atomic_patterns_true_verb.items())

    atomic_patterns = ps.iter_atomic_patterns(return_bitarrays=True)
    assert isinstance(atomic_patterns, Iterator)
    assert list(atomic_patterns) == list(atomic_patterns_true.items())

    ##################################################################
    # Test NgramSetPatterns when some atomic patterns are comparable #
    ##################################################################
    atomic_patterns_true = OrderedDict([
        ('hello', bitarray('111111111')),
        ('world', bitarray('111111001')),
        ('hello world', bitarray('001111001')),
        ('!', bitarray('110111111'))
    ])
    atomic_patterns_true = OrderedDict([(bip.NgramSetPattern([k]), v) for k, v in atomic_patterns_true.items()])

    ps = PatternStructure()
    ps._atomic_patterns = atomic_patterns_true
    atomic_patterns = OrderedDict(list(ps.iter_atomic_patterns(return_bitarrays=True)))
    assert len(atomic_patterns) == len(atomic_patterns_true)
    assert list(atomic_patterns) == list(atomic_patterns_true)
    assert atomic_patterns == atomic_patterns_true

    ps._atomic_patterns_order = [bitarray('0010'), bitarray('0010'), bitarray('0000'), bitarray('0000')]
    stop_pattern = bip.NgramSetPattern(['world'])
    atomic_patterns_true_stopped = atomic_patterns_true.copy()
    del atomic_patterns_true_stopped[bip.NgramSetPattern(['hello world'])]

    atomic_patterns_stopped, pattern = [], None
    iterator = ps.iter_atomic_patterns(return_bitarrays=True, kind='ascending controlled')
    next(iterator)  # initialisation

    while True:
        go_deeper = pattern is None or pattern != stop_pattern
        try:
            pattern, extent = iterator.send(go_deeper)
        except StopIteration:
            break
        atomic_patterns_stopped.append((pattern, extent))
    atomic_patterns_stopped = OrderedDict(atomic_patterns_stopped)
    assert len(atomic_patterns_stopped) == len(atomic_patterns_true_stopped)
    assert list(atomic_patterns_stopped) == list(atomic_patterns_true_stopped)
    assert atomic_patterns_stopped == atomic_patterns_true_stopped


def test_mine_concepts():
    # data is inspired by newzealand_en context from FCA_repository
    data = OrderedDict([
        ('Stewart Island', {'Hiking', 'Observing Nature', 'Sightseeing Flights'}),
        ('Fjordland NP', {'Hiking', 'Observing Nature', 'Sightseeing Flights'}),
        ('Invercargill', {'Hiking', 'Observing Nature', 'Sightseeing Flights'}),
        ('Milford Sound', {'Hiking', 'Observing Nature', 'Sightseeing Flights'}),
        ('MT. Aspiring NP', {'Hiking', 'Observing Nature', 'Sightseeing Flights'}),
        ('Te Anau', {'Hiking', 'Jet Boating', 'Observing Nature', 'Sightseeing Flights'}),
        ('Dunedin', {'Hiking', 'Observing Nature', 'Sightseeing Flights'}),
        ('Oamaru', {'Hiking', 'Observing Nature'}),
        ('Queenstown', {'Bungee Jumping', 'Hiking', 'Jet Boating', 'Parachute Gliding', 'Sightseeing Flights', 'Skiing',
                       'Wildwater Rafting'}),
        ('Wanaka', {'Bungee Jumping', 'Hiking', 'Jet Boating', 'Parachute Gliding', 'Sightseeing Flights',
                   'Skiing', 'Wildwater Rafting'}),
        ('Otago Peninsula', {'Hiking', 'Observing Nature'}),
        ('Haast', {'Hiking', 'Observing Nature'}),
        ('Catlins', {'Hiking', 'Observing Nature'})
    ])
    data = OrderedDict([(obj, bip.ItemSetPattern(descr)) for obj, descr in data.items()])

    # the intents are ordered lexicographically w.r.t. their extents ordered w.r.t. objects_order
    concepts_true = [
        (bitarray('1111111111111'), ('Hiking',)),  # extent: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        (bitarray('1111111100111'), ('Hiking', 'Observing Nature')),  # extent: [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
        (bitarray('1111111011000'), ('Hiking', 'Sightseeing Flights')),  # extent: [0, 1, 2, 3, 4, 5, 6, 8, 9]
        (bitarray('1111111000000'), ('Hiking', 'Observing Nature', 'Sightseeing Flights')),
        # extent: [0, 1, 2, 3, 4, 5, 6]
        (bitarray('0000010011000'), ('Hiking', 'Jet Boating', 'Sightseeing Flights')),  # extent: [5, 8, 9]
        (bitarray('0000000011000'), ('Bungee Jumping', 'Hiking', 'Jet Boating', 'Parachute Gliding', 'Sightseeing Flights',
          'Skiing', 'Wildwater Rafting')),  # extent: [8, 9]
        (bitarray('0000010000000'), ('Hiking', 'Jet Boating', 'Observing Nature', 'Sightseeing Flights')),
        # extent: [5]
        (bitarray('0000000000000'), ('Bungee Jumping', 'Hiking', 'Jet Boating', 'Observing Nature', 'Parachute Gliding',
                                     'Sightseeing Flights', 'Skiing', 'Wildwater Rafting')),  # extent: []
    ]
    concepts_true = [(extent, bip.ItemSetPattern(intent)) for extent, intent in concepts_true]

    ps = PatternStructure()
    ps.fit(data)

    concepts = ps.mine_concepts(min_support=0, min_delta_stability=0, return_objects_as_bitarrays=True)
    assert len(concepts) == len(concepts_true)
    assert concepts == concepts_true


    concepts = ps.mine_concepts(min_support=0, min_delta_stability=0, return_objects_as_bitarrays=True, algorithm='gSofia')
    assert len(concepts) == len(concepts_true)
    assert concepts == concepts_true

    with pytest.warns(UserWarning):
        concepts = ps.mine_concepts(min_support=10000, min_delta_stability=0, return_objects_as_bitarrays=True,
                                    algorithm='CloseByOne object-wise')
    assert len(concepts) == len(concepts_true)
    assert concepts == concepts_true

    freq_concepts_true = [(extent, intent) for extent, intent in concepts_true if extent.count() >= 3]
    freq_concepts = ps.mine_concepts(min_support=3, return_objects_as_bitarrays=True)
    assert len(freq_concepts) == len(freq_concepts_true)
    assert freq_concepts == freq_concepts_true

    # TODO: Add tests for min_delta_stability threshold


def test_iter_patterns():
    atomic_patterns_extents = OrderedDict([
        (bip.NgramSetPattern(['hello']), bitarray('111111111')),
        (bip.NgramSetPattern(['world']), bitarray('111111001')),
        (bip.NgramSetPattern(['hello world']), bitarray('001111001')),
        (bip.NgramSetPattern(['!']), bitarray('110111111'))
    ])
    context = OrderedDict([
        ('a', bip.NgramSetPattern(['hello', 'world', '!'])),
        ('b', bip.NgramSetPattern(['hello', 'world', '!'])),
        ('c', bip.NgramSetPattern(['hello world'])),
        ('d', bip.NgramSetPattern(['hello world', '!'])),
        ('e', bip.NgramSetPattern(['hello world', '!'])),
        ('f', bip.NgramSetPattern(['hello world', '!'])),
        ('g', bip.NgramSetPattern(['hello', '!'])),
        ('h', bip.NgramSetPattern(['hello', '!'])),
        ('i', bip.NgramSetPattern(['hello world', '!'])),
    ])

    ps = PatternStructure()
    ps.fit(context, compute_atomic_patterns=False)
    ps._atomic_patterns = atomic_patterns_extents

    iterator_tested = mec.iter_all_patterns_ascending(atomic_patterns_extents, min_support=3)
    iterator = ps.iter_patterns(min_support=3/9, return_objects_as_bitarrays=True)
    assert list(iterator) == list(iterator_tested)

    iterator_tested = mec.iter_all_patterns_ascending(atomic_patterns_extents, min_support=3, controlled_iteration=True)
    iterator = ps.iter_patterns(min_support=3 / 9, kind='ascending controlled', return_objects_as_bitarrays=True)
    next(iterator)
    next(iterator_tested)
    assert list(iterator) == list(iterator_tested)


def test_n_atomic_patterns():
    atomic_patterns_true = OrderedDict([
        ('hello', bitarray('111111111')),
        ('world', bitarray('111111001')),
        ('hello world', bitarray('001111001')),
        ('!', bitarray('110111111'))
    ])
    atomic_patterns_true = OrderedDict([(bip.NgramSetPattern([k]), v) for k, v in atomic_patterns_true.items()])

    ps = PatternStructure()
    ps._atomic_patterns = atomic_patterns_true
    assert ps.n_atomic_patterns == 4


def test_iter_keys():
    patterns = [['hello world', 'who is there'], ['hello world'], ['world is there']]
    patterns = [bip.NgramSetPattern(ngram) for ngram in patterns]
    context = dict(zip('abc', patterns))
    ps = PatternStructure()
    ps.fit(context)

    atomic_patterns = OrderedDict(ps.iter_atomic_patterns(return_bitarrays=True))

    intents = [intent for extent, intent in ps.mine_concepts()]
    for intent in intents:
        iter_trusted = set(mec.iter_keys_of_pattern(intent, atomic_patterns=atomic_patterns))
        iterator = set(ps.iter_keys(intent))
        assert iterator == iter_trusted, f"Problem with intent {intent}"

        extent = ps.extent(intent)
        for key in iterator:
            assert ps.extent(key) == extent, f"Problem with the extent of intent {intent}"

    iter_trusted = mec.iter_keys_of_patterns(intents, atomic_patterns)
    iter_trusted = {(ptrn, intents[intent_i]) for ptrn, intent_i in iter_trusted}

    iterator = set(ps.iter_keys(intents))
    assert iterator == iter_trusted

    iterator = set(ps.iter_keys(intents[::-1]))
    assert iterator == iter_trusted

    ############################
    # Test for IntervalPattern #
    ############################
    class BoundedIntervalPattern(bip.IntervalPattern):
        BoundsUniverse = (0, 1)

    patterns = [BoundedIntervalPattern(0), BoundedIntervalPattern(1), BoundedIntervalPattern(1)]
    context = dict(zip('abc', patterns))
    ps = PatternStructure()
    ps.fit(context)
    atomic_patterns = OrderedDict(ps.iter_atomic_patterns(return_bitarrays=True))

    intents = [intent for extent, intent in ps.mine_concepts()]
    for intent in intents:
        iter_trusted = set(mec.iter_keys_of_pattern(intent, atomic_patterns=atomic_patterns))
        iterator = set(ps.iter_keys(intent))
        assert iterator == iter_trusted, f"Problem with intent {intent}"

        extent = ps.extent(intent)
        for key in iterator:
            assert ps.extent(key) == extent, f"Problem with the extent of intent {intent}"