from collections import OrderedDict

from bitarray import bitarray, frozenbitarray as fbarray

import paspailleur.algorithms.base_functions as bfuncs
import paspailleur.pattern_structures.built_in_patterns as bip
import paspailleur.algorithms.mine_equivalence_classes as mec


def test_iter_intents_via_ocbo():
    # data is inspired by newzealand_en context from FCA_repository
    data = {
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
    data = {obj: bip.ItemSetPattern(descr) for obj, descr in data.items()}
    objects_order = ['Stewart Island', 'Fjordland NP', 'Invercargill', 'Milford Sound', 'MT. Aspiring NP', 'Te Anau',
                     'Dunedin', 'Oamaru', 'Queenstown', 'Wanaka', 'Otago Peninsula', 'Haast', 'Catlins']
    data_list = [data[k] for k in objects_order]

    # the intents are ordered lexicographically w.r.t. their extents ordered w.r.t. objects_order
    intents_true = OrderedDict([
        (('Bungee Jumping', 'Hiking', 'Jet Boating', 'Observing Nature', 'Parachute Gliding', 'Sightseeing Flights',
         'Skiing', 'Wildwater Rafting'), bitarray('0000000000000')),  # extent: []
        (('Hiking', 'Observing Nature', 'Sightseeing Flights'), bitarray('1111111000000')),  # extent: [0, 1, 2, 3, 4, 5, 6]
        (('Hiking', 'Observing Nature'), bitarray('1111111100111')),  # extent: [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
        (('Hiking',), bitarray('1111111111111')),  # extent: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        (('Hiking', 'Sightseeing Flights'), bitarray('1111111011000')),  # extent: [0, 1, 2, 3, 4, 5, 6, 8, 9]
        (('Hiking', 'Jet Boating', 'Observing Nature', 'Sightseeing Flights'), bitarray('0000010000000')),  # extent: [5]
        (('Hiking', 'Jet Boating', 'Sightseeing Flights'), bitarray('0000010011000')),  # extent: [5, 8, 9]
        (('Bungee Jumping', 'Hiking', 'Jet Boating', 'Parachute Gliding', 'Sightseeing Flights', 'Skiing',
         'Wildwater Rafting'), bitarray('0000000011000'))  # extent: [8, 9]
    ])
    intents_true = OrderedDict([(bip.ItemSetPattern(intent), extent) for intent, extent in intents_true.items()])

    intents = OrderedDict(mec.iter_intents_via_ocbo(data_list))
    assert len(intents) == len(intents_true)
    assert list(intents) == list(intents_true)
    assert intents == intents_true


def test_iter_all_patterns_ascending():
    #######################################################################
    # Tests for ItemSetPattern where all atomic patterns are incomparable #
    #######################################################################
    atomic_patterns_extents = OrderedDict([
        ('Hiking', bitarray('111111111')),
        ('Observing Nature', bitarray('111111001')),
        ('Sightseeing Flights', bitarray('001111111')),
    ])
    atomic_patterns_extents = OrderedDict([(bip.ItemSetPattern({k}), v) for k, v in atomic_patterns_extents.items()])

    all_patterns_true = OrderedDict([
        ('', bitarray('111111111')),
        ('Hiking', bitarray('111111111')),
        ('Hiking, Observing Nature', bitarray('111111001')),
        ('Hiking, Observing Nature, Sightseeing Flights', bitarray('001111001')),
        ('Hiking, Sightseeing Flights', bitarray('001111111')),
        ('Observing Nature', bitarray('111111001')),
        ('Observing Nature, Sightseeing Flights', bitarray('001111001')),
        ('Sightseeing Flights', bitarray('001111111'))
    ])
    all_patterns_true = OrderedDict([(bip.ItemSetPattern(k.split(', ') if k else []), v)
                                     for k, v in all_patterns_true.items()])

    all_patterns = OrderedDict(list(mec.iter_all_patterns_ascending(atomic_patterns_extents, min_support=0)))
    assert len(all_patterns) == len(all_patterns_true)
    assert list(all_patterns) == list(all_patterns_true)
    assert all_patterns == all_patterns_true

    all_patterns = OrderedDict(list(mec.iter_all_patterns_ascending(atomic_patterns_extents, min_support=7)))
    assert all_patterns == OrderedDict([(k, v) for k, v in all_patterns_true.items() if v.count() >= 7])

    ######################################################################
    # Test for NgramSetPattern where some atomic patterns are comparable #
    ######################################################################
    atomic_patterns_extents = OrderedDict([
        ('hello', bitarray('111111111')),
        ('world', bitarray('111111001')),
        ('hello world', bitarray('001111001')),
        ('!', bitarray('110111111'))
    ])
    atomic_patterns_extents = OrderedDict([(bip.NgramSetPattern([k]), v) for k, v in atomic_patterns_extents.items()])

    all_patterns_true = OrderedDict([
        ('', bitarray('111111111')),
        ('hello', bitarray('111111111')),
        ('hello, world', bitarray('111111001')),
        ('hello world', bitarray('001111001')),
        ('hello world, !', bitarray('000111001')),
        ('hello, world, !', bitarray('110111001')),
        ('hello, !', bitarray('110111111')),
        ('world', bitarray('111111001')),
        ('world, !', bitarray('110111001')),
        ('!', bitarray('110111111')),
    ])
    all_patterns_true = OrderedDict([(bip.NgramSetPattern(k.split(', ') if k else []), v)
                                     for k, v in all_patterns_true.items()])

    all_patterns = OrderedDict(list(mec.iter_all_patterns_ascending(atomic_patterns_extents, min_support=0)))
    assert len(all_patterns) == len(all_patterns_true)
    assert list(all_patterns) == list(all_patterns_true)
    assert all_patterns == all_patterns_true

    ######################################################
    # Test for NgramSetPattern with breadth-first search #
    ######################################################
    all_patterns_true_breadth = OrderedDict([
        ('', bitarray('111111111')),
        ('hello', bitarray('111111111')),
        ('world', bitarray('111111001')),
        ('!', bitarray('110111111')),
        ('hello, world', bitarray('111111001')),
        ('hello, !', bitarray('110111111')),
        ('world, !', bitarray('110111001')),
        ('hello world', bitarray('001111001')),
        ('hello, world, !', bitarray('110111001')),
        ('hello world, !', bitarray('000111001')),
    ])
    all_patterns_true_breadth = OrderedDict([(bip.NgramSetPattern(k.split(', ') if k else []), v)
                                             for k, v in all_patterns_true_breadth.items()])

    all_patterns = OrderedDict(list(mec.iter_all_patterns_ascending(atomic_patterns_extents, min_support=0, depth_first=False)))
    assert len(all_patterns) == len(all_patterns_true_breadth)
    assert list(all_patterns) == list(all_patterns_true_breadth)
    assert all_patterns == all_patterns_true_breadth

    #####################################################
    # Test NgramSetPattern with controllable navigation #
    #####################################################
    stop_pattern = bip.NgramSetPattern(['hello', 'world'])
    all_patterns_true_stopped = OrderedDict([(k, v) for k, v in all_patterns_true.items()
                                             if not k > stop_pattern])
    iterator = mec.iter_all_patterns_ascending(atomic_patterns_extents, controlled_iteration=True)
    _ = next(iterator)
    all_patterns_stopped, pattern = [], None
    while True:
        go_deeper = (pattern is None) or (pattern != stop_pattern)
        try:
            pattern, extent = iterator.send(go_deeper)
        except StopIteration:
            break
        all_patterns_stopped.append((pattern, extent))
    all_patterns_stopped = OrderedDict(all_patterns_stopped)
    assert len(all_patterns_stopped) == len(all_patterns_true_stopped)
    assert list(all_patterns_stopped) == list(all_patterns_true_stopped)
    assert all_patterns_stopped == all_patterns_true_stopped

    stop_pattern = bip.NgramSetPattern([])
    iterator = mec.iter_all_patterns_ascending(atomic_patterns_extents, controlled_iteration=True)
    _ = next(iterator)
    all_patterns_stopped, pattern = [], None
    while True:
        go_deeper = (pattern is None) or (pattern != stop_pattern)
        try:
            pattern, extent = iterator.send(go_deeper)
        except StopIteration:
            break
        all_patterns_stopped.append((pattern, extent))
    all_patterns_stopped = OrderedDict(all_patterns_stopped)
    assert len(all_patterns_stopped) == 1


def test_list_stable_extents_via_gsofia():
    ###################################
    # Simplest tests for simple cases #
    ###################################
    atomic_patterns = OrderedDict([
        (bip.Pattern('frozenset({1})'), fbarray('110')),
        (bip.Pattern('frozenset({2})'), fbarray('101')),
        (bip.Pattern('frozenset({3})'), fbarray('011'))
    ])
    ordering = [fbarray('000'), fbarray('000'), fbarray('000')]
    extents_true = {
        fbarray('111'),
        fbarray('110'), fbarray('101'), fbarray('011'),
        fbarray('100'), fbarray('010'), fbarray('001'),
        fbarray('000')
    }

    atomic_patterns_iterator = bfuncs.iter_patterns_ascending(atomic_patterns, ordering, controlled_iteration=True)
    stable_extents = mec.list_stable_extents_via_gsofia(atomic_patterns_iterator, min_delta_stability=0)
    assert stable_extents == extents_true

    atomic_patterns_iterator = bfuncs.iter_patterns_ascending(atomic_patterns, ordering, controlled_iteration=True)
    stable_extents = mec.list_stable_extents_via_gsofia(atomic_patterns_iterator, min_delta_stability=1)
    assert stable_extents == extents_true - {fbarray('000')}

    atomic_patterns_iterator = bfuncs.iter_patterns_ascending(atomic_patterns, ordering, controlled_iteration=True)
    stable_extents = mec.list_stable_extents_via_gsofia(atomic_patterns_iterator, min_delta_stability=2)
    assert stable_extents == set()

    #########################################################################
    # More elaborate but still toy-ish case with comparable atomic patterns #
    #########################################################################
    atomic_patterns_extents = OrderedDict([
        ('hello', bitarray('111111111')),
        ('world', bitarray('111111001')),
        ('hello world', bitarray('001111001')),
        ('!', bitarray('110111111'))
    ])
    atomic_patterns_extents = OrderedDict([(bip.NgramSetPattern([k]), fbarray(v)) for k, v in atomic_patterns_extents.items()])
    ordering = [fbarray('0010'), fbarray('0010'), fbarray('0000'), fbarray('0000')]

    extents_stabilities = {
        fbarray('111111111'): 1,  # "hello"
        fbarray('111111001'): 1,  # "hello", "world"
        fbarray('110111111'): 2,  # "hello", "!"
        fbarray('001111001'): 1,  # "hello world"
        fbarray('110111001'): 2,  # "hello", "world", "!"
        fbarray('000111001'): 4,  # "hello world", "!"
    }

    for delta_min in [1, 2, 3, 4, 5]:
        stable_extents_true = {ext for ext, delta in extents_stabilities.items() if delta >= delta_min}

        atomic_patterns_iterator = bfuncs.iter_patterns_ascending(atomic_patterns_extents, ordering, controlled_iteration=True)
        stable_extents = set(mec.list_stable_extents_via_gsofia(atomic_patterns_iterator, min_delta_stability=delta_min))
        assert stable_extents == stable_extents_true,  f"Problem with {delta_min=}"


def test_iter_keys_of_pattern():
    # CopyPaste from tests from Caspailleur
    atomic_patterns = OrderedDict([
        (bip.ItemSetPattern(['a']), fbarray('1100')), (bip.ItemSetPattern(['b']), fbarray('0011')),
        (bip.ItemSetPattern(['c']), fbarray('1011')), (bip.ItemSetPattern(['d']), fbarray('0110')),
        (bip.ItemSetPattern(['e']), fbarray('0000'))
    ])
    patterns_keys_true = {
        bip.ItemSetPattern([]): [bip.ItemSetPattern([])],
        bip.ItemSetPattern(['a']): [bip.ItemSetPattern(['a'])],
        bip.ItemSetPattern(['c']): [bip.ItemSetPattern(['c'])],
        bip.ItemSetPattern(['d']): [bip.ItemSetPattern(['d'])],
        bip.ItemSetPattern(['a', 'c']): [bip.ItemSetPattern(['a', 'c'])],
        bip.ItemSetPattern(['a', 'd']): [bip.ItemSetPattern(['a', 'd'])],
        bip.ItemSetPattern(['b', 'c']): [bip.ItemSetPattern(['b'])],
        bip.ItemSetPattern(['b', 'c', 'd']): [bip.ItemSetPattern(['b', 'd']), bip.ItemSetPattern(['c', 'd'])],
        bip.ItemSetPattern(['a', 'b', 'c', 'd', 'e']): [
            bip.ItemSetPattern(['e']), bip.ItemSetPattern(['a', 'b']), bip.ItemSetPattern(['a', 'c', 'd'])]
    }

    for pattern, keys_true in patterns_keys_true.items():
        keys = list(mec.iter_keys_of_pattern(pattern, atomic_patterns))
        assert keys == keys_true, f"{pattern}: true keys {keys_true}, mined keys {keys}"


def test_iter_keys_of_patterns():
    atomic_patterns = OrderedDict([
        (bip.ItemSetPattern(['a']), fbarray('1100')), (bip.ItemSetPattern(['b']), fbarray('0011')),
        (bip.ItemSetPattern(['c']), fbarray('1011')), (bip.ItemSetPattern(['d']), fbarray('0110')),
        (bip.ItemSetPattern(['e']), fbarray('0000'))
    ])
    patterns_keys_true = {
        bip.ItemSetPattern([]): [bip.ItemSetPattern([])],
        bip.ItemSetPattern(['a']): [bip.ItemSetPattern(['a'])],
        bip.ItemSetPattern(['c']): [bip.ItemSetPattern(['c'])],
        bip.ItemSetPattern(['d']): [bip.ItemSetPattern(['d'])],
        bip.ItemSetPattern(['a', 'c']): [bip.ItemSetPattern(['a', 'c'])],
        bip.ItemSetPattern(['a', 'd']): [bip.ItemSetPattern(['a', 'd'])],
        bip.ItemSetPattern(['b', 'c']): [bip.ItemSetPattern(['b'])],
        bip.ItemSetPattern(['b', 'c', 'd']): [bip.ItemSetPattern(['b', 'd']), bip.ItemSetPattern(['c', 'd'])],
        bip.ItemSetPattern(['a', 'b', 'c', 'd', 'e']): [
            bip.ItemSetPattern(['e']), bip.ItemSetPattern(['a', 'b']), bip.ItemSetPattern(['a', 'c', 'd'])]
    }
    patterns = list(patterns_keys_true)

    patterns_keys = dict()
    for key, pattern_i in mec.iter_keys_of_patterns(patterns, atomic_patterns):
        pattern = patterns[pattern_i]
        if pattern not in patterns_keys:
            patterns_keys[pattern] = []
        patterns_keys[pattern].append(key)

    assert patterns_keys == patterns_keys_true

    patterns = patterns[::-1]
    patterns_keys = dict()
    for key, pattern_i in mec.iter_keys_of_patterns(patterns, atomic_patterns):
        pattern = patterns[pattern_i]
        if pattern not in patterns_keys:
            patterns_keys[pattern] = []
        patterns_keys[pattern].append(key)

    assert patterns_keys == patterns_keys_true

    ########################
    # NgramSetPattern case #
    ########################
    #patterns = [['hello world', 'who is there'], ['hello world'], ['world is there']]
    #patterns = [bip.NgramSetPattern(ngram) for ngram in patterns]

    atomic_patterns = OrderedDict([
        (bip.NgramSetPattern({'world'}), fbarray('111')),
        (bip.NgramSetPattern({'hello'}), fbarray('110')),
        (bip.NgramSetPattern({'hello world'}), fbarray('110')),
        (bip.NgramSetPattern({'is'}), fbarray('101')),
        (bip.NgramSetPattern({'there'}), fbarray('101')),
        (bip.NgramSetPattern({'is there'}), fbarray('101')),
        (bip.NgramSetPattern({'who'}), fbarray('100')),
        (bip.NgramSetPattern({'who is'}), fbarray('100')),
        (bip.NgramSetPattern({'who is there'}), fbarray('100')),
        (bip.NgramSetPattern({'world is'}), fbarray('001')),
        (bip.NgramSetPattern({'world is there'}), fbarray('001'))
    ])
    patterns_keys_true = {
        bip.NgramSetPattern(['world']): [bip.NgramSetPattern([])],
        bip.NgramSetPattern(['hello world']): [bip.NgramSetPattern(['hello'])],
        bip.NgramSetPattern(['is there', 'world']): [bip.NgramSetPattern(['is']),
                                                     bip.NgramSetPattern(['there'])],
        bip.NgramSetPattern(['who is there', 'hello world']): [bip.NgramSetPattern(['who']),
                                                               bip.NgramSetPattern(['is', 'hello']),
                                                               bip.NgramSetPattern(['hello', 'there'])
                                                               ],
        bip.NgramSetPattern(['world is there']): [bip.NgramSetPattern(['world is'])],
        bip.NgramSetPattern(['who is there', 'world is there', 'hello world']): [
            bip.NgramSetPattern(['hello', 'world is']),
            bip.NgramSetPattern(['who', 'world is']),
        ]
    }
    patterns = list(patterns_keys_true)

    patterns_keys = dict()
    for key, pattern_i in mec.iter_keys_of_patterns(patterns, atomic_patterns):
        pattern = patterns[pattern_i]
        if pattern not in patterns_keys:
            patterns_keys[pattern] = []
        patterns_keys[pattern].append(key)

    assert patterns_keys == patterns_keys_true

