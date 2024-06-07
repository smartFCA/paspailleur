from paspailleur.pattern_structures import CartesianPS, IntervalPS, DisjunctiveSetPS, ConjunctiveSetPS, NgramPS, BoundStatus as BS
import math
from bitarray import frozenbitarray as fbarray


def test_intersect_patterns():
    a = ((1, 2, BS.CLOSED), (3, 5, BS.CLOSED))
    b = ((0, 2, BS.CLOSED), (3, 5, BS.CLOSED))
    cps = CartesianPS(basic_structures=[IntervalPS(), IntervalPS()])
    assert cps.join_patterns(a, b) == b
    assert cps.join_patterns(b, a) == b

    assert cps.join_patterns(cps.max_pattern, a) == a
    assert cps.join_patterns(a, cps.max_pattern) == a


def test_bin_attributes():
    data = [
        ((0, 1, BS.CLOSED), (10, 20, BS.CLOSED)),
        ((1, 2, BS.CLOSED), (10, 20, BS.CLOSED))
    ]
    patterns_true = (
        (0, (0, 2, BS.CLOSED)), (0, (1, 2, BS.CLOSED)), (0, (0, 1, BS.CLOSED)), (0, (None, None, BS.CLOSED)),
        (1, (10, 20, BS.CLOSED)), (1, (None, None, BS.CLOSED)),
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
    patterns, flags = list(zip(*list(cps.iter_attributes(data))))
    assert patterns == patterns_true
    assert flags == flags_true

    patterns, flags = list(zip(*list(cps.iter_attributes(data, min_support=0.5))))
    assert set(flags) == {flg for flg in flags_true if flg.count() > 0}

    patterns, flags = list(zip(*list(cps.iter_attributes(data, min_support=1))))
    assert set(flags) == {flg for flg in flags_true if flg.count() > 0}


def test_is_subpattern():
    cps = CartesianPS(basic_structures=[IntervalPS(), IntervalPS()])

    assert cps.is_less_precise([(0, 1, BS.CLOSED), (3, 5, BS.CLOSED)], [(1, 1, BS.CLOSED), (3, 4, BS.CLOSED)])
    assert not cps.is_less_precise([(0, 1, BS.CLOSED), (3, 5, BS.CLOSED)], [(1, 1, BS.CLOSED), (2, 4, BS.CLOSED)])


def test_n_bin_attributes():
    data = [
        ((0, 1, BS.CLOSED), (10, 20, BS.CLOSED)),
        ((1, 2, BS.CLOSED), (10, 20, BS.CLOSED))
    ]

    cps = CartesianPS(basic_structures=[IntervalPS(), IntervalPS()])
    assert cps.n_attributes(data) == 6


def test_binarize():
    data = [
        ((0, 1, BS.CLOSED), (10, 20, BS.CLOSED)),
        ((1, 2, BS.CLOSED), (10, 20, BS.CLOSED))
    ]
    patterns_true = [
        (0, (0, 2, BS.CLOSED)), (0, (1, 2, BS.CLOSED)), (0, (0, 1, BS.CLOSED)), (0, (None, None, BS.CLOSED)),
        (1, (10, 20, BS.CLOSED)), (1, (None, None, BS.CLOSED)),
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


def test_intent():
    data = [
        ((0, 1, BS.CLOSED), (10, 20, BS.CLOSED)),
        ((1, 2, BS.CLOSED), (10, 20, BS.CLOSED))
    ]

    cps = CartesianPS(basic_structures=[IntervalPS(), IntervalPS()])
    assert cps.intent(data) == ((0, 2, BS.CLOSED), (10, 20, BS.CLOSED))


def test_extent():
    data = [
        [(0, 1, BS.CLOSED), (10, 20, BS.CLOSED)],
        [(1, 2, BS.CLOSED), (10, 20, BS.CLOSED)]
    ]

    cps = CartesianPS(basic_structures=[IntervalPS(), IntervalPS()])
    assert list(cps.extent(data, [(1, 2, BS.CLOSED), (10, 20, BS.CLOSED)])) == [1]


def test_preprocess_data():
    data = [
        [(0, 1), 'x', 'hello world'],
        [(0, 3), 'y', 'hello']
    ]

    cps = CartesianPS(basic_structures=[IntervalPS(), DisjunctiveSetPS(), NgramPS()])
    assert list(cps.preprocess_data(data)) == [
        ((0., 1., BS.CLOSED), frozenset({'x'}), frozenset({('hello', 'world')})),
        ((0., 3., BS.CLOSED), frozenset({'y'}), frozenset({('hello',)}))
    ]

    data = [(1, (3, 4), 'ClassA'),
            (2, (2, 5), 'ClassB')]
    cps = CartesianPS(basic_structures=[IntervalPS(), IntervalPS(), DisjunctiveSetPS()])
    dp = list(cps.preprocess_data(data))
    assert dp == [((1., 1., BS.CLOSED), (3., 4., BS.CLOSED), frozenset({'ClassA'})),
                  ((2., 2., BS.CLOSED), (2., 5., BS.CLOSED), frozenset({'ClassB'}))]
    assert cps.basic_structures[0].min_bounds == (1., 2.)
    assert cps.basic_structures[0].max_bounds == (1., 2.)
    assert cps.basic_structures[1].min_bounds == (2., 3.)
    assert cps.basic_structures[1].max_bounds == (4., 5.)
    assert cps.basic_structures[2].min_pattern == frozenset({'ClassA', 'ClassB'})


def test_verbalize():
    ps = CartesianPS(basic_structures=[IntervalPS(), NgramPS(), DisjunctiveSetPS(), ConjunctiveSetPS()])
    description = [(1, 2.43, BS.CLOSED), {('hello', 'world'), ('hello',)}, {'a', 'b'}, set()]
    verbose = "0: [1.00, 2.43], 1: hello world; hello, 2: a, b, 3: ∅"
    assert ps.verbalize(description) == verbose

    params = {0: {'number_format': '.0f'}, 1: {'ngram_separator': ', '}}
    verbose = "0: [1, 2]\n1: hello world, hello\n2: a, b\n3: ∅"
    assert ps.verbalize(description, basic_structures_params=params, separator='\n') == verbose


def test_closest_less_precise():
    ps = CartesianPS(basic_structures=[
        IntervalPS(ndigits=2), NgramPS(),
        DisjunctiveSetPS(all_values={'a', 'b', 'c', 'd'}), ConjunctiveSetPS()
    ])

    description = [(1, 2.43, BS.CLOSED), frozenset({('hello', 'world'), ('!',)}), frozenset({'b', 'c'}), frozenset()]
    next_descrs = set(ps.closest_less_precise(description, use_lectic_order=False))
    next_descrs_true = {
        tuple([(1, 2.44, BS.LCLOSED), frozenset({('hello', 'world'), ('!',)}), frozenset({'b', 'c'}), frozenset()]),
        tuple([(0.99, 2.43, BS.RCLOSED), frozenset({('hello', 'world'), ('!',)}), frozenset({'b', 'c'}), frozenset()]),
        tuple([(1, 2.43, BS.CLOSED), frozenset({('hello', 'world')}), frozenset({'b', 'c'}), frozenset()]),
        tuple([(1, 2.43, BS.CLOSED), frozenset({('hello',), ('world',), ('!',)}), frozenset({'b', 'c'}), frozenset()]),
        tuple([(1, 2.43, BS.CLOSED), frozenset({('hello', 'world'), ('!',)}), frozenset({'a', 'b', 'c'}), frozenset()]),
        tuple([(1, 2.43, BS.CLOSED), frozenset({('hello', 'world'), ('!',)}), frozenset({'b', 'c', 'd'}), frozenset()]),
    }
    assert next_descrs == next_descrs_true

    # TODO: Setup Tests for lectic order
    #next_descrs = set(ps.closest_less_precise(description, use_lectic_order=True))
    #next_descrs_true = {
    #    tuple([(1, 2.43, BS.LCLOSED), frozenset({('hello', 'world'), ('!',)}), frozenset({'b', 'c'}), frozenset()]),
    #    tuple([(1, 2.43), frozenset({('hello', 'world')}), frozenset({'b', 'c'}), frozenset()]),
    #    tuple([(1, 2.43), frozenset({('hello',), ('world',), ('!',)}), frozenset({'b', 'c'}), frozenset()]),
    #    tuple([(1, 2.43), frozenset({('hello', 'world'), ('!',)}), frozenset({'b', 'c', 'd'}), frozenset()]),
    #}
    #assert next_descrs == next_descrs_true


def test_closest_more_precise():
    ps = CartesianPS(basic_structures=[
        IntervalPS(ndigits=2), NgramPS(),
        DisjunctiveSetPS(all_values={'a', 'b', 'c', 'd'}), ConjunctiveSetPS(all_values={'x'})
    ])

    description = [(1, 2.43, BS.CLOSED), frozenset({('hello', 'world'), ('!',)}), frozenset({'b', 'c'}), frozenset()]
    next_descrs = set(ps.closest_more_precise(description, use_lectic_order=False))
    next_descrs_true = {
        tuple([(1, 2.43, BS.RCLOSED), frozenset({('hello', 'world'), ('!',)}), frozenset({'b', 'c'}), frozenset()]),
        tuple([(1, 2.43, BS.LCLOSED), frozenset({('hello', 'world'), ('!',)}), frozenset({'b', 'c'}), frozenset()]),

        tuple([(1, 2.43, BS.CLOSED), frozenset({('hello', 'world', '!')}), frozenset({'b', 'c'}), frozenset()]),
        tuple([(1, 2.43, BS.CLOSED), frozenset({('!', 'hello', 'world')}), frozenset({'b', 'c'}), frozenset()]),
        tuple([(1, 2.43, BS.CLOSED), frozenset({('hello', 'world'), ('!', '!')}), frozenset({'b', 'c'}), frozenset()]),
        tuple([(1, 2.43, BS.CLOSED), frozenset({('hello', 'world', 'hello'), ('!',)}), frozenset({'b', 'c'}), frozenset()]),
        tuple([(1, 2.43, BS.CLOSED), frozenset({('hello', 'hello', 'world'), ('!',)}), frozenset({'b', 'c'}), frozenset()]),
        tuple([(1, 2.43, BS.CLOSED), frozenset({('hello', 'world'), ('!', 'hello')}), frozenset({'b', 'c'}), frozenset()]),
        tuple([(1, 2.43, BS.CLOSED), frozenset({('hello', 'world'), ('hello', '!')}), frozenset({'b', 'c'}), frozenset()]),
        tuple([(1, 2.43, BS.CLOSED), frozenset({('hello', 'world', 'world'), ('!',)}), frozenset({'b', 'c'}), frozenset()]),
        tuple([(1, 2.43, BS.CLOSED), frozenset({('world', 'hello', 'world'), ('!',)}), frozenset({'b', 'c'}), frozenset()]),
        tuple([(1, 2.43, BS.CLOSED), frozenset({('hello', 'world'), ('!', 'world')}), frozenset({'b', 'c'}), frozenset()]),
        tuple([(1, 2.43, BS.CLOSED), frozenset({('hello', 'world'), ('world', '!')}), frozenset({'b', 'c'}), frozenset()]),

        tuple([(1, 2.43, BS.CLOSED), frozenset({('hello', 'world'), ('!',)}), frozenset({'b'}), frozenset()]),
        tuple([(1, 2.43, BS.CLOSED), frozenset({('hello', 'world'), ('!',)}), frozenset({'c'}), frozenset()]),

        tuple([(1, 2.43, BS.CLOSED), frozenset({('hello', 'world'), ('!',)}), frozenset({'b', 'c'}), frozenset({'x'})]),
    }
    assert next_descrs == next_descrs_true

    # TODO: Double check the following test
    #next_descrs = set(ps.closest_more_precise(description, use_lectic_order=True))
    #next_descrs_true = {
    #    tuple([(1, 2.42), frozenset({('hello', 'world'), ('!',)}), frozenset({'b', 'c'}), frozenset()]),

    #    tuple([(1, 2.43), frozenset({('hello', 'world', '!')}), frozenset({'b', 'c'}), frozenset()]),
    #    tuple([(1, 2.43), frozenset({('hello', 'world'), ('!', '!')}), frozenset({'b', 'c'}), frozenset()]),
    #    tuple([(1, 2.43), frozenset({('hello', 'world', 'hello'), ('!',)}), frozenset({'b', 'c'}), frozenset()]),
    #    tuple([(1, 2.43), frozenset({('hello', 'world'), ('!', 'hello')}), frozenset({'b', 'c'}), frozenset()]),
    #    tuple([(1, 2.43), frozenset({('hello', 'world', 'world'), ('!',)}), frozenset({'b', 'c'}), frozenset()]),
    #    tuple([(1, 2.43), frozenset({('hello', 'world'), ('!', 'world')}), frozenset({'b', 'c'}), frozenset()]),
    #
    #    tuple([(1, 2.43), frozenset({('hello', 'world'), ('!',)}), frozenset({'b', 'c'}), frozenset({'x'})]),
    #}
    #assert next_descrs == next_descrs_true


def test_keys():
    ps = CartesianPS(basic_structures=[
        IntervalPS(ndigits=2), IntervalPS(ndigits=2),
    ])
    data = [
        (3, 3), (6, 3), (3, 5), (6, 5),
        (0.5, 5), (1, 4), (2, 1), (2, 7), (7, 7)
    ]
    data = list(ps.preprocess_data(data))

    # TODO: Optmise keys-computation and uncomment the tests
    # keys = ps.keys(((3, 6, BS.CLOSED), (3, 5, BS.CLOSED)), data)
    # keys_true = [
    #    ((2, 7, BS.OPEN), (-math.inf, math.inf, BS.OPEN)),
    #    ((2, math.inf, BS.OPEN), (-math.inf, 7, BS.OPEN)),
    #    ((1, math.inf, BS.OPEN), (1, 7, BS.OPEN))
    #]
    #assert keys == keys_true


def test_passkeys():
    ps = CartesianPS(basic_structures=[
        IntervalPS(ndigits=2), IntervalPS(ndigits=2),
    ])
    data = [
        (3, 3), (6, 3), (3, 5), (6, 5),
        (0.5, 5), (1, 4), (2, 1), (2, 7), (7, 7)
    ]
    data = list(ps.preprocess_data(data))

    pkeys = ps.passkeys(((3, 6, BS.CLOSED), (3, 5, BS.CLOSED)), data)
    pkeys_true = [((2, 7, BS.OPEN), (-math.inf, math.inf, BS.OPEN))]
    assert pkeys == pkeys_true

    pkeys = ps.passkeys(((1, 6, BS.CLOSED), (3, 5, BS.CLOSED)), data)
    pkeys_true = [((0.5, math.inf, BS.OPEN), (1, 7, BS.OPEN))]
    assert pkeys == pkeys_true

    # Harry Potter-inspired complex data
    ps = CartesianPS(basic_structures=[IntervalPS(ndigits=2), DisjunctiveSetPS(), NgramPS()])
    data = [
        (11, 'Gryffindor', 'Harry Potter'),
        (11, 'Gryffindor', 'Ron Weasley'),
        (13, 'Gryffindor', 'George Weasley'),
        (13, 'Gryffindor', 'Fred Weasley'),
        (11, 'Slytherin', 'Draco Malfoy')
    ]
    data = list(ps.preprocess_data(data))
    schools = ps.min_pattern[1]

    intent = ((11, 13, BS.CLOSED), schools, frozenset())
    pkeys = ps.passkeys(intent, data)
    pkeys_true = [((-math.inf, math.inf, BS.OPEN), schools, frozenset())]
    assert set(pkeys) == set(pkeys_true)

    intent = ((11, 11, BS.CLOSED), frozenset({'Gryffindor'}), frozenset({('Harry', 'Potter')}))
    pkeys = ps.passkeys(intent, data)
    pkeys_true = [((-math.inf, math.inf, BS.OPEN), schools, frozenset({('Harry',)})),
                  ((-math.inf, math.inf, BS.OPEN), schools, frozenset({('Potter',)}))]
    assert set(pkeys) == set(pkeys_true)

    intent = ((11, 13, BS.CLOSED), frozenset({'Gryffindor'}), frozenset({('Weasley',)}))
    pkeys = ps.passkeys(intent, data)
    pkeys_true = [
        ((-math.inf, math.inf, BS.OPEN), schools, frozenset({('Weasley',)})),
    ]
    assert set(pkeys) == set(pkeys_true)

    intent = ((11, 11, BS.CLOSED), schools, frozenset())
    pkeys = ps.passkeys(intent, data)
    pkeys_true = [
        ((-math.inf, 13, BS.OPEN), schools, frozenset()),
    ]
    assert set(pkeys) == set(pkeys_true)
