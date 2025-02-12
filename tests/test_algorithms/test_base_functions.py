from bitarray import bitarray, frozenbitarray as fbarray
from paspailleur.pattern_structures.pattern import Pattern

from paspailleur.algorithms import base_functions as bfunc
from paspailleur.pattern_structures import built_in_patterns as bip


def test_rearrange_indices():
    order_before = [
        bitarray('01111111'),
        bitarray('00010001'),
        bitarray('00001111'),
        bitarray('00000001'),
        bitarray('00000111'),
        bitarray('00000011'),
        bitarray('00000001'),
        bitarray('00000000')
    ]
    elements_before = [bip.IntervalPattern('[-inf, inf]'), bip.IntervalPattern('<= 11.0'),
                      bip.IntervalPattern('>= 0.0'), bip.IntervalPattern('<= 10.0'),
                      bip.IntervalPattern('>= 2.0'), bip.IntervalPattern('> 2.0'),
                      bip.IntervalPattern('>= 5.0'), bip.IntervalPattern('ø')]
    elements_after = [
        bip.IntervalPattern('>= 5.0'), bip.IntervalPattern('<= 11.0'),
        bip.IntervalPattern('ø'), bip.IntervalPattern('> 2.0'),
        bip.IntervalPattern('>= 0.0'), bip.IntervalPattern('>= 2.0'),
        bip.IntervalPattern('<= 10.0'), bip.IntervalPattern('[-inf, inf]')
    ]
    order_after_true = [
        bitarray('00100000'),  # >=5
        bitarray('00100010'),  # <=11
        bitarray('00000000'),  # ø
        bitarray('10100000'),  # >2
        bitarray('10110100'),  # >=0
        bitarray('10110000'),  # >=2
        bitarray('00100000'),  # <=10
        bitarray('11111110'),  # [-inf, inf]
    ]

    order_after = bfunc.rearrange_indices(order_before, elements_before, elements_after)
    assert order_after == order_after_true


def test_order_patterns_via_extents():
    patterns_extents = [
        (Pattern(frozenset('a',)), fbarray('111')),
        (Pattern(frozenset('ab')), fbarray('110')),
        (Pattern(frozenset('ac')), fbarray('101')),
        (Pattern(frozenset('abc')), fbarray('100')),
        (Pattern(frozenset('abd')), fbarray('010'))
    ]
    patterns_order_true = [bitarray('01111'), bitarray('00011'), bitarray('00010'), bitarray('00000'), bitarray('00000')]
    patterns_order = bfunc.order_patterns_via_extents(patterns_extents)
    assert patterns_order == patterns_order_true

    patterns_extents = [
        (Pattern(frozenset('ab')), fbarray('110')),
        (Pattern(frozenset('a', )), fbarray('111')),
        (Pattern(frozenset('abc')), fbarray('100')),
        (Pattern(frozenset('ac')), fbarray('101')),
        (Pattern(frozenset('abd')), fbarray('010'))
    ]
    patterns_order_true = [bitarray('00101'), bitarray('10111'), bitarray('00000'), bitarray('00100'), bitarray('00000')]
    patterns_order = bfunc.order_patterns_via_extents(patterns_extents)
    assert patterns_order == patterns_order_true

    patterns_extents = [
        (bip.IntervalPattern('[-inf, inf]'), fbarray('111')),
        (bip.IntervalPattern('[0, inf]'), fbarray('111')),
        (bip.IntervalPattern('[-inf, 11]'), fbarray('111')),
        (bip.IntervalPattern('[-inf, 10]'), fbarray('101')),
        (bip.IntervalPattern('[2, inf]'), fbarray('011')),
        (bip.IntervalPattern('(2, inf]'), fbarray('011')),
        (bip.IntervalPattern('[5, inf]'), fbarray('001')),
        (bip.IntervalPattern('ø'), fbarray('000')),
    ]
    patterns_order_true = [
        bitarray('01111111'),  # [-inf, inf]
        bitarray('00001111'),  # >=0
        bitarray('00010001'),  # <=11
        bitarray('00000001'),  # <= 10
        bitarray('00000111'),  # >=2
        bitarray('00000011'),  # >2
        bitarray('00000001'),  # >=5
        bitarray('00000000')  # ø
    ]
    patterns_order = bfunc.order_patterns_via_extents(patterns_extents)
    assert patterns_order == patterns_order_true

    patterns_extents = [
        (bip.IntervalPattern('[5, inf]'), fbarray('001')),
        (bip.IntervalPattern('[-inf, 11]'), fbarray('111')),
        (bip.IntervalPattern('ø'), fbarray('000')),
        (bip.IntervalPattern('(2, inf]'), fbarray('011')),
        (bip.IntervalPattern('[0, inf]'), fbarray('111')),
        (bip.IntervalPattern('[2, inf]'), fbarray('011')),
        (bip.IntervalPattern('[-inf, 10]'), fbarray('101')),
        (bip.IntervalPattern('[-inf, inf]'), fbarray('111'))
    ]
    patterns_order_true = [
        bitarray('00100000'),  # >=5
        bitarray('00100010'),  # <=11
        bitarray('00000000'),  # ø
        bitarray('10100000'),  # >2
        bitarray('10110100'),  # >=0
        bitarray('10110000'),  # >=2
        bitarray('00100000'),  # <=10
        bitarray('11111110'),  # [-inf, inf]
    ]
    patterns_order = bfunc.order_patterns_via_extents(patterns_extents)
    assert patterns_order == patterns_order_true
