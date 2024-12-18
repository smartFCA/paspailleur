import math

import pytest

from paspailleur.pattern_structures import built_in_patterns as bip


def test_ItemSetPattern():
    a = bip.ItemSetPattern([1, 3, 2])
    a2 = bip.ItemSetPattern({1, 2, 3})
    b = bip.ItemSetPattern([1, 2])
    c = bip.ItemSetPattern([1, 2, 6, 2])
    c2 = bip.ItemSetPattern([1, 2, 6])
    z = bip.ItemSetPattern('123')

    assert a == a2
    assert c == c2
    assert b <= a
    assert not (a <= b)

    assert z.value == {'1', '2', '3'}

    a = bip.ItemSetPattern(range(1, 5))
    b = bip.ItemSetPattern(range(3, 7))
    meet = bip.ItemSetPattern(range(3, 5))
    join = bip.ItemSetPattern(range(1, 7))
    assert a & b == meet
    assert a | b == join

    try:
        {a, b, meet, join}
    except TypeError as e:
        assert e

    a = bip.ItemSetPattern(range(1, 5))
    assert str(a) == "ItemSetPattern({1, 2, 3, 4})"

    a = bip.ItemSetPattern({1, 2, 3, 4})
    b = bip.ItemSetPattern({3, 4, 5, 6, 7})
    sub = bip.ItemSetPattern({1, 2})
    assert a - b == sub


def test_IntervalPattern():
    a = bip.IntervalPattern(((1, True), (10, False)))
    a2 = bip.IntervalPattern('[1, 10)')
    assert a.value == ((1, True), (10, False))
    assert a == a2

    a = bip.IntervalPattern('[-inf, ∞)')
    assert a._lower_bound == -math.inf
    assert a._upper_bound == math.inf

    a = bip.IntervalPattern('[1, 10]')
    b = bip.IntervalPattern('[1, 20]')
    assert a & b == b
    assert a | b == a
    assert a >= b
    assert a.issuperpattern(b)
    assert b <= a
    assert b.issubpattern(a)
    assert not a.issubpattern(b)

    a = bip.IntervalPattern('[1, 10)')
    b = bip.IntervalPattern('(1, 10]')
    meet = bip.IntervalPattern('[1, 10]')
    join = bip.IntervalPattern('(1, 10)')
    assert a & b == meet
    assert a | b == join

    assert str(a) == 'IntervalPattern([1.0, 10.0))'

    assert {a, b}  # Test if hashable


def test_ClosedIntervalPattern():
    a = bip.ClosedIntervalPattern((1, 10))
    a2 = bip.ClosedIntervalPattern('[1, 10]')
    assert a.value == (1, 10)
    assert a == a2

    with pytest.raises(AssertionError):
        bip.ClosedIntervalPattern('(10, 25)')
        bip.ClosedIntervalPattern('(10, 25]')
        bip.ClosedIntervalPattern('[10, 25)')

    a = bip.ClosedIntervalPattern('[-inf, ∞]')
    assert a._lower_bound == -math.inf
    assert a._upper_bound == math.inf

    a = bip.ClosedIntervalPattern('[1, 10]')
    b = bip.ClosedIntervalPattern('[1, 20]')
    assert a & b == b
    assert a | b == a
    assert a >= b
    assert a.issuperpattern(b)
    assert b <= a
    assert b.issubpattern(a)
    assert not a.issubpattern(b)

    assert str(a) == 'ClosedIntervalPattern([1.0, 10.0])'

    assert {a, b}  # Test if hashable


def test_NgramSetPattern():
    a = bip.NgramSetPattern({('hello', 'world')})
    a2 = bip.NgramSetPattern(['hello     world   '])
    assert a.value == {('hello', 'world')}
    assert a == a2

    a = bip.NgramSetPattern(['hello world', 'who is there'])
    b = bip.NgramSetPattern(['hello there'])
    meet = bip.NgramSetPattern(['hello', 'there'])
    join = bip.NgramSetPattern(['hello world', 'who is there', 'hello there'])
    assert a & b == meet
    assert a | b == join
    assert meet <= a
    assert a <= join

    a = bip.NgramSetPattern(['hello world', 'who is there'])
    assert str(a) == "NgramSetPattern({'who is there', 'hello world'})"

    assert {a, b, meet, join}  # Test if hashable
