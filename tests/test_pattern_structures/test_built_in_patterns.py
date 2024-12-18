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
    assert not (b <= a)

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



