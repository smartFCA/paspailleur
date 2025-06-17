import pytest

from paspailleur.pattern_structures import pattern_factory as pf, built_in_patterns as bip


def test_pattern_factory():
    new_pattern = pf.pattern_factory(bip.IntervalPattern)
    assert isinstance(new_pattern(42), bip.IntervalPattern)
    assert new_pattern.__name__ == "FactoredPattern"

    new_pattern = pf.pattern_factory(bip.IntervalPattern, 'CustomName')
    assert new_pattern.__name__ == "CustomName"

    new_pattern = pf.pattern_factory(bip.IntervalPattern, BoundsUniverse=tuple(range(5)))
    assert new_pattern.__name__ == "FactoredPattern"
    assert new_pattern.BoundsUniverse == tuple(range(5))

    with pytest.raises(ValueError):
        new_pattern = pf.pattern_factory(bip.IntervalPattern, RandomV = 42)

    new_pattern = pf.pattern_factory('IntervalPattern', BoundsUniverse=tuple(range(5)))
    assert isinstance(new_pattern(2), bip.IntervalPattern)
