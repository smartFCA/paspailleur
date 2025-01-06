from functools import reduce
from typing import Callable, Optional

from bitarray import bitarray, frozenbitarray as fbarray
from bitarray.util import zeros as bazeros, subset as basubset

from paspailleur.pattern_structures.pattern import Pattern


def extension(pattern: Pattern, objects_per_pattern: dict[Pattern, bitarray]) -> bitarray:
    n_objects = len(list(objects_per_pattern.values())[0])
    empty_extent = bazeros(n_objects)
    sub_extents = (extent for ptrn, extent in objects_per_pattern.items() if pattern <= ptrn)
    return reduce(fbarray.__or__, sub_extents, empty_extent)


def intention(objects: bitarray, objects_per_pattern: dict[Pattern, bitarray]) -> Optional[Pattern]:
    super_patterns = [ptrn for ptrn, irr_ext in objects_per_pattern.items()
                      if basubset(objects, irr_ext)] #if basubset(irr_ext, objects)]
    super_patterns = [ptrn for ptrn, irr_ext in objects_per_pattern.items() if (objects & irr_ext).any()]
    if super_patterns:
        first_pattern = super_patterns[0]
        return reduce(first_pattern.__class__.__and__, super_patterns, first_pattern)

    some_pattern = next(pattern for pattern in objects_per_pattern.keys())
    if some_pattern.max_pattern is not None:
        return some_pattern.max_pattern
    return maximal_pattern(objects_per_pattern)


def minimal_pattern(objects_per_pattern: dict[Pattern, bitarray]) -> Pattern:
    some_pattern = next(pattern for pattern in objects_per_pattern)
    if some_pattern.min_pattern is not None:
        return some_pattern.min_pattern

    return reduce(some_pattern.__class__.__and__, objects_per_pattern, some_pattern)


def maximal_pattern(objects_per_pattern: dict[Pattern, bitarray]) -> Pattern:
    some_pattern = next(pattern for pattern in objects_per_pattern)
    if some_pattern.max_pattern is not None:
        return some_pattern.max_pattern

    return reduce(some_pattern.__class__.__or__, objects_per_pattern, some_pattern)


def group_objects_by_patterns(objects_patterns: list[Pattern]) -> dict[Pattern, bitarray]:
    empty_extent = bazeros(len(objects_patterns))

    objects_by_patterns = dict()
    for g_idx, pattern in enumerate(objects_patterns):
        if pattern not in objects_by_patterns:
            objects_by_patterns[pattern] = empty_extent.copy()
        objects_by_patterns[pattern][g_idx] = True

    assert sum(objs.count() for objs in objects_by_patterns.values()) == len(objects_patterns)

    return objects_by_patterns
