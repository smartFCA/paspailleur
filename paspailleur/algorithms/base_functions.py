from collections import OrderedDict
from functools import reduce
from typing import Optional, Generator, Union, Any

from bitarray import bitarray, frozenbitarray as fbarray
from bitarray.util import zeros as bazeros, subset as basubset

from paspailleur.pattern_structures.pattern import Pattern


def extension(pattern: Pattern, objects_per_pattern: dict[Pattern, bitarray]) -> bitarray:
    """Return the set of objects whose patterns are more precise than `pattern`.

    `objects_per_pattern` matches objects' patterns with the objects themselves
    (in case some objects share the same patterns)
    """
    n_objects = len(list(objects_per_pattern.values())[0])
    empty_extent = bazeros(n_objects)
    super_patterns = (ptrn for ptrn in objects_per_pattern if pattern <= ptrn)
    sub_extents = (objects_per_pattern[ptrn] for ptrn in super_patterns)
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


def iter_patterns_ascending(
        patterns: Union[list[Pattern], OrderedDict[Pattern, Any]],
        greater_patterns_ordering: list[bitarray],
        controlled_iteration: bool = False
) -> Generator[Union[Pattern, tuple[Pattern, Any]], bool, None]:
    assert all(not greater_ptrns[:i].any() for i, greater_ptrns in enumerate(greater_patterns_ordering)), \
        'The list of `patterns` from the smaller to the greater patterns. ' \
        'So for every i-th pattern, there should be no greater patter among patterns[:i]'

    patterns_to_pass = ~bazeros(len(patterns))
    patterns_list = list(patterns)

    if controlled_iteration:
        yield  # Initialisation

    while patterns_to_pass.any():
        i = patterns_to_pass.find(True)
        patterns_to_pass[i] = False

        pattern = patterns_list[i]
        yielded_value = (pattern, patterns[pattern]) if isinstance(patterns, dict) else pattern
        go_more_precise = yield yielded_value
        if controlled_iteration and not go_more_precise:
            patterns_to_pass &= ~greater_patterns_ordering[i]
