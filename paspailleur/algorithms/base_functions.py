from collections import OrderedDict
from functools import reduce
from typing import Optional, Generator, Union, Any

from bitarray import bitarray, frozenbitarray as fbarray
from bitarray.util import zeros as bazeros, subset as basubset

from tqdm.auto import tqdm

from caspailleur.order import sort_intents_inclusion, inverse_order
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


def rearrange_indices(order_before: list[bitarray], elements_before: list, elements_after: list) -> list[bitarray]:
    els_before_idx_map = {element_before: idx for idx, element_before in enumerate(elements_before)}
    after_before_mapping = [els_before_idx_map[element_after] for element_after in elements_after]
    before_after_mapping = {before_idx: after_idx for after_idx, before_idx in enumerate(after_before_mapping)}

    order_after = []
    for after_idx, before_idx in enumerate(after_before_mapping):
        ba_before = order_before[before_idx]

        ba_after = ba_before & ~ba_before
        for idx in ba_before.search(True):
            ba_after[before_after_mapping[idx]] = True

        order_after.append(ba_after)

    return order_after


def order_patterns_via_extents(patterns_extents: list[tuple[Pattern, fbarray]], use_tqdm: bool = False) -> list[bitarray]:
    def group_patterns_by_extents(patterns_extent_idx: list[tuple[Pattern, int]]) -> list[list[Pattern]]:
        n_extents = max(ext_i for _, ext_i in patterns_extent_idx) + 1
        patterns_per_extents_: list[list[Pattern]] = [list() for _ in range(n_extents)]
        for pattern, extent_i in patterns_extent_idx:
            if not patterns_per_extents_[extent_i]:
                patterns_per_extents_[extent_i].append(pattern)
                continue

            # extent in patterns_per_extent, i.e. there are already some known patterns per extent
            equiv_patterns = patterns_per_extents_[extent_i]
            greater_patterns = (i for i, other in enumerate(equiv_patterns) if pattern <= other)
            first_greater_pattern = next(greater_patterns, len(equiv_patterns))
            patterns_per_extents_[extent_i].insert(first_greater_pattern, pattern)
        return patterns_per_extents_

    def sort_extents_subsumption(extents):
        extents = sorted(extents, key=lambda extent: (-extent.count(), tuple(extent.search(True))))
        extents_to_idx_map = {extent: idx for idx, extent in enumerate(extents)}

        empty_extent = extents[0] & ~extents[0]
        added_top, added_bottom = False, False
        if not extents[0].all():
            extents.insert(0, ~empty_extent)
            added_top = True
        if extents[-1].any():
            extents.append(empty_extent)
            added_bottom = True
        inversed_extents_subsumption_order = inverse_order(
            sort_intents_inclusion(extents[::-1], use_tqdm=False, return_transitive_order=True)[1])
        extents_subsumption_order = [ba[::-1] for ba in inversed_extents_subsumption_order[::-1]]

        if added_top:
            extents.pop(0)
            extents_subsumption_order = [ba[1:] for ba in extents_subsumption_order[1:]]
        if added_bottom:
            extents.pop(-1)
            extents_subsumption_order = [ba[:-1] for ba in extents_subsumption_order[:-1]]
        return extents, extents_to_idx_map, extents_subsumption_order

    def select_greater_patterns_candidates(
            pattern: Pattern, extent_i: int, pattern_idx_map: dict[Pattern, int],
            subextents_order: list[bitarray],
            ptrns_per_exts: list[list[Pattern]]
    ):
        greater_candidates = bazeros(len(pattern_idx_map))

        patterns_same_extent = ptrns_per_exts[extent_i]
        n_greater_patterns_same_extent = len(patterns_same_extent) - patterns_same_extent.index(pattern) - 1
        greater_candidates[pattern_idx+1: pattern_idx+n_greater_patterns_same_extent+1] = True

        for smaller_extent_idx in subextents_order[extent_i].search(True):
            patterns_smaller_extent = ptrns_per_exts[smaller_extent_idx]
            first_smaller_pattern_idx = pattern_idx_map[patterns_smaller_extent[0]]
            n_patterns_smaller_extent = len(patterns_smaller_extent)
            greater_candidates[first_smaller_pattern_idx:first_smaller_pattern_idx+n_patterns_smaller_extent] = True
        return greater_candidates

    def select_greater_patterns(
            pattern: Pattern, candidates_ba: bitarray,
            patterns_list: list[tuple[Pattern, int]], superpatterns_order: list[bitarray]
    ):
        greater_patterns = candidates_ba & ~candidates_ba

        candidates_ba = bitarray(candidates_ba)
        while candidates_ba.any():
            other_idx = candidates_ba.find(True)
            candidates_ba[other_idx] = False

            other = patterns_list[other_idx][0]
            if pattern < other:
                greater_patterns[other_idx] = True
                greater_patterns |= superpatterns_order[other_idx]
                candidates_ba &= ~superpatterns_order[other_idx]
        return greater_patterns

    extents_list, extents_idx_map, extents_order = sort_extents_subsumption({ext for _, ext in patterns_extents})
    patterns_extents: list[tuple[Pattern, int]] = [(pattern, extents_idx_map[extent]) for pattern, extent in patterns_extents]

    patterns_per_extents: list[list[Pattern]] = group_patterns_by_extents(patterns_extents)
    patterns_list: list[tuple[Pattern, int]] = [
        (pattern, extent_i)
        for extent_i, patterns_same_extent, in enumerate(patterns_per_extents)
        for pattern in patterns_same_extent
    ]
    patterns_idx_map: dict[Pattern, int] = {pattern: i for i, (pattern, _) in enumerate(patterns_list)}

    patterns_order: list[bitarray] = [None for _ in patterns_extents]
    patterns_iterator = reversed(range(len(patterns_list)))
    if use_tqdm:
        patterns_iterator = tqdm(patterns_iterator, total=len(patterns_list),
                                 disable=not use_tqdm, desc='Compute order of patterns')
    for pattern_idx in patterns_iterator:
        pattern, extent_idx = patterns_list[pattern_idx]

        # select patterns that might be greater than the current one
        patterns_to_test = select_greater_patterns_candidates(pattern, extent_idx,
                                                              patterns_idx_map, extents_order, patterns_per_extents)

        # find patterns that are greater than the current one
        super_patterns = select_greater_patterns(pattern, patterns_to_test, patterns_list, patterns_order)
        patterns_order[pattern_idx] = super_patterns

    patterns_order = rearrange_indices(patterns_order, [p for p, _ in patterns_list], [p for p, _ in patterns_extents])
    return patterns_order
