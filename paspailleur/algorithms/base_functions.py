from collections import OrderedDict
from functools import reduce
from typing import Optional, Generator, Union, Any

from bitarray import bitarray, frozenbitarray as fbarray
from bitarray.util import zeros as bazeros, subset as basubset

from tqdm.auto import tqdm

from caspailleur.order import sort_intents_inclusion, inverse_order
from paspailleur.pattern_structures.pattern import Pattern


def extension(pattern: Pattern, objects_per_pattern: dict[Pattern, bitarray]) -> bitarray:
    """
    Return the extent of a pattern (a set of objects whose patterns are more precise than pattern).

    Parameters
    ----------
    pattern: Pattern
        The pattern whose extent is computed.
    objects_per_pattern: dict[Pattern, bitarray]
        Matches patterns to bitarrays representing associated objects.

    Returns
    -------
    extent: bitarray
        A bitarray representing the extent of the input pattern.
    
    Examples
    --------
    >>> from paspailleur.pattern_structures.patterns import ItemSetPattern
    >>> objects_patterns = [Pattern(frozenset('abc')), Pattern(frozenset('abc')), Pattern(frozenset('abcd')), Pattern(frozenset('acde'))]
    #  or simply:
    # >>> objects_patterns = [Pattern(frozenset(x)) for x in ['abc', 'abc', 'abcd', 'acde']]
    >>> from paspailleur.algorithms import base_functions as bfuncs
    >>> obj_to_patterns = bfuncs.group_objects_by_patterns(objects_patterns)
    >>> bfuncs.extension(p, obj_to_patterns)
    bitarray('1111')
    
    Notes
    -----
    The "objects_per_pattern" dictionary can be created from objects' descriptions using "group_objects_by_patterns" function defined below.
    """
    n_objects = len(list(objects_per_pattern.values())[0])
    empty_extent = bazeros(n_objects)
    super_patterns = (ptrn for ptrn in objects_per_pattern if pattern <= ptrn)
    sub_extents = (objects_per_pattern[ptrn] for ptrn in super_patterns)
    return reduce(fbarray.__or__, sub_extents, empty_extent)


def intention(objects: bitarray, objects_per_pattern: dict[Pattern, bitarray]) -> Optional[Pattern]:
    """
    Compute the intent of a given set of objects.

    Parameters
    ----------
    objects: bitarray
        A bitarray representing selected objects.
    objects_per_pattern: dict[Pattern, bitarray]
        A mapping from patterns to object bitarrays.

    Returns
    -------
    intent: Optional[Pattern]
        The most specific pattern shared by all objects, if any.

    Examples
    --------
    >>> from bitarray.util import zeros
    >>> obj_ba = zeros(len(objects_names))
    >>> obj_ba[0] = obj_ba[1] = 1
    >>> bfuncs.intention(obj_ba, obj_to_patterns)
    ItemSetPattern({'Hiking', 'Observing Nature', 'Sightseeing Flights'})
    """
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
    """
    Compute the minimal pattern across all object patterns.

    Parameters
    ----------
    objects_per_pattern: dict[Pattern, bitarray]
        Mapping from patterns to object bitarrays.

    Returns
    -------
    minimal: Pattern
        The minimal pattern.

    Examples
    --------
    >>> bfuncs.minimal_pattern(obj_to_patterns)
    ItemSetPattern({'Hiking'})
    """
    some_pattern = next(pattern for pattern in objects_per_pattern)
    if some_pattern.min_pattern is not None:
        return some_pattern.min_pattern

    return reduce(some_pattern.__class__.__and__, objects_per_pattern, some_pattern)


def maximal_pattern(objects_per_pattern: dict[Pattern, bitarray]) -> Pattern:
    """
    Compute the maximal pattern across all object patterns.

    Parameters
    ----------
    objects_per_pattern: dict[Pattern, bitarray]
        Mapping from patterns to object bitarrays.

    Returns
    -------
    maximal: Pattern
        The maximal pattern.

    Examples
    --------
    >>> bfuncs.maximal_pattern(obj_to_patterns)
    ItemSetPattern({'Hiking', 'Sightseeing Flights', 'Jet Boating', 'Wildwater Rafting', 'Bungee Jumping'})
    """
    some_pattern = next(pattern for pattern in objects_per_pattern)
    if some_pattern.max_pattern is not None:
        return some_pattern.max_pattern

    return reduce(some_pattern.__class__.__or__, objects_per_pattern, some_pattern)


def group_objects_by_patterns(objects_patterns: list[Pattern]) -> dict[Pattern, bitarray]:
    """
    Group objects by their associated patterns.

    Parameters
    ----------
    objects_patterns: list[Pattern]
        A list where each element corresponds to a pattern describing an object.

    Returns
    -------
    objects_by_patterns: dict[Pattern, bitarray]
        Dictionary mapping patterns to bitarrays indicating which objects correspond to them.

    Examples
    --------
    >>> bfuncs.group_objects_by_patterns(objects_patterns)
    {ItemSetPattern({...}): bitarray(...), ...}
    """
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
    """
    Iterate through patterns in ascending order of generalization.

    Parameters
    ----------
    patterns: Union[list[Pattern], OrderedDict[Pattern, Any]]
        List or ordered dict of patterns.
    greater_patterns_ordering: list[bitarray]
        Ordering information of which patterns are greater.
    controlled_iteration: bool, optional
        If True, allow step-wise iteration with external control (default is False).

    Yields
    ------
    Generator[Union[Pattern, tuple[Pattern, Any]], bool, None]
        Each pattern or pattern-value pair, controlled by input from send().

    Examples
    --------
    >>> pattern_order = order_patterns_via_extents(list(obj_to_patterns.items()))
    >>> for pattern in bfuncs.iter_patterns_ascending(list(obj_to_patterns), pattern_order):
    print(pattern)
    """
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
    """
    Rearrange a list of orderings after element reordering.

    Parameters
    ----------
    order_before: list[bitarray]
        The original ordering as bitarrays.
    elements_before: list
        The elements before reordering.
    elements_after: list
        The elements after reordering.

    Returns
    -------
    order_after: list[bitarray]
        Reordered list of bitarrays.

    Examples
    --------
    >>> before = [bitarray('010'), bitarray('001'), bitarray('000')]
    >>> elems_before = ['A', 'B', 'C']
    >>> elems_after = ['C', 'A', 'B']
    >>> bfuncs.rearrange_indices(before, elems_before, elems_after)
    [bitarray('001'), bitarray('100'), bitarray('000')]
    """
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
    """
    Generate the partial order of patterns based on their extents.

    This function generates the partial order of patterns using extents for optimising the algorithm.
    It returns for each pattern a bitarray indicating which other patterns are more general.

    Parameters
    ----------
    patterns_extents: list of tuple[Pattern, fbarray]
        List of patterns and their associated extents.
    use_tqdm: bool, optional
        If True, display a progress bar (default is False).

    Returns
    -------
    patterns_order: list[bitarray]
        A list of bitarrays representing the ordered patterns based on their extents.

    Examples
    --------
    >>> pattern_extents = list(obj_to_patterns.items())
    >>> order = bfuncs.order_patterns_via_extents(pattern_extents)
    >>> order[0]
    bitarray('010')
    """
    def group_patterns_by_extents(patterns_extent_idx: list[tuple[Pattern, int]]) -> list[list[Pattern]]:
        """
        Group patterns by their corresponding extent index.

        Parameters
        ----------
        patterns_extent_idx: list of tuple[Pattern, int]
            List of patterns paired with their extent indices.

        Returns
        -------
        list[list[Pattern]]
            Grouped patterns ordered by extent index.
        """
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
        """
        Sort extents by size and determine their subsumption (containment) order.

        Parameters
        ----------
        extents: list[bitarray]
            List of extents.

        Returns
        -------
        sorted_extents: tuple
            Sorted extents, a map of extents to indices, and a list of bitarrays indicating extent subsumption.
        """
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
        """ 
        Select candidate patterns that may be more general than the given pattern.
        
        Parameters
        ----------
        pattern: Pattern
            The reference pattern.
        extent_i: int
            Index of the pattern's extent.
        pattern_idx_map: dict[Pattern, int]
            Mapping from pattern to its index.
        subextents_order: list[bitarray]
            Subsumption ordering of extents.
        ptrns_per_exts: list[list[Pattern]]
            Patterns grouped by extent.

        Returns
        -------
        greater_candidates: bitarray
            Bitarray representing indices of candidate greater patterns.
        """
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
        """
        Filter true greater patterns from candidate patterns.

        Parameters
        ----------
        pattern: Pattern
            The base pattern.
        candidates_ba: bitarray
            Bitarray of candidate pattern indices.
        patterns_list: list of tuple[Pattern, int]
            Patterns and their extent indices.
        superpatterns_order: list[bitarray]
            Current known order of patterns.

        Returns
        -------
        greater_patterns: bitarray
            Bitarray indicating which candidate patterns are truly more general.
        """
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
