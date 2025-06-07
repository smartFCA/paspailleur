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
    >>> objects_patterns = [Pattern(frozenset('abc')), Pattern(frozenset('abc')), Pattern(frozenset('abcd')), Pattern(frozenset('acde'))]
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
    >>> objects_patterns = [Pattern(frozenset('abc')), Pattern(frozenset('abc')), Pattern(frozenset('abcd')), Pattern(frozenset('acde'))]
    >>> from paspailleur.algorithms import base_functions as bfuncs
    >>> obj_to_patterns = bfuncs.group_objects_by_patterns(objects_patterns)
    >>> obj_ba = bitarray('1100')
    >>> bfuncs.intention(obj_ba, obj_to_patterns)
    Pattern(frozenset({'a', 'b', 'c'}))
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


def patternise_description(
        active_atoms: bitarray,
        atomic_patterns: list[Pattern], subatoms_order: list[bitarray],
        trusted_input: bool = False
) -> Pattern:
    """
    Reconstruct pattern from its atomic representation

    The function runs the join operation on `atomic_patterns` indexed by `active_atoms`,
    but provides some optimisations using `subatoms_order`.

    Important:
    The list of `atomic_patterns` should be topologically sorted.
    That is, for every pattern, all its subpatterns should have smaller indices.


    Parameters
    ----------
    active_atoms: bitarray
        Bitarray that represents the pattern-to-output in a binary format.
        That is, `active_atoms[i]` is True when `atomic_patterns[i]` is less precise than pattern-to-output.
        Should be the same length as the list of atomic_patterns.
    atomic_patterns: list[Pattern]
        The list of all atomic patterns.
    subatoms_order: list[bitarray]
        Partial order on the set of atomic patterns.
        Value `subatoms_order[i][j]` is True when `atomic_patterns[i]` is less precise than `atomic_patterns[j]`.
    trusted_input: bool, default=`False`
        A flag whether the list of atomic patterns is guaranteed to be topologically sorted.
        That is, if we know for sure, that for every pattern, all its subpatterns would have smaller indices.

    Returns
    -------
    pattern: Pattern
        Pattern obtained by joining `atomic_patterns` selected by `active_atoms`.

    """
    if not trusted_input:
        assert all(not subatoms[i:].any() for i, subatoms in enumerate(subatoms_order)), \
            ('Partial order `subatoms_order` should be topologically sorted. '
             'That is, every smaller elements should have smaller indices')

    min_pattern = atomic_patterns[0].get_min_pattern()
    if not active_atoms.any():
        return min_pattern

    active_atoms = bitarray(active_atoms)
    i = len(active_atoms)
    while i > 0 and active_atoms[:i].any():
        i = active_atoms.find(True, 0, i, right=True)
        active_atoms = active_atoms & ~subatoms_order[i]
    return reduce(atomic_patterns[0].__class__.join, (atomic_patterns[i] for i in active_atoms.search(True)))


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
    >>> objects_patterns = [Pattern(frozenset('abc')), Pattern(frozenset('abc')), Pattern(frozenset('abcd')), Pattern(frozenset('acde'))]
    >>> from paspailleur.algorithms import base_functions as bfuncs
    >>> obj_to_patterns = bfuncs.group_objects_by_patterns(objects_patterns)
    >>> bfuncs.minimal_pattern(obj_to_patterns)
    Pattern(frozenset({'a'}))
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
    >>> objects_patterns = [Pattern(frozenset('abc')), Pattern(frozenset('abc')), Pattern(frozenset('abcd')), Pattern(frozenset('acde'))]
    >>> from paspailleur.algorithms import base_functions as bfuncs
    >>> obj_to_patterns = bfuncs.group_objects_by_patterns(objects_patterns)
    >>> bfuncs.maximal_pattern(obj_to_patterns)
    Pattern(frozenset({'a', 'b', 'c', 'd', 'e'}))
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
    >>> objects_patterns = [Pattern(frozenset('abc')), Pattern(frozenset('abc')), Pattern(frozenset('abcd')), Pattern(frozenset('acde'))]
    >>> from paspailleur.algorithms import base_functions as bfuncs
    >>> obj_to_patterns = bfuncs.group_objects_by_patterns(objects_patterns)
    >>> bfuncs.group_objects_by_patterns(objects_patterns)
    {Pattern(frozenset({'a', 'b', 'c'})): bitarray('1100'), Pattern(frozenset({'a', 'b', 'c', 'd'})): bitarray('0010'), Pattern(frozenset({'a', 'c', 'd', 'e'})): bitarray('0001')}
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
    >>> objects_patterns = [Pattern(frozenset('abc')), Pattern(frozenset('abc')), Pattern(frozenset('abcd')), Pattern(frozenset('acde'))]
    >>> from paspailleur.algorithms import base_functions as bfuncs
    >>> obj_to_patterns = bfuncs.group_objects_by_patterns(objects_patterns)
    >>> pattern_order = bfuncs.order_patterns_via_extents(list(obj_to_patterns.items()))
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
    >>> from paspailleur.algorithms import base_functions as bfuncs
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
    >>> objects_patterns = [Pattern(frozenset('abc')), Pattern(frozenset('abc')), Pattern(frozenset('abcd')), Pattern(frozenset('acde'))]
    >>> from paspailleur.algorithms import base_functions as bfuncs
    >>> obj_to_patterns = bfuncs.group_objects_by_patterns(objects_patterns)
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


def iterate_antichains(
        descending_order: list[bitarray], max_length: int = None
) -> Generator[tuple[int, ...], bool, None]:
    """
    Iterate antichains of indices whose partial order is defined by `descending_order` parameter.

    Antichain is a term from Order theory that represents a set of incomparable elements.
    That is, a subset of indices {i, j, k, ..., n} makes an antichain when every pair of indices (i, j), (i, k), ...,
    represents a pair of incomparable elements: e.g. `descending_order[i][j] == descending_order[j, i] == False`.

    Important:
    Elements in `descending_order` should be lexicographically ordered.
    That is, for every i-th element, all its lesser elements should have lower indices: from 0 to i-1.


    Parameters
    ----------
    descending_order: list[bitarray]
        Defined the partial order of indices.
        Value `descending_order[i][j]==True` indicates that i-th element is greater than the j-th element.
    max_length: int, default = len(descending_order)
        Maximal length of an antichain to yield.

    Yields
    ------
    antichains_iterator: Generator[list[int], bool, None]
        Generator of antichains of the partial order defined by `descending_order`.
        The navigation can be controlled using boolean value into `antichains_iterator.send()`.
        If the passed value is `True` then the generator will pass through dominating antichains.

    Examples
    --------
    Use the function as a generator:
    >>> descending_order = [bitarray('0000'), bitarray('0100'), bitarray('1000'), bitarray('0000')]
    >>> list(iterate_antichains(descending_order))  # get the list of all possible antichains
    [(), (0,), (1,), (1, 0), (2,), (2, 1), (3,), (3, 0), (3, 1), (3, 1, 0), (3, 2), (3, 2, 1)]

    Control the navigation over antichains:
    >>> descending_order = [bitarray('0000'), bitarray('0100'), bitarray('1000'), bitarray('0000')]
    >>> iterator = iterate_antichains(descending_order)
    >>> iterator.send(None)  # send None value to get the first antichain which is the empty tuple
    ()
    >>> iterator.send(True)  # get the next antichain while saying True for antichains that dominate the empty antichain
    (0,)
    >>> iterator.send(False)  # get the next antichain while forbidding any antichain that has elements greater than the 0th
    (1,)
    >>> list(iterator)  # generate all antichains that are left to iterate
    [(3,), (3, 1)]

    The second iteration method has omitted all antichains that contain element 0.
    It has also skipped all antichains that contain element 2,
    because the 2nd element is defined as greater than the 0th: `descending_order[2][0]==True`.

    """
    assert not any(greaters[i+1:].any() for i, greaters in enumerate(descending_order)), \
        "`descending_order` should be lexicographically ordered."

    n_elements = len(descending_order)
    max_length = max_length if max_length is not None else n_elements


    def ba_from_indices(indices: tuple[int, ...], default: bitarray = bazeros(n_elements)) -> fbarray:
        ba = bitarray(default)
        for i in indices:
            ba[i] = True
        return fbarray(ba)

    descending_order = [ba_from_indices((i,), ba) for i, ba in enumerate(descending_order)]

    refine_antichain = yield tuple()
    if refine_antichain == False or max_length == 0:
        return
    forbidden_antichains: set[fbarray] = set()

    queue: list[tuple[tuple[int, ...], bitarray]] = [((i,), lesser) for i, lesser in enumerate(descending_order)][::-1]
    while queue:
        antichain, sub_elements = queue.pop()
        if any(basubset(forbidden, sub_elements) for forbidden in forbidden_antichains):
            continue

        refine_antichain = yield antichain
        if refine_antichain == False:
            forbidden_antichains.add(ba_from_indices(antichain))
            continue

        if len(antichain) == max_length:
            continue

        next_elements = [next_i for next_i in (~sub_elements).search(True, 0, min(antichain))]
        next_elements = next_elements[::-1]
        queue.extend([(antichain+(next_i,), sub_elements | descending_order[next_i]) for next_i in next_elements])
