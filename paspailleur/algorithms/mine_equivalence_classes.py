import heapq
from functools import reduce, partial
from itertools import takewhile
from collections import OrderedDict, deque

from caspailleur import inverse_order
from tqdm.auto import tqdm
from typing import Iterator, Generator, Collection, Iterable, Optional, Union
from bitarray import bitarray, frozenbitarray as fbarray
from bitarray.util import zeros as bazeros, subset as basubset, count_and

from paspailleur.algorithms import base_functions as bfuncs
from paspailleur.algorithms.base_functions import iterate_antichains
from paspailleur.pattern_structures.pattern import Pattern


# TODO: Rewrite Lindig algorithm with the new Pattern Structure architecture
def list_intents_via_Lindig_complex(data: list, pattern_structure) -> list['PatternDescription']:
    """Get the list of intents of pattern concepts from `data` described by `pattern_structure` running Lindig algorithm
    from "Fast Concept Analysis" by Christian Lindig, Harvard University, Division of Engineering and Applied Sciences

    WARNING: The function does not work at the moment as
    it was written for the outdated version of PatternStructure code architecture.

    Parameters
    ----------
    data:
        list of objects described by a pattern structure
    pattern_structure:
        type of pattern structure related to data

    Returns
    -------
    Lattice_data_intents:
        list of intents of pattern concepts
    """

    class NotFound(Exception):
        pass

    def compute_bits_intersection(bits: list[bitarray], len_bitarray):
        """
        Compute the intersection of a list of bitarrays.

        Parameters
        ----------
        bits: list[bitarray]
            List of bitarrays to intersect.
        len_bitarray: int
            Length of the bitarrays.

        Returns
        -------
        intersection: bitarray
            Bitarray resulting from intersecting all input bitarrays.
        """
        if bits == []:
            return(bitarray([1 for _ in range(len_bitarray)]))
        bit = bits[0]
        for obj in bits[1:]:
            bit = bit & obj
        return(bit)

    def find_upper_neighbors(data: list, concept_extent: list, objects_indices: list):
        """
        Find upper neighbors of a given concept extent.

        Parameters
        ----------
        data: list
            The dataset.
        concept_extent: list
            Current concept extent.
        objects_indices: list
            All object indices.

        Returns
        -------
        neighbors: list[list[int]]
            List of upper neighbor extents.
        """
        min_set = [obj for obj in objects_indices if obj not in concept_extent]
        concept_extent_values = [data[i] for i in concept_extent]
        neighbors = []

        for g in [obj for obj in objects_indices if obj not in concept_extent]:
            B1 = ps.intent(concept_extent_values + [data[g]])
            A1 = list(ps.extent(B1, data))
            if len(set(min_set) & set([obj for obj in A1 if obj not in concept_extent and obj not in [g]])) == 0:
                neighbors.append(A1)
            else:
                min_set.remove(g)
        return neighbors

    def find_next_concept_extent(data: list, concept_extent: list, List_extents: list):
        """
        Find the next concept extent in the lattice.

        Parameters
        ----------
        data: list
            The dataset.
        concept_extent: list
            The current extent.
        List_extents: list
            List of known extents.

        Returns
        -------
        next_concept_extent: list
            The next extent in the concept lattice.
        """
        bin_col_names, rows = ps.binarize(data)
        next_concept_extent = None
        concept_extent_bit = compute_bits_intersection([rows[i] for i in concept_extent], len(rows[0]))
        for extent in List_extents:
            extent_bit = compute_bits_intersection([rows[i] for i in extent], len(rows[0]))
            if extent_bit < concept_extent_bit and (next_concept_extent is None or extent_bit > next_concept_extent_bit):
                next_concept_extent = extent
                next_concept_extent_bit = compute_bits_intersection([rows[i] for i in next_concept_extent], len(rows[0]))
        if next_concept_extent is not None:
            return next_concept_extent
        raise NotFound("Next concept not found in Lattice")

    ps = pattern_structure
    Lattice_data_extents = []  # extents set
    concept_extent = []  # Initial concept extent
    objects_indices = list(ps.extent([], data))
    Lattice_data_extents.append(concept_extent)  # Insert the initial concept extent into Lattice

    while True:
        for parent in find_upper_neighbors(data, concept_extent, objects_indices):
            if parent not in Lattice_data_extents:
              Lattice_data_extents.append(parent)

        try:
            concept_extent = find_next_concept_extent(data, concept_extent, Lattice_data_extents)
        except NotFound:
            break

    Lattice_data_intents = []
    for i in range(len(Lattice_data_extents)):
        Lattice_data_intents.append(ps.intent([data[j] for j in Lattice_data_extents[i]]))

    return Lattice_data_intents


def iter_intents_via_ocbo(
        objects_patterns: list[Pattern]
    ) -> Iterator[tuple[Pattern, bitarray]]:
    """
    Iterate intents by applying the object-wise Close By One algorithm.

    References
    ----------
    Kuznetsov, S. O. (1993). A fast algorithm for computing all intersections of objects from an arbitrary semilattice. Nauchno-Tekhnicheskaya Informatsiya Seriya 2-Informatsionnye Protsessy i Sistemy, (1), 17-20.
    
    Parameters
    ----------
    objects_patterns: list[Pattern]
        List of patterns, one per object.

    Returns
    -------
    intent_extent_pairs: Iterator[tuple[Pattern, bitarray]]
        Yields each pattern (intent) and its extent.
    """
    objects_per_pattern = bfuncs.group_objects_by_patterns(objects_patterns)

    n_objects = len(objects_patterns)
    # create a stack of pairs: 'known_extent', 'object_to_add'
    stack: list[tuple[bitarray, int]] = [(bazeros(n_objects), -1)]
    while stack:
        known_extent, object_to_add = stack.pop()
        proto_extent = known_extent.copy()
        if object_to_add >= 0:
            proto_extent[object_to_add] = True

        intent = bfuncs.intention(proto_extent, objects_per_pattern)
        extent = bfuncs.extension(intent, objects_per_pattern)

        has_objects_not_in_lex_order = (object_to_add >= 0) and (extent & ~proto_extent)[:object_to_add].any()
        if has_objects_not_in_lex_order:
            continue

        yield intent, extent
        next_steps = [(extent, g) for g in extent.search(False, object_to_add+1)]
        stack.extend(next_steps[::-1])


def iter_all_patterns_ascending(
        atomic_patterns_extents: OrderedDict[Pattern, bitarray],
        min_support: int = 0, depth_first: bool = True,
        controlled_iteration: bool = False,
    ) -> Generator[tuple[Pattern, bitarray], bool, None]:
    """
    Iterate all patterns in ascending order of precision using atomic patterns.

    Parameters
    ----------
    atomic_patterns_extents: OrderedDict[Pattern, bitarray]
        Atomic patterns and their extents.
    min_support: int, optional
        Minimum support for yielded patterns.
    depth_first: bool, optional
        Whether to use depth-first traversal (default True).
    controlled_iteration: bool, optional
        If True, allows caller to control traversal.

    Returns
    -------
    pattern_extent_stream: Generator[tuple[Pattern, bitarray], bool, None]
        Yields each pattern and its extent.

    Examples
    --------
    >>> from paspailleur.pattern_structures.built_in_patterns import ItemSetPattern
    >>> atomic_patterns_extents = OrderedDict([
    ...    (ItemSetPattern({'A'}), bitarray('1110')),
    ...    (ItemSetPattern({'B'}), bitarray('1101')),
    ...    (ItemSetPattern({'C'}), bitarray('1011'))
    ... ])
    
    --- Non-controlled iteration ---
    >>> for p, e in mec.iter_all_patterns_ascending(atomic_patterns_extents):
    ...    print(p, e)
    
    --- Controlled iteration ---
    >>> gen = mec.iter_all_patterns_ascending(atomic_patterns_extents, controlled_iteration=True)
    >>> next(gen)  # initialize
    >>> refine_pattern = True
    >>> while True:
    ...     try:
    ...         pattern, extent = gen.send(refine_pattern)  # control exploration
    ...     except StopIteration:
    ...         break
    """
    # The algo is inspired by CloseByOne
    # For the start, let us just rewrite CloseByOne algorithm
    # with no though on how to optimise it for this particular case
    atomic_patterns = list(atomic_patterns_extents)
    first_pattern = atomic_patterns[0]
    total_extent = atomic_patterns_extents[first_pattern] | ~atomic_patterns_extents[first_pattern]
    meet_func, join_func = first_pattern.__class__.__and__, first_pattern.__class__.__or__

    min_pattern = reduce(meet_func, atomic_patterns) if first_pattern.min_pattern is None else first_pattern.min_pattern
    if controlled_iteration:
        yield  # for initialisation

    # create a stack of pairs: 'involved_patterns', 'pattern_to_add'
    n_atoms = len(atomic_patterns_extents)
    stack: list[tuple[bitarray, int]] = [(bazeros(n_atoms), -1)]
    while stack:
        involved_patterns, pattern_to_add = stack.pop()
        proto_closure = involved_patterns.copy()
        if pattern_to_add != -1:
            proto_closure[pattern_to_add] = True

        proto_closure_extents = (atomic_patterns_extents[atomic_patterns[i]] for i in proto_closure.search(True))
        extent = reduce(bitarray.__and__, proto_closure_extents, total_extent)
        if extent.count() < min_support:
            continue

        new_pattern = reduce(join_func, (atomic_patterns[i] for i in proto_closure.search(True)), min_pattern)
        has_atoms_not_in_lex_order = (pattern_to_add != -1) and any(
            atomic_patterns[i] <= new_pattern
            for i in involved_patterns.search(False, 0, pattern_to_add)
        )
        if has_atoms_not_in_lex_order:
            continue

        go_more_precise = yield new_pattern, extent  # if controlled_iteration is False, go_more_precise = None
        if go_more_precise is not None and not go_more_precise:
            continue

        closure = proto_closure.copy()
        for i in proto_closure.search(False, pattern_to_add + 1):
            closure[i] = atomic_patterns[i] <= new_pattern
        previous_pattern_next_steps = [(closure, i) for i in closure.search(False, pattern_to_add + 1)][::-1]
        stack = stack + previous_pattern_next_steps if depth_first else previous_pattern_next_steps + stack


def list_stable_extents_via_gsofia(
        atomic_patterns_iterator: Generator[tuple[Pattern, fbarray], bool, None],
        min_delta_stability: int = 0,
        n_stable_extents: int = None,
        min_supp: int = 0,
        use_tqdm: bool = False,
        n_atomic_patterns: int = None
    ) -> set[fbarray]:
    """
    Identify stable extents using the gSofia algorithm.

    References
    ----------
    Efficient Mining of Subsample-Stable Graph Patterns by Aleksey Buzmakov; Sergei O. Kuznetsov; Amedeo Napoli. Published in: 2017 IEEE International Conference on Data Mining (ICDM)

    Parameters
    ----------
    atomic_patterns_iterator: Generator
        Generator yielding atomic patterns and their extents.
    min_delta_stability: int, optional
        Minimum delta stability to accept an extent.
    n_stable_extents: int, optional
        Maximum number of stable extents to return.
    min_supp: int, optional
        Minimum support required for an extent.
    use_tqdm: bool, optional
        Whether to show progress bar.
    n_atomic_patterns: int, optional
        Number of atomic patterns expected.

    Returns
    -------
    stable_extents: set[fbarray]
        Set of stable extents.

    Notes
    -----
    The extents returned with n_stable_extents parameter are not necessarily the n most stable extents. They are just n extents that seem to be the very stable.
    """
    def maximal_bitarrays(bas: Collection[fbarray]) -> set[fbarray]:
        """
        Remove any bitarray that is a subset of another.

        Parameters
        ----------
        bas: Collection[fbarray]
            Collection of candidate bitarrays.

        Returns
        -------
        max_set: set[fbarray]
            Set of bitarrays that are maximal under subset inclusion.
        """
        bas = sorted(bas, key=lambda ba: ba.count(), reverse=True)
        i = 0
        while i < len(bas):
            ba = bas[i]
            has_subarray = any(basubset(ba, bigger_ba) for bigger_ba in bas[:i])
            if has_subarray:
                del bas[i]
            else:
                i += 1
        return set(bas)

    def n_most_stable_extents(
            extents_data: Iterable[tuple[fbarray, tuple[int, set[fbarray]]]],
            n_most_stable=n_stable_extents
        ) -> list[tuple[fbarray, tuple[int, set[fbarray]]]]:
        """
        Select the top-N most stable extents based on delta index.

        Parameters
        ----------
        extents_data: Iterable[tuple[fbarray, tuple[int, set[fbarray]]]]
            A list of tuples mapping an extent to its delta and children.
        n_most_stable: int, optional
            The number of stable extents to retain.

        Returns
        -------
        most_stable: list[tuple[fbarray, tuple[int, set[fbarray]]]]
            The N most stable extents.
        """
        extents_data = list(extents_data)
        most_stable_extents = heapq.nlargest(n_most_stable, extents_data, key=lambda x: x[1][0])

        # remove least-stable most stable extents if some of the left out extents have the same stability
        thold = most_stable_extents[-1][1][0]
        n_borderline_most_stable = sum(1 for _ in takewhile(lambda x: x[1][0] == thold, reversed(most_stable_extents)))
        n_borderline_total = sum(delta == thold for _, (delta, _) in extents_data)
        if n_borderline_most_stable < n_borderline_total:
            most_stable_extents = most_stable_extents[:-n_borderline_most_stable]
        return most_stable_extents

    def init_new_pattern(
            new_extent: fbarray, new_atomic_extent: fbarray, old_children: Collection[fbarray],
            min_stability=min_delta_stability
        ) -> tuple[Optional[int], Optional[set[fbarray]]]:
        """
        Compute the delta and children of a new stable extent.

        Parameters
        ----------
        new_extent: fbarray
            The new extent being considered.
        new_atomic_extent: fbarray
            The extent of the atomic pattern currently being processed.
        old_children: Collection[fbarray]
            Children of the parent extent.
        min_stability: int
            Minimum allowed delta stability.

        Returns
        -------
        stability_data: tuple[Optional[int], Optional[set[fbarray]]]
            A tuple containing delta stability value and its valid children.
        """
        # Find the delta-index of the new extent and its children extents, aka "InitNewPattern" in the gSofia paper
        new_delta, new_children = new_extent.count(), []
        for child in old_children:
            child_new = child & new_atomic_extent
            new_delta = min(new_delta, new_extent.count() - child_new.count())
            if new_delta < min_stability:
                return new_delta, None
            new_children.append(child_new)
        return new_delta, maximal_bitarrays(new_children)

    if not atomic_patterns_iterator.gi_suspended:
        next(atomic_patterns_iterator)
    atomic_patterns_iterator = atomic_patterns_iterator
    pbar = tqdm(total=n_atomic_patterns, disable=not use_tqdm, desc='gSofia algorithm')

    # dict: extent => (delta_index, children_extents)
    stable_extents: dict[fbarray, tuple[int, set[fbarray]]] = dict()
    refine_previous_pattern: bool = True

    # special treatment for the first atomic pattern
    atomic_pattern, atomic_extent = atomic_patterns_iterator.send(refine_previous_pattern)
    top_extent = atomic_extent | ~atomic_extent
    n_objects = len(top_extent)
    stable_extents[top_extent] = n_objects - atomic_extent.count(), {atomic_extent}
    stable_extents[atomic_extent] = atomic_extent.count(), set()

    while True:
        try:
            atomic_pattern, atomic_extent = atomic_patterns_iterator.send(refine_previous_pattern)
        except StopIteration:
            break

        old_stable_extents, stable_extents = dict(stable_extents), dict()
        refine_previous_pattern = False
        for extent, (delta, children) in old_stable_extents.items():
            # Create new extent
            extent_new: fbarray = extent & atomic_extent
            if extent_new == extent:
                stable_extents[extent] = delta, children
                refine_previous_pattern = True
                continue

            # Update the stability of the old extent given its new child: `extent_new`
            delta = min(delta, extent.count() - extent_new.count())
            if delta >= min_delta_stability:
                stable_extents[extent] = delta, children | {extent_new}

            # Skip the new extent if it is too small
            if extent_new.count() < min_supp:
                # the pattern is to rare, so all refined (i.e. more precise) objects_patterns would be even rarer
                continue

            delta_new, children_new = init_new_pattern(extent_new, atomic_extent, children, min_delta_stability)
            if delta_new < min_delta_stability:  # Skip the new extent if it is too unstable
                continue

            # At this point we know that `extent_new` is big enough and is stable enough
            stable_extents[extent_new] = (delta_new, children_new)
            refine_previous_pattern = True

        # after generating all new stable extents
        if n_stable_extents is not None and len(stable_extents) > n_stable_extents:
            stable_extents = dict(n_most_stable_extents(stable_extents.items(), n_stable_extents))
        pbar.update(1)

    pbar.close()

    return set(stable_extents)


def iter_keys_of_pattern(
        pattern: Pattern,
        atomic_patterns: OrderedDict[Pattern, fbarray],
        max_length: Optional[int] = None
    ) -> Iterator[Pattern]:
    """
    Yield key patterns that generate the same extent as the given pattern.

    Parameters
    ----------
    pattern: Pattern
        The target pattern.
    atomic_patterns: OrderedDict[Pattern, fbarray]
        Atomic patterns and their extents.
    max_length: Optional[int], optional
        Maximum length of key patterns.

    Returns
    -------
    keys: Iterator[Pattern]
        Iterator of key patterns.
    """
    # second condition implies the first one. But the first one is easier to test
    atoms_to_iterate = [atom for atom in atomic_patterns if atom <= pattern]
    if not atoms_to_iterate:
        yield pattern.min_pattern if pattern.min_pattern is not None else pattern
        return

    extent = reduce(fbarray.__and__, (atomic_patterns[atom] for atom in atoms_to_iterate))
    pattern_iterator = iter_all_patterns_ascending(
        OrderedDict([(p, atomic_patterns[p]) for p in atoms_to_iterate]),
        min_support=extent.count(), depth_first=False, controlled_iteration=True
    )
    next(pattern_iterator)  # initialise the iterator

    go_more_precise = True
    found_keys = []
    while True:
        try:
            key_candidate, candidate_extent = pattern_iterator.send(go_more_precise)
        except StopIteration:
            break

        if not basubset(extent, candidate_extent):
            go_more_precise = False
            continue

        appropriate_length = True if max_length is None else len(key_candidate) < max_length

        # extent is a subset of candidate_extent
        if candidate_extent != extent:
            go_more_precise = appropriate_length
            continue

        # candidate_extent == extent
        more_precise_than_found_key = any(found_key <= key_candidate for found_key in found_keys)
        if not more_precise_than_found_key:
            yield key_candidate
            go_more_precise = False
            found_keys.append(key_candidate)

        go_more_precise &= appropriate_length


def iter_keys_of_patterns(
        patterns: list[Pattern],
        atomic_patterns: OrderedDict[Pattern, fbarray],
        max_length: Optional[int] = None
    ) -> Iterator[tuple[Pattern, int]]:
    """
    Yield key patterns for a list of patterns, maintaining index association.

    atomic_patterns should be sorted in topological order.
    So every i-th atomic pattern should be not-smaller than any previous (1, 2, ..., i-i) atomic pattern

    Parameters
    ----------
    patterns: list[Pattern]
        List of patterns to generate keys for.
    atomic_patterns: OrderedDict[Pattern, fbarray]
        Atomic patterns and their extents.
    max_length: Optional[int], optional
        Maximum key length.

    Returns
    -------
    keys_with_index: Iterator[tuple[Pattern, int]]
        Iterator of (key, original pattern index) tuples.
    """
    n_objects = len(atomic_patterns[next(p for p in atomic_patterns)])
    top_extent = fbarray(~bazeros(n_objects))

    atoms_to_iterate = []
    patterns_extents = {pattern: top_extent for pattern in patterns}
    for atom, atom_extent in atomic_patterns.items():
        activated_atom = False
        for pattern in patterns_extents:
            if not atom <= pattern:
                continue
            patterns_extents[pattern] = patterns_extents[pattern] & atom_extent
            activated_atom = True
        if activated_atom:
            atoms_to_iterate.append(atom)

    if not atoms_to_iterate:
        min_pattern_candidates = [p.min_pattern if p.min_pattern is not None else p for p in patterns]
        min_pattern = reduce(patterns[0].__class__.__and__, min_pattern_candidates)
        for pattern_idx in range(len(patterns)):
            yield min_pattern, pattern_idx
        return

    patterns_per_extents = {}
    for i, pattern in enumerate(patterns):
        extent = patterns_extents[pattern]
        if extent not in patterns_per_extents:
            patterns_per_extents[extent] = []
        patterns_per_extents[extent].append(i)

    min_support = min(extent.count() for extent in patterns_per_extents)
    pattern_iterator = iter_all_patterns_ascending(
        OrderedDict([(atom, atomic_patterns[atom]) for atom in atoms_to_iterate]),
        min_support=min_support, depth_first=False, controlled_iteration=True
    )
    next(pattern_iterator)  # initialise the iterator

    keys_per_pattern: list[list[Pattern]] = [[] for _ in patterns]
    go_more_precise = True
    while True:
        try:
            key_candidate, candidate_extent = pattern_iterator.send(go_more_precise)
        except StopIteration:
            break
        candidate_extent = fbarray(candidate_extent)

        if not any(basubset(extent, candidate_extent) for extent in patterns_per_extents):
            # not a key of any specified pattern, and no greater pattern is a key of any given pattern
            go_more_precise = False
            continue

        appropriate_length = True if max_length is None else len(key_candidate) < max_length

        if candidate_extent not in patterns_per_extents:
            # not a key of any specified pattern, but some greater pattern might be a key
            go_more_precise = appropriate_length
            continue

        # candidate_extent in patterns_per_extents, so key_candidate might be a key of some specified pattern
        for pattern_i in patterns_per_extents[candidate_extent]:
            pattern = patterns[pattern_i]
            if key_candidate <= pattern:
                if not any(found_key <= key_candidate for found_key in keys_per_pattern[pattern_i]):
                    yield key_candidate, pattern_i
                    keys_per_pattern[pattern_i].append(key_candidate)
        go_more_precise = any(basubset(extent, candidate_extent) and extent != candidate_extent
                              for extent in patterns_per_extents)

        go_more_precise &= appropriate_length


def iter_keys_of_patterns_via_atoms(
        patterns: list[tuple[Pattern, fbarray]],
        atomic_patterns: OrderedDict[Pattern, fbarray],
        subatoms_order: list[fbarray] = None,
        max_length: int = None,
        use_tqdm: bool = False
) -> Iterator[tuple[Pattern, int]]:
    """
    Yield the least precise patterns (aka keys) that describe the same extent as `patterns`

    Parameters
    ----------
    patterns: list[tuple[Pattern, fbarray]]
        A list of target patterns and their extents
    atomic_patterns: OrderedDict[Pattern, fbarray]
        Atomic patterns (and their extents) that will be used for finding keys.
    subatoms_order: list[fbarray], optional
        Partial order of `atomic_patterns` represented with list of frozenbitarrays.
        The value `subatoms_order[i][j] == True` means that j-th atomic pattern is less precise than i-th atomic pattern.
        If the value is not provided (i.e. equals to `None`), then the partial order will be computed inside this function.
    max_length: Optional[int], default = len(atomic_patterns)
        Maximum number of atomic pattern that a key can consist of.
        This parameter can be used for "early-stopping" to avoid generating too complex keys.
    use_tqdm: bool, default = False
        Flag whether to show tqdm progress bar or not.


    Yields
    ------
    key: Pattern
        One of the least precise patterns that describe the same extent as i-th provided pattern (identified by `pattern_index`)
    pattern_index: int
        Index of the provided pattern described by `key`

    """
    def extract_subatoms_data(
            patterns_: list[tuple[Pattern, fbarray]], atomic_patterns_: OrderedDict[Pattern, fbarray],
            subatoms_order_: list[fbarray],
    ) -> tuple[list[Pattern], list[fbarray], list[fbarray], list[fbarray]]:
        subatom_indices: list[int] = []
        subatoms: list[Pattern] = []
        subatom_extents: list[fbarray] = []
        patterns_per_subatom: list[fbarray] = []

        for atom_i, (atom, aextent) in enumerate(atomic_patterns_.items()):
            superpattern_flags = [basubset(extent, aextent) and atom <= pattern for pattern, extent in patterns_]
            if not any(superpattern_flags):
                continue

            subatom_indices.append(atom_i)
            subatoms.append(atom)
            subatom_extents.append(aextent)
            patterns_per_subatom.append(fbarray(superpattern_flags))

        if subatoms_order_:
            subatoms_order_ = [bitarray([subatoms_order_[i][j] for j in subatom_indices]) for i in subatom_indices]
        else:
            subatoms_order_ = inverse_order(bfuncs.order_patterns_via_extents(list(zip(subatoms, subatom_extents))))
        subatoms_order_ = list(map(fbarray, subatoms_order_))
        assert all(not subs_ba[i:].any() for i, subs_ba in enumerate(subatoms_order_)), \
            ("Patterns in the OrderedDict of `atomic_patterns` should be ordered by increasing precision: "
             "that is more precise atomic patterns should follow less atomic patterns. "
             "In other words, greater atomic patterns have to be provided after the lesser atomic patterns.")

        return subatoms, subatom_extents, patterns_per_subatom, subatoms_order_


    subatoms, subatom_extents, patterns_per_subatom, subatoms_order = extract_subatoms_data(
        patterns, atomic_patterns, subatoms_order)
    global_extent = subatom_extents[0] | ~subatom_extents[0]

    # now subatoms, subatom_extents, and subatoms_order are fully established
    antichain_iterator = iterate_antichains(subatoms_order, max_length=max_length)
    refine_antichain = None
    found_closures: list[list[fbarray]] = [[] for _ in patterns]
    pbar = tqdm(total=None, disable=not use_tqdm, desc="Iterate key candidates", unit_scale=True)
    while True:
        try:
            antichain = antichain_iterator.send(refine_antichain)
        except StopIteration:
            break
        pbar.update(1)

        superpatterns = reduce(fbarray.__and__, (patterns_per_subatom[i] for i in antichain), ~bazeros(len(patterns)))
        if not superpatterns.any():
            refine_antichain = False
            continue

        ac_extent = reduce(fbarray.__and__, (subatom_extents[i] for i in antichain), global_extent)
        same_extent_patterns = [i for i in superpatterns.search(True) if patterns[i][1] == ac_extent]
        if not same_extent_patterns:
            refine_antichain = True
            continue

        ac_closure = reduce(bitarray.__or__, (subatoms_order[i] for i in antichain), bitarray(len(subatoms_order)))
        for i in antichain: ac_closure[i] = True

        for pattern_i in same_extent_patterns:
            dominates_found_key = any(basubset(found_closure, ac_closure) for found_closure in found_closures[pattern_i])
            if dominates_found_key:
                continue

            pattern = patterns[pattern_i][0]
            is_subpattern = all(subatoms[i] <= pattern for i in antichain)
            if not is_subpattern:
                continue

            found_closures[pattern_i].append(ac_closure)
            key = reduce(pattern.__class__.__or__, (subatoms[i] for i in antichain), pattern.min_pattern)
            yield key, pattern_i

        refine_antichain = superpatterns.count() > len(same_extent_patterns)
    pbar.close()


def iter_intents_via_cboi(
        atomic_patterns: OrderedDict[Pattern, fbarray],
        superatoms_order: list[fbarray],
        min_support: int = 0,
        yield_pattern_intents: bool = True
) -> Iterator[tuple[Union[Pattern, fbarray], fbarray]]:
    """
    Iterate pattern concepts using algorithm Close-by-One-with-Implications

    The original CbOI algorithm was described in the language of attribute implications in (Belfodil et al., 2019).
    This implementation describes essentially the same algorithm, but uses the language of partial order on attributes.

    Parameters
    ----------
    atomic_patterns: OrderedDict[Pattern, frozenbitarray]
        Mapping from atomic patterns to what objects they describe.
        The latter is represented with its characteristic vector stored as a frozenbitarray.
        So `atomic_patterns[p][i] == True` means that atomic pattern `p` describes `i`-th object.
    superatoms_order: list[frozenbitarray]
        Partial order on atomic patterns.
        For every i-th atomic pattern, it shows the indices of all greater atomic patterns.
        The partial order should be topologically sorted:
        for every i-th atomic pattern, all greater patterns should have greater indices.
    min_support: int, default = 0
        Minimal number of objects that a concept should describe.
    yield_pattern_intents: bool, default = True
        Flag whether to yield concept's intent as Pattern or as a frozenbitarray,
        whose True elements corresponds to atomic patterns of the Pattern.

    Yields
    ------
    extent: fbarray
        Concept's extent, i.e. the maximal subset of objects described by concept's intent.
    intent: Pattern or fbarray
        Concept's intent.
        If `yield_pattern_intents == True` then represent intent as the actual Pattern.
        If `yield_pattern_intents == False` then represent intent with frozenbitarray
        describing indices of all atomic patterns that are less precise than the pattern.
        (Then the actual pattern can be obtained as a Pattern.join of all listed atomic patterns).

    References
    ----------
    Belfodil, A., Belfodil, A., & Kaytoue, M. (2019, May).
    Mining Formal Concepts using Implications between Items.
    In International Conference on Formal Concept Analysis (pp. 173-190). Cham: Springer International Publishing.

    """
    assert all(not superatoms[:i].any() for i, superatoms in enumerate(superatoms_order)), \
        ("The value in `superatoms_order` should be topologically sorted. "
         "That is, for every i-th element, all greater elements should have greater indices.")

    #############################
    # Initialise all the values #
    #############################
    def filter_next_atoms(superatoms_order_: list[fbarray]) -> Iterator[fbarray]:
        for superatoms in superatoms_order_:
            nexts = bitarray(superatoms)
            i = 0
            while nexts[i:].any():
                i = nexts.find(True, i)
                nexts &= ~superatoms_order_[i]
                i += 1
            yield nexts

    atomic_patterns, atomic_extents = zip(*atomic_patterns.items())
    subatoms_order = inverse_order(superatoms_order)
    pattern_intent = partial(bfuncs.patternise_description,
                             atomic_patterns=atomic_patterns, subatoms_order=subatoms_order, trusted_input=True)

    min_atoms = bitarray([not subs.any() for subs in subatoms_order])
    nextatoms = list(filter_next_atoms(superatoms_order))
    top_intent = bitarray([aextent.all() for aextent in atomic_extents])
    top_next_candidates = reduce(fbarray.__or__, (nextatoms[i] for i in top_intent.search(True)), min_atoms)
    n_objects, n_atoms = len(atomic_extents[0]), len(atomic_extents)
    top_extent, top_banned = ~bazeros(n_objects), bazeros(n_atoms)

    #################
    # The main loop #
    #################
    stack = deque([(top_extent, top_intent, top_banned, top_next_candidates)])
    while stack:
        ext, descr, banned, all_next_candidates = stack.pop()
        yield pattern_intent(descr) if yield_pattern_intents else descr, fbarray(ext)

        next_candidates = [i for i in (all_next_candidates & ~descr).search(True, right=True)
                           if basubset(subatoms_order[i], descr)]  # only add i-th atom when having all its subatoms
        for m in next_candidates:
            ext_new = ext & atomic_extents[m]
            if ext_new.count() < min_support:
                continue

            intent, next_next_candidates = bitarray(descr), bitarray(all_next_candidates)
            for i in descr.search(False):
                if not basubset(ext_new, atomic_extents[i]):  # i-th atom is not in the intent
                    continue
                if banned[i]:
                    break
                intent[i] = True
                next_next_candidates |= nextatoms[i]
            else:  # no break, i.e. no banned atom found
                stack.append((ext_new, intent, bitarray(banned), next_next_candidates))

            banned[m] = True


def iter_keys_via_talky_gi(
        atomic_patterns: OrderedDict[Pattern, fbarray],
        superatoms_order: list[fbarray],
        min_support: int = 0,
        yield_pattern_keys: bool = True,
        max_key_length: int = None,
        test_subsets: bool = True
) -> Iterator[tuple[Union[Pattern, fbarray], fbarray]]:
    assert all(not superatoms[:i].any() for i, superatoms in enumerate(superatoms_order)), \
        ("The value in `superatoms_order` should be topologically sorted. "
         "That is, all greater elements should have greater indices.")

    max_key_length = max_key_length if max_key_length is not None else len(atomic_patterns)

    subatoms_order = inverse_order(superatoms_order)
    atomic_patterns, atomic_extents = zip(*atomic_patterns.items())
    pattern_key = partial(bfuncs.patternise_description,
                          atomic_patterns=atomic_patterns, subatoms_order=subatoms_order, trusted_input=True)

    n_objects, n_attrs = len(atomic_extents[0]), len(atomic_extents)

    singletons = [bazeros(n_attrs) for _ in range(n_attrs)]
    for i in range(n_attrs): singletons[i][i] = True

    total_extent = atomic_extents[0] | ~atomic_extents[0]

    min_gens_per_extent: dict[fbarray, list[fbarray]] = dict()  # extent => atomic antichains
    min_gen_supports: dict[fbarray, int] = dict()  # atomic antichain => support
    stack = deque([(fbarray(bazeros(n_attrs)), total_extent)])
    while stack:
        descr, extent = stack.pop()
        support = extent.count()
        atomic_closure = reduce(fbarray.__or__, (subatoms_order[i] for i in descr.search(True)), descr)

        if extent not in min_gens_per_extent:
            min_gens_per_extent[extent] = list()

        equiv_subpatterns_found = any(basubset(found_mg, atomic_closure) for found_mg in min_gens_per_extent[extent])
        if equiv_subpatterns_found:
            continue

        yield pattern_key(descr) if yield_pattern_keys else descr, fbarray(extent)
        min_gens_per_extent[extent].append(descr)
        min_gen_supports[descr] = support

        if descr.count() == max_key_length:
            continue

        next_atoms = atomic_closure.search(False, 0, descr.find(True) if descr.any() else len(descr), right=True)
        for next_atom in next_atoms:
            next_support = count_and(extent, atomic_extents[next_atom])
            if next_support < min_support:
                continue

            next_antichain = descr | singletons[next_atom]
            if test_subsets:
                for old_atom in descr.search(True):
                    old_descr = next_antichain & ~singletons[old_atom]
                    if old_descr not in min_gen_supports or min_gen_supports[old_descr] == next_support:
                        all_subsets_are_not_equiv_mingens = False
                        break
                else:  # no break, that is all subsets of `new_antichain` are minimal generators
                    all_subsets_are_not_equiv_mingens = True

                if not all_subsets_are_not_equiv_mingens:
                    continue

            next_extent = extent & atomic_extents[next_atom]
            stack.append((next_antichain, next_extent))
