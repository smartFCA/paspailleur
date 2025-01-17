import heapq
from functools import reduce
from itertools import takewhile
from collections import OrderedDict

from tqdm.auto import tqdm
from typing import Iterator, Generator, Collection, Iterable, Optional
from bitarray import bitarray, frozenbitarray as fbarray
from bitarray.util import zeros as bazeros, subset as basubset

from paspailleur.algorithms import base_functions as bfuncs
from paspailleur.pattern_structures import AbstractPS
from paspailleur.pattern_structures.pattern import Pattern


def list_intents_via_Lindig_complex(data: list, pattern_structure: AbstractPS) -> list['PatternDescription']:
    """Get the list of intents of pattern concepts from `data` described by `pattern_structure` running Lindig algorithm
    from "Fast Concept Analysis" by Christian Lindig, Harvard University, Division of Engineering and Applied Sciences

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
        if bits == []:
            return(bitarray([1 for _ in range(len_bitarray)]))
        bit = bits[0]
        for obj in bits[1:]:
            bit = bit & obj
        return(bit)

    def find_upper_neighbors(data: list, concept_extent: list, objects_indices: list):
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
    """Iterate intents in patterns by running object-wise version of Close By One algorithm"""
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
    def maximal_bitarrays(bas: Collection[fbarray]) -> set[fbarray]:
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
    atomic_patterns_iterator = tqdm(atomic_patterns_iterator, total=n_atomic_patterns, disable=not use_tqdm)

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

    return set(stable_extents)


def iter_keys_of_pattern(pattern: Pattern, atomic_patterns: OrderedDict[Pattern, fbarray]) -> Iterator[Pattern]:
    # second condition implies the first one. But the first one is easier to test
    atoms_to_iterate = (atom for atom in atomic_patterns if atom <= pattern)
    atoms_to_iterate = sorted(atoms_to_iterate, key=lambda atom: atomic_patterns[atom].count())
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
    while True:
        try:
            key_candidate, candidate_extent = pattern_iterator.send(go_more_precise)
        except StopIteration:
            break

        go_more_precise = extent != candidate_extent
        if not go_more_precise:
            yield key_candidate


def iter_keys_of_patterns(patterns: list[Pattern], atomic_patterns: OrderedDict[Pattern, fbarray])\
        -> Iterator[tuple[Pattern, int]]:
    """

    Important: atomic_patterns should be sorted in topological order.
    So every i-th atomic pattern should be not-smaller than any previous (1, 2, ..., i-i) atomic pattern
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

        go_more_precise = any(basubset(extent, candidate_extent) and extent != candidate_extent
                              for extent in patterns_per_extents)

        if candidate_extent not in patterns_per_extents:
            continue

        # candidate_extent in atoms_per_extent
        target_patterns = patterns_per_extents[candidate_extent]
        target_patterns = filter(lambda i: key_candidate <= patterns[i], target_patterns)
        target_patterns = filter(lambda i: not any(k <= key_candidate for k in keys_per_pattern[i]), target_patterns)

        for pattern_i in target_patterns:
            yield key_candidate, pattern_i
            keys_per_pattern[pattern_i].append(key_candidate)
