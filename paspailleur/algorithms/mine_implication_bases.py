from functools import reduce, partial
from typing import Iterable, OrderedDict, Iterator, Union

from bitarray import frozenbitarray as fbarray, bitarray
from bitarray.util import zeros as bazeros, subset as basubset

from paspailleur.pattern_structures.pattern import Pattern
from paspailleur.algorithms import base_functions as bfuncs


def iter_proper_premises_from_atomised_premises(
        premise_extent_iterator: Iterable[tuple[bitarray, bitarray]],
        minsup_atomic_patterns: OrderedDict[Pattern, bitarray],
        minsup_subatoms_order: list[bitarray],
        maxsup_atomic_patterns: OrderedDict[Pattern, bitarray],
        maxsup_subatoms_order: list[bitarray] = None,
        yield_patterns: bool = True,
        reduce_conclusions: bool = False,
) -> Iterator[Union[tuple[Pattern, Pattern], tuple[bitarray, bitarray]]]:
    """
    Iterate proper premises and their conclusion based on the premise candidates represented with indices of their atoms

    Important:
    The sets of `minsup_atomic_patterns`, `maxsup_atomic_patterns`, premises in `premise_extent_iterator`
    have to be topologically sorted. That is, the greater the atomic pattern, the greater index it should have.


    Parameters
    ----------
    premise_extent_iterator: Iterable[tuple[bitarray, bitarray]]
        Pairs of premise candidates and their extents.
        The indices of True elements in premise candidates correspond to atomic patterns in `minsup_atomic_patterns`.
    minsup_atomic_patterns: OrderedDict[Pattern, bitarray]
        Support-minimal atomic patterns and their extents.
        Atomic pattern is support-minimal when all smaller atomic patterns describe more objects.
    minsup_subatoms_order: list[bitarray]
        Partial order on support-minimal atomic patterns.
        Value `minsup_subatoms_order[i][j]` is True when j-th sup.min. atomic pattern is smaller than the i-th one.
        The order should be topologically sorted, that is the greater patterns should have greater indices.
    maxsup_atomic_patterns: OrderedDict[Pattern, bitarray]
        Support-maximal atomic patterns and their extents.
        Value `maxsup_subatoms_order[i][j]` is True when j-th sup.max. atomic pattern is smaller than the i-th one.
        Atomic pattern is support-maximal when all greater atomic patterns describe fewer objects.
    maxsup_subatoms_order: list[bitarray]
        Partial order on support-maximal atomic patterns.
    yield_patterns: bool, default True
        Flag whether to output proper premises and their conclusions as Patterns, or a bitarrays.
    reduce_conclusions: bool, default False
        Flag whether to output the reduced conclusion for each premise (to not repeat the conclusions of other premises)
        or the full conclusion.

    Returns
    -------
    premise: Pattern or bitarray
        Proper premise represented as Pattern (when `yield_patterns` is True)
        or as a bitarray that references `minsup_atomic_patterns`.
    conclusion: Pattern or bitarray
        Conclusion represented as Pattern (when `yield_patterns` is True)
        or as a bitarray that references `maxsup_atomic_patterns`.
        When `reduce_conclusions` is True, output only the part of the conclusion
        that cannot be deduced from other implications.

    """
    if yield_patterns:
        assert maxsup_subatoms_order is not None, \
            "Provide partial order on `maxsup_atomic_patterns` to be able to yield patterns."
    assert all(not subatoms[i:].any() for i, subatoms in enumerate(minsup_subatoms_order)),  \
        ("The dict of `minsup_atomic_patterns` should be topologically sorted. "
         "That is, for every pattern, all its smaller patterns should have smaller indices.")
    assert all(not subatoms[i:].any() for i, subatoms in enumerate(maxsup_subatoms_order)), \
        ("The dict of `maxsup_atomic_patterns` should be topologically sorted. "
         "That is, for every pattern, all its smaller patterns should have smaller indices.")

    minsup_atomic_patterns = list(minsup_atomic_patterns)
    maxsup_atomic_patterns, maxsup_atomic_extents = zip(*maxsup_atomic_patterns.items())
    n_maxsup_atoms = len(maxsup_atomic_patterns)

    pattern_premise = partial(
        bfuncs.patternise_description,
        atomic_patterns=minsup_atomic_patterns, subatoms_order=minsup_subatoms_order, trusted_input=True)
    pattern_conclusion = partial(
        bfuncs.patternise_description,
        atomic_patterns=maxsup_atomic_patterns, subatoms_order=maxsup_subatoms_order, trusted_input=True)


    # pairs of (all minsup atoms smaller than prop. premise, all maxsup atoms implied by prop. premise)
    proper_premises: list[tuple[bitarray, bitarray]] = []
    for candidate, extent in premise_extent_iterator:
        premise_full = reduce(fbarray.__or__, (minsup_subatoms_order[i] for i in candidate.search(True)), candidate)

        subclosures = (closure for premise, closure in proper_premises if basubset(premise, premise_full))
        implied_conclusion = reduce(fbarray.__or__, subclosures, bazeros(n_maxsup_atoms))
        covers_unimplied_atom = any(basubset(extent, maxsup_atomic_extents[atom_i])
                                    for atom_i in implied_conclusion.search(False))
        if not covers_unimplied_atom:
            continue

        conclusion_full = bitarray(implied_conclusion)
        for i in implied_conclusion.search(False): conclusion_full[i] = basubset(extent, maxsup_atomic_extents[i])

        key_atoms = {minsup_atomic_patterns[i] for i in candidate.search(True)}
        nontrivial_added_conclusion = bazeros(n_maxsup_atoms)
        for i in (conclusion_full & ~implied_conclusion).search(True):
            nontrivial_added_conclusion[i] = maxsup_atomic_patterns[i] not in key_atoms
        if not nontrivial_added_conclusion.any():
            continue

        proper_premises.append((premise_full, conclusion_full))
        conclusion_final = nontrivial_added_conclusion if reduce_conclusions else conclusion_full
        if yield_patterns:
            yield pattern_premise(premise_full), pattern_conclusion(conclusion_final)
        else:
            yield premise_full, conclusion_final


def iter_pseudo_intents_from_atomised_premises(
        premises: Iterable[bitarray],
        atomic_patterns: OrderedDict[Pattern, bitarray],
        subatoms_order: list[bitarray],
        yield_patterns: bool = True,
        reduce_conclusions: bool = False,
) -> Iterator[Union[tuple[Pattern, Pattern], tuple[bitarray, bitarray]]]:
    """
    Iterate pseudo intents and their conclusion based on the premise candidates represented with indices of their atoms

    Important:
    The sets of `atomic_patterns`, have to be topologically sorted.
    That is, the greater the atomic pattern, the greater index it should have.


    Parameters
    ----------
    premises: Iterable[bitarray]
        List of premises to convert into pseudo-intents.
        The indices of True elements in premise candidates correspond to atomic patterns in `atomic_patterns`.
    atomic_patterns: OrderedDict[Pattern, bitarray]
        Atomic patterns and their extents. Dictionary should contain both support-minimal and support-maximal patterns..
    subatoms_order: list[bitarray]
        Partial order on atomic patterns.
        Value `subatoms_order[i][j]` is True when j-th atomic pattern is smaller than the i-th one.
        The order should be topologically sorted, that is the greater patterns should have greater indices.
    yield_patterns: bool, default True
        Flag whether to output proper premises and their conclusions as Patterns, or a bitarrays.
    reduce_conclusions: bool, default False
        Flag whether to output the reduced conclusion for each premise (to not repeat the conclusions of other premises)
        or the full conclusion.

    Returns
    -------
    premise: Pattern or bitarray
        Proper premise represented as Pattern (when `yield_patterns` is True)
        or as a bitarray that references `minsup_atomic_patterns`.
    conclusion: Pattern or bitarray
        Conclusion represented as Pattern (when `yield_patterns` is True)
        or as a bitarray that references `maxsup_atomic_patterns`.
        When `reduce_conclusions` is True, output only the part of the conclusion
        that cannot be deduced from other implications.

    """
    assert all(not subatoms[i:].any() for i, subatoms in enumerate(subatoms_order)), \
        ("The dict of `subatoms_order` should be topologically sorted. "
         "That is, for every pattern, all its smaller patterns should have smaller indices.")

    atomic_patterns, atomic_extents = zip(*atomic_patterns.items())
    top_extent = atomic_extents[0] | ~atomic_extents[0]

    paternise = partial(bfuncs.patternise_description, atomic_patterns=atomic_patterns, subatoms_order=subatoms_order)

    conclusions = []
    for premise in premises:
        extent = reduce(fbarray.__and__, (atomic_extents[i] for i in premise.search(True)), top_extent)

        premise_full = reduce(fbarray.__or__, (subatoms_order[i] for i in premise.search(True)), premise)
        conclusion_full = bitarray(premise_full)
        for i in premise_full.search(False): conclusion_full[i] = basubset(extent, atomic_extents[i])
        conclusions.append(conclusion_full)

    for premise, conclusion_full in zip(premises, conclusions):
        premise_saturated = bitarray(premise)
        for other_premise, other_conclusion in zip(premises, conclusions):
            if other_premise == premise:
                continue

            if basubset(other_premise, premise_saturated) and other_premise != premise_saturated:
                premise_saturated = premise_saturated | other_conclusion
            if premise_saturated == conclusion_full:
                break
        if premise_saturated == conclusion_full:
            continue

        conclusion = (conclusion_full & ~premise_saturated) if reduce_conclusions else conclusion_full
        if yield_patterns:
            yield paternise(premise), paternise(conclusion)
        else:
            yield premise, conclusion
