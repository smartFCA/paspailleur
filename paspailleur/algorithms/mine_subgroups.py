import math
from functools import reduce

from bitarray import bitarray, frozenbitarray as fbarray
from bitarray.util import count_and, subset as basubset
from typing import Literal, Iterator, Callable, OrderedDict, Optional
from tqdm.auto import tqdm

from caspailleur.order import inverse_order
from paspailleur.algorithms import base_functions as bfuncs
from paspailleur.pattern_structures.pattern import Pattern


#from paspailleur.pattern_structures.pattern_structure import PatternStructure
#from paspailleur.pattern_structures.pattern import Pattern


def setup_quality_measure_function(
        quality_measure: Literal['Accuracy', 'Precision', 'Recall', 'Jaccard', 'F1', 'WRAcc'],
        quality_threshold: float,
        n_positives: int, n_objects: int,
    ) -> tuple[Callable[[int, int], float], int, int]:
    """
    Set up a quality measure function based on a specified metric and threshold.

    Parameters
    ----------
    quality_measure: Literal
        The metric to use for subgroup quality evaluation.
    quality_threshold: float
        The minimum acceptable value for the quality function.
    n_positives: int
        Number of positive (goal) objects.
    n_objects: int
        Total number of objects.

    Returns
    -------
    quality_setup: tuple
        A tuple of (quality function, minimum true positives, maximum false positives).
    """
    quality_func = None
    if quality_measure == 'Accuracy':
        def quality_func(tp, fp):
            """
            Compute Accuracy: (TP - FP)/N + const.

            Parameters
            ----------
            tp: int
                True positives.
            fp: int
                False positives.

            Returns
            -------
            score: float
                Accuracy score.
            """
            return tp / n_objects - fp / n_objects + (1 - n_positives / n_objects)
        tp_min = n_objects * (quality_threshold-1) + n_positives
        fp_max = n_objects * (1 - quality_threshold)

    if quality_measure == 'Precision':
        def quality_func(tp, fp):
            """
            Compute Precision: TP / (TP + FP)

            Parameters
            ----------
            tp: int
                True positives.
            fp: int
                False positives.

            Returns
            -------
            score: float
                Precision score.
            """
            return tp / (tp + fp)
        tp_min = 1
        fp_max = n_positives / quality_threshold - n_positives

    if quality_measure == 'Recall':
        def quality_func(tp, fp):
            """
            Compute Recall: TP / P

            Parameters
            ----------
            tp: int
                True positives.
            fp: int
                False positives (unused).

            Returns
            -------
            score: float
                Recall score.
            """
            return tp / n_positives
        tp_min = n_positives * quality_threshold
        fp_max = n_objects - n_positives

    if quality_measure == 'Jaccard':
        def quality_func(tp, fp):
            """
            Compute Jaccard index: 2 * TP / (FP + P)

            Parameters
            ----------
            tp: int
                True positives.
            fp: int
                False positives.

            Returns
            -------
            score: float
                Jaccard score.
            """
            return 2 * tp / (fp + n_positives)
        tp_min = quality_threshold * n_positives
        fp_max = n_positives / quality_threshold - n_positives

    if quality_measure == 'F1':
        def quality_func(tp, fp):
            """
            Compute F1-score: 2 * TP / (TP + FP + P)

            Parameters
            ----------
            tp: int
                True positives.
            fp: int
                False positives.

            Returns
            -------
            score: float
                F1 score.
            """
            return 2 * tp / (tp + fp + n_positives)
        tp_min = quality_threshold * n_positives / (2 - quality_threshold)
        fp_max = 2 * n_positives / quality_threshold - 2 * n_positives

    if quality_measure == 'WRAcc':
        def quality_func(tp, fp):
            """
            Compute WRAcc: TP/N - (TP+FP)*P/N^2

            Parameters
            ----------
            tp: int
                True positives.
            fp: int
                False positives.

            Returns
            -------
            score: float
                Weighted Relative Accuracy.
            """
            return tp / n_objects - (tp + fp) * n_positives / n_objects / n_objects
        tp_min = quality_threshold * n_objects * n_objects / (n_objects - n_positives)
        fp_max = n_objects - n_positives - (n_objects*n_objects*quality_threshold)/n_positives

    if quality_func is None:
        raise ValueError(f'Provided quality measure {quality_measure=} is not supported. '
                         "The supported measures are {'Accuracy', 'Precision', 'Recall', 'Jaccard', 'F1', 'WRAcc'}")

    return quality_func, math.ceil(tp_min), math.floor(fp_max)


def iter_subgroups_bruteforce(
        pattern_structure,#: PatternStructure,
        goal_objects: bitarray,
        quality_threshold: float,
        quality_func: Callable[[int, int], float],
        tp_min: int = None,
        fp_max: int = None,
        max_pattern_length: int = None
    ):# -> Iterator[tuple[Pattern, bitarray, float]]:
    """
    Find less precise patterns that describe goal objects with sufficient quality via brute-force.

    IMPORTANT
    ---------
    The algorithm does not replicate any existing Subgroup Discovery algorithm (at least, intentionally).
    It should make the job done due to its greediness, but it might be well behind the State-of-the-Art algorithms.
    
    Parameters
    ----------
    pattern_structure: PatternStructure
        The pattern structure to mine from.
    goal_objects: bitarray
        Bitarray indicating which objects are goal/positive.
    quality_threshold: float
        Minimum acceptable value for the quality function.
    quality_func: Callable[[int, int], float]
        A function that computes the quality based on true and false positives.
    tp_min: int, optional
        Minimum number of true positives.
    fp_max: int, optional
        Maximum number of false positives.
    max_pattern_length: int, optional
        Maximum allowed length of a pattern.

    Returns
    -------
    subgroups: Iterator[tuple[Pattern, bitarray, float]]
        Yields qualifying patterns, their extents, and quality scores.
    """
    patterns_iterator = pattern_structure.iter_patterns(
        kind='ascending controlled', min_support=tp_min, depth_first=False, return_objects_as_bitarrays=True)
    next(patterns_iterator)  # initialise iterator

    found_patterns = [] #: list[tuple[Pattern, bitarray]] = []

    go_deeper = True
    while True:
        try:
            pattern, extent = patterns_iterator.send(go_deeper)
        except StopIteration:
            break

        if max_pattern_length is None:
            test_greater_patterns = True
        else:
            pattern_length = len(pattern)
            if pattern_length > max_pattern_length:
                go_deeper = False
                continue
            test_greater_patterns = pattern_length < max_pattern_length

        tp = count_and(extent, goal_objects)
        if tp_min is not None and tp < tp_min:
            go_deeper = False
            continue

        fp = extent.count() - tp
        if fp_max is not None and fp > fp_max:
            go_deeper = test_greater_patterns
            continue

        quality = quality_func(tp, fp)
        if quality < quality_threshold:
            go_deeper = test_greater_patterns
            continue

        found_less_precise = any(
            found_pattern <= pattern for found_pattern, found_extent in found_patterns
            if basubset(extent, found_extent)  # if not(extent<=found_extent) then not(found_pattern<=pattern)
        )
        if found_less_precise:
            go_deeper = False
            continue

        # quality >= quality_threshold
        yield pattern, extent, quality
        found_patterns.append((pattern, extent))
        go_deeper = False


def iter_subgroups_via_atoms(
        atomic_patterns: OrderedDict[Pattern, bitarray],
        goal_objects: bitarray,
        quality_threshold: float,
        quality_func: Callable[[int, int], float],
        tp_min: int = None,
        max_subgroup_length: int = None,
        subatoms_order: list[bitarray] = None,
        use_tqdm: bool = False,
) -> Iterator[tuple[Pattern, fbarray, float]]:
    """
    Mine patterns that describe given `goal_objects` good-enough w.r.t. the `quality_func` and `quality_threshold`.

    The mined patterns are the least precise patterns whose `quality_func` value is higher than the `quality_threshold`.

    Such patterns can also be called "subgroups" when related to Subgroup Discovery field.
    The algorithm implemented in this function is rather "bruteforce" and
    only uses a smart atomic patterns "antichain traversal" as optimisation.
    Therefore, it can be much slower than the State-of-the-Art algorithms of Subgroup Discovery.


    Parameters
    ----------
    atomic_patterns: OrderedDict[Pattern, bitarray]
        Ordered Dictionary of atomic patterns and their extents (represented with bitarrays).
        Every yielded pattern is a join of a subset of atomic patterns.
        The dictionary should be Ordered in order to reflect the specificity order of the atomic patterns.
        That is, the less precise atomic patterns should be placed in the "beginning" of the dictionary,
        the more precise patterns should be placed in the "end" of the dictionary,
        and every atomic pattern should be placed after all its smaller atomic patterns.
    goal_objects: bitarray
        A subset of objects to find a pattern to. Should be represented with a bitarray where i-th element equals
        to True when i-th object belongs to the set of "goal" objects.
    quality_threshold: float
        The minimal bound when a pattern can be considered good enough and be yielded by the function.
        If a pattern is considered good enough, none of its more precise patterns will be tested for their quality.
    quality_func: Callable[[int, int], float]
        A function to evaluate the quality of a pattern.
        The function should follow a specific interface: it takes the number of true-positive and false-positive objects
        described by a pattern, and outputs the score value. The greater the score is, the better fitted is the pattern.
        Examples of such quality functions can be generated using function
        `paspailleur.algorithms.mine_subgroups.setup_quality_measure_function`.
    tp_min: int, optional
        Minimal number of true positives that a pattern should describe.
        When provided, this value helps to leave out patterns with too small extents.
        When not provided, it is considered to be 0.
    max_subgroup_length: int, default = len(atomic_patterns)
        The maximal number of atomic patterns that can be joined together to form a pattern.
        When provided, this value helps to leave out patterns that consist of too many atomic patterns,
        so the patterns that are deemed to be "too complex".
    subatoms_order: list[bitarray], optional
        Subatoms order on atomic patterns from `atomic_patterns` represented with list of bitarrays.
        The value `subatoms_order[i][j]` should equal True when j-th atomic pattern is less precise than
        i-th atomic pattern: `subatoms_order[i][j] == list(atomic_patterns)[j] <= list(atomic_patterns)[i]`.
        When not provided, all values of `subatoms_order` are computed inside the function.
    use_tqdm: bool, optional
        A flag whether to use tqdm progress bar to track the number of patterns that was processed by the function.
        Defaults to False.

    Yields
    ------
    subgroup: Pattern
        A pattern that describes the `goal_objects`.
    extent: frozenbitarray
        A set of objects described by `subgroup` represented with a frozenbitarray.
    score: float
        The value of `quality_func` for `subgroup`.

    """
    def extract_subatoms_data(
            extent: fbarray, atomic_patterns_: OrderedDict[Pattern, fbarray],
            subatoms_order_: Optional[list[fbarray]],
            tp_min_: int,
    ) -> tuple[list[Pattern], list[fbarray], list[fbarray]]:
        atom_indices: list[int] = []
        atoms: list[Pattern] = []
        atom_extents: list[fbarray] = []

        for atom_i, (atom, aextent) in enumerate(atomic_patterns_.items()):
            if count_and(extent, aextent) < tp_min_:
                continue

            atom_indices.append(atom_i)
            atoms.append(atom)
            atom_extents.append(aextent)

        if subatoms_order_:
            subatoms_order_ = [bitarray([subatoms_order_[i][j] for j in atom_indices]) for i in atom_indices]
        else:
            subatoms_order_ = inverse_order(bfuncs.order_patterns_via_extents(list(zip(atoms, atom_extents))))
        subatoms_order_ = list(map(fbarray, subatoms_order_))

        assert all(not subs_ba[i:].any() for i, subs_ba in enumerate(subatoms_order_)), \
            ("Patterns in the OrderedDict of `atomic_patterns` should be ordered by increasing precision: "
             "that is more precise atomic patterns should follow less atomic patterns. "
             "In other words, greater atomic patterns have to be provided after the lesser atomic patterns.")

        return atoms, atom_extents, subatoms_order_

    tp_min: int = 0 if tp_min is None else tp_min
    global_extent = goal_objects | ~goal_objects

    atoms, atom_extents, subatoms_order = extract_subatoms_data(goal_objects, atomic_patterns, subatoms_order, tp_min)

    # now subatoms, subatom_extents, and subatoms_order are fully established
    antichain_iterator = bfuncs.iterate_antichains(subatoms_order, max_length=max_subgroup_length)
    refine_antichain = None
    found_closures: list[fbarray] = []
    pbar = tqdm(total=None, disable=not use_tqdm, desc="Iterate subgroup candidates", unit_scale=True)
    while True:
        try:
            antichain = antichain_iterator.send(refine_antichain)
        except StopIteration:
            break
        pbar.update(1)

        ac_extent = reduce(fbarray.__and__, (atom_extents[i] for i in antichain), global_extent)
        tp = count_and(ac_extent, goal_objects)
        fp = ac_extent.count() - tp

        if tp < tp_min:
            refine_antichain = False
            continue

        ac_quality = quality_func(tp, fp)
        if ac_quality < quality_threshold:
            refine_antichain = True
            continue

        ac_closure = reduce(bitarray.__or__, (subatoms_order[i] for i in antichain), bitarray(len(subatoms_order)))
        for i in antichain: ac_closure[i] = True

        dominates_found_closure = any(basubset(closure, ac_closure) for closure in found_closures)
        if dominates_found_closure:
            refine_antichain = False
            continue

        found_closures.append(ac_closure)
        pattern = reduce(atoms[0].__class__.__or__, (atoms[i] for i in antichain), atoms[0].min_pattern)
        yield pattern, ac_extent, ac_quality
        refine_antichain = False
    pbar.close()
