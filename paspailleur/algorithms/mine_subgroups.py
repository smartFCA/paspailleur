import math

from bitarray import bitarray, frozenbitarray as fbarray
from bitarray.util import count_and, subset as basubset
from typing import Literal, Iterator, Callable

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
