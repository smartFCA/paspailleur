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
    quality_func = None
    if quality_measure == 'Accuracy':
        def quality_func(tp, fp):
            return tp / n_objects - fp / n_objects + (1 - n_positives / n_objects)
        tp_min = n_objects * (quality_threshold-1) + n_positives
        fp_max = n_objects * (1 - quality_threshold)

    if quality_measure == 'Precision':
        def quality_func(tp, fp):
            return tp / (tp + fp)
        tp_min = 1
        fp_max = n_positives / quality_threshold - n_positives

    if quality_measure == 'Recall':
        def quality_func(tp, fp):
            return tp / n_positives
        tp_min = n_positives * quality_threshold
        fp_max = n_objects - n_positives

    if quality_measure == 'Jaccard':
        def quality_func(tp, fp):
            return 2 * tp / (fp + n_positives)
        tp_min = quality_threshold * n_positives
        fp_max = n_positives / quality_threshold - n_positives

    if quality_measure == 'F1':
        def quality_func(tp, fp):
            return 2 * tp / (tp + fp + n_positives)
        tp_min = quality_threshold * n_positives / (2 - quality_threshold)
        fp_max = 2 * n_positives / quality_threshold - 2 * n_positives

    if quality_measure == 'WRAcc':
        def quality_func(tp, fp):
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
    """Find less precise patterns within pattern structure that describe goal objects with high enough quality

    IMPORTANT: The algorithm does not replicate any existing Subgroup Discovery algorithm (at least, intentionally).
    It should make the job done due to its greediness, but it might be well behind the State-of-the-Art algorithms.
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
