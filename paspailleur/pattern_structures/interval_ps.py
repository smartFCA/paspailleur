from typing import Iterator, Optional
from bitarray import frozenbitarray as fbarray
from .abstract_ps import AbstractPS


class IntervalPS(AbstractPS):
    PatternType = Optional[tuple[float, float]]

    def intersect_patterns(self, a: PatternType, b: PatternType) -> PatternType:
        if a is None or b is None:
            return None

        return min(a[0], b[0]), max(a[1], b[1])

    def bin_attributes(self, data: list[PatternType]) -> Iterator[tuple[PatternType, fbarray]]:
        lower_bounds, upper_bounds = [sorted(set(bounds)) for bounds in zip(*data)]
        min_, max_ = lower_bounds[0], upper_bounds[-1]
        lower_bounds.pop(0)
        upper_bounds.pop(-1)

        yield (min_, max_), fbarray([True]*len(data))

        for lb in lower_bounds:
            yield (lb, max_), fbarray((lb <= x for x, _ in data))

        for ub in upper_bounds[::-1]:
            yield (min_, ub), fbarray((x <= ub for _, x in data))

        yield None, fbarray([False]*len(data))

    def is_subpattern(self, a: PatternType, b: PatternType) -> bool:
        if b is None:
            return True

        if a is None:
            return False

        return a[0] <= b[0] <= b[1] <= a[1]

    def n_bin_attributes(self, data: list[PatternType]) -> int:
        return len({lb for lb, ub in data}) + len({ub for ub in data})
