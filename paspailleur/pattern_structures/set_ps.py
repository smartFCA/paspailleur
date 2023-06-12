from typing import Iterator, Optional
from bitarray import frozenbitarray as fbarray
from .abstract_ps import AbstractPS

from itertools import combinations


class SetPS(AbstractPS):
    PatternType = frozenset

    def intersect_patterns(self, a: PatternType, b: PatternType) -> PatternType:
        return a | b

    def bin_attributes(self, data: list[PatternType]) -> Iterator[tuple[PatternType, fbarray]]:
        unique_values = set()
        for data_row in data:
            unique_values |= data_row
        unique_values = sorted(unique_values)

        for comb_size in range(len(unique_values), -1, -1):
            combs = combinations(unique_values, comb_size)
            for combination in combs:
                pattern = frozenset(combination)
                yield pattern, fbarray((data_row & pattern == data_row for data_row in data))

    def is_subpattern(self, a: PatternType, b: PatternType) -> bool:
        return a & b == b

    def n_bin_attributes(self, data: list[PatternType]) -> int:
        unique_values = set()
        for data_row in data:
            unique_values |= data_row
        return 2**len(unique_values)
