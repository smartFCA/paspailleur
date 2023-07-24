from typing import Iterator, Optional
from bitarray import frozenbitarray as fbarray
from .abstract_ps import AbstractPS

from itertools import combinations


class SetPS(AbstractPS):
    PatternType = frozenset
    bottom = frozenset()  # Bottom pattern, more specific than any other one

    def join_patterns(self, a: PatternType, b: PatternType) -> PatternType:
        """Return the most precise common pattern, describing both patterns `a` and `b`"""
        return a | b

    def iter_bin_attributes(self, data: list[PatternType]) -> Iterator[tuple[PatternType, fbarray]]:
        """Iterate binary attributes obtained from `data` (from the most general to the most precise ones)

        :parameter
            data: list[PatternType]
             list of object descriptions
        :return
            iterator of (description: PatternType, extent of the description: frozenbitarray)
        """
        unique_values = set()
        for data_row in data:
            unique_values |= data_row
        unique_values = sorted(unique_values)

        for comb_size in range(len(unique_values), -1, -1):
            combs = combinations(unique_values, comb_size)
            for combination in combs:
                pattern = frozenset(combination)
                yield pattern, fbarray((data_row & pattern == data_row for data_row in data))

    def is_less_precise(self, a: PatternType, b: PatternType) -> bool:
        """Return True if pattern `a` is less precise than pattern `b`"""
        return a & b == b

    def n_bin_attributes(self, data: list[PatternType]) -> int:
        """Count the number of attributes in the binary representation of `data`"""
        unique_values = set()
        for data_row in data:
            unique_values |= data_row
        return 2**len(unique_values)
