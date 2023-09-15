from collections import deque
from functools import reduce
from math import ceil
from typing import Iterator, TypeVar
from bitarray import frozenbitarray as fbarray, bitarray
from bitarray.util import zeros as bazeros
from .abstract_ps import AbstractPS

from itertools import combinations

T = TypeVar('T')


class SuperSetPS(AbstractPS):
    """A PS where every description is a set of values. And the bigger is the set, the less precise is the description

    E.g. description {'green', 'yellow', 'red'} is less precise than {'green', 'yellow'}
    as the former describes all the objects that are 'green' OR 'yellow' OR 'red'
    and the latter only describes the objects that are 'green' OR 'yellow'.

    Such Pattern Structure can be applied for categorical values in tabular data.
    """
    PatternType = frozenset[T]
    max_pattern = frozenset()  # Bottom pattern, more specific than any other one

    def join_patterns(self, a: PatternType, b: PatternType) -> PatternType:
        """Return the most precise common pattern, describing both patterns `a` and `b`"""
        if a == self.max_pattern:
            return b
        if b == self.max_pattern:
            return a
        return a | b

    def is_less_precise(self, a: PatternType, b: PatternType) -> bool:
        """Return True if pattern `a` is less precise than pattern `b`"""
        if b == self.max_pattern:
            return True
        if a == self.max_pattern:  # and b != max_pattern
            return False
        return a & b == b

    def iter_bin_attributes(self, data: list[PatternType], min_support: int | float= 0) -> Iterator[tuple[PatternType, fbarray]]:
        """Iterate binary attributes obtained from `data` (from the most general to the most precise ones)

        :parameter
            data: list[PatternType]
             list of object descriptions
            min_support: int
             minimal amount of objects an attribute should describe (in natural numbers, not per cents)
        :return
            iterator of (description: PatternType, extent of the description: frozenbitarray)
        """
        min_support = ceil(len(data) * min_support) if 0 < min_support < 1 else int(min_support)

        unique_values = set()
        for data_row in data:
            unique_values |= data_row
        unique_values = sorted(unique_values)

        for comb_size in range(len(unique_values), -1, -1):
            combs = combinations(unique_values, comb_size)
            for combination in combs:
                pattern = frozenset(combination)
                extent = fbarray((data_row & pattern == data_row for data_row in data))
                if extent.count() < min_support:  # TODO: Optimize min_support check
                    continue
                yield pattern, extent

    def n_bin_attributes(self, data: list[PatternType], min_support: int | float = 0, use_tqdm: bool = False) -> int:
        """Count the number of attributes in the binary representation of `data`"""
        if min_support == 0:
            unique_values = set()
            for data_row in data:
                unique_values |= data_row
            return 2**len(unique_values)
        return super().n_bin_attributes(data, min_support)


class SubSetPS(AbstractPS):
    """A PS where every description is a set of values. And the smaller is the set, the less precise is the description

    E.g. description {'green', 'cubic'} is less precise than {'green', 'cubic', 'heavy'}
    as the former describes all the objects that are 'green' AND 'cubic'
    and the latter describes the objects that are 'green' AND 'cubic' AND 'heavy'.

    """
    PatternType = frozenset[T]
    max_pattern = frozenset({'<ALL_VALUES>'})  # Maximal pattern that should be more precise than any other pattern

    def join_patterns(self, a: PatternType, b: PatternType) -> PatternType:
        """Return the most precise common pattern, describing both patterns `a` and `b`"""
        if b == self.max_pattern:
            return a
        if a == self.max_pattern:
            return b
        return a.intersection(b)

    def is_less_precise(self, a: PatternType, b: PatternType) -> bool:
        """Return True if pattern `a` is less precise than pattern `b`"""
        if b == self.max_pattern:
            return True
        if a == self.max_pattern:  # and b != max_pattern
            return False
        return a.issubset(b)

    def iter_bin_attributes(self, data: list[PatternType], min_support: int | float = 0)\
            -> Iterator[tuple[PatternType, fbarray]]:
        """Iterate binary attributes obtained from `data` (from the most general to the most precise ones)

        :parameter
            data: list[PatternType]
             list of object descriptions
            min_support: int
             minimal amount of objects an attribute should describe (in natural numbers, not per cents)
        :return
            iterator of (description: PatternType, extent of the description: frozenbitarray)
        """
        min_support = ceil(len(data) * min_support) if 0 < min_support < 1 else int(min_support)

        empty_extent = bazeros(len(data))
        vals_extents: dict[T, bitarray] = {}
        for i, pattern in enumerate(data):
            for v in pattern:
                if v not in vals_extents:
                    vals_extents[v] = empty_extent.copy()
                vals_extents[v][i] = True

        total_pattern = {v for v, extent in vals_extents.items() if extent.all()}
        yield frozenset(total_pattern), fbarray(~empty_extent)

        vals_to_pop = [v for v, extent in vals_extents.items() if extent.count() < min_support or extent.all()]
        for v in vals_to_pop:
            del vals_extents[v]

        vals, extents = [list(vals) for vals in zip(*vals_extents.items())]
        n_vals = len(vals)

        queue = deque([({v}, extent, i) for i, (v, extent) in enumerate(zip(vals, extents))])
        while queue:
            words, extent, max_val_id = queue.popleft()
            yield frozenset(words), fbarray(extent)

            for i in range(max_val_id + 1, n_vals):
                next_extent = extent & extents[i]
                if not next_extent.any() or next_extent.count() < min_support:
                    continue

                if extent & extents[i] == extent or extents[i] & extent == extents[i]:
                    continue

                next_words = words | {vals[i]}
                queue.append((next_words, next_extent, i))

        if min_support == 0:
            bottom_extent = reduce(lambda a, b: a & b, vals_extents.values(), ~empty_extent)
            if not bottom_extent.any():
                yield frozenset(vals_extents), fbarray(bottom_extent)
