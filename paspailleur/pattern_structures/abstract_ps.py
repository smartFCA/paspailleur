from dataclasses import dataclass
from typing import TypeVar, Iterator, Iterable
from bitarray import frozenbitarray as fbarray
from bitarray.util import zeros as bazeros


@dataclass
class ProjectionNotFoundError(ValueError):
    projection_number: int

    def __str__(self):
        return f"Projection #{self.projection_number} could not be computed"


class AbstractPS:
    PatternType = TypeVar('PatternType')
    bottom: PatternType  # Bottom pattern, more specific than any other one

    def join_patterns(self, a: PatternType, b: PatternType) -> PatternType:
        """Return the most precise common pattern, describing both patterns `a` and `b`"""
        raise NotImplementedError

    def is_less_precise(self, a: PatternType, b: PatternType) -> bool:
        """Return True if pattern `a` is less precise than pattern `b`"""
        return self.join_patterns(a, b) == a

    def extent(self, pattern: PatternType, data: list[PatternType]) -> Iterator[int]:
        """Return indices of rows in `data` whose description contains `pattern`"""
        return (i for i, obj_description in enumerate(data) if self.is_less_precise(pattern, obj_description))

    def intent(self, data: list[PatternType], indices: Iterable[int] = None) -> PatternType:
        """Return common pattern of all rows in `data`"""
        iterator = (data[i] for i in indices) if indices is not None else data

        intent = None
        for obj_description in iterator:
            if intent is None:
                intent = obj_description
                continue

            intent = self.join_patterns(intent, obj_description)
        return intent

    def iter_bin_attributes(self, data: list[PatternType], min_support: int = 0) -> Iterator[tuple[PatternType, fbarray]]:
        """Iterate binary attributes obtained from `data` (from the most general to the most precise ones)

        :parameter
            data: list[PatternType]
             list of object descriptions
            min_support: int
             minimal amount of objects an attribute should describe (in natural numbers, not per cents)
        :return
            iterator of (description: PatternType, extent of the description: frozenbitarray)
        """
        raise NotImplementedError

    def n_bin_attributes(self, data: list[PatternType], min_support: int = 0) -> int:
        """Count the number of attributes in the binary representation of `data`"""
        return sum(1 for _ in self.iter_bin_attributes(data, min_support))

    def binarize(self, data: list[PatternType], min_support: int = 0) -> tuple[list[PatternType], list[fbarray]]:
        """Binarize the data into Formal Context

        :parameter
            data: list[PatternType]
                List of row descriptions
            min_support: int
                minimal amount of objects an attribute should describe (in natural numbers, not per cents)
        :return
            patterns: list[PatternType]
                Patterns corresponding to the attributes in the binarised data (aka binary attribute names)
            itemsets_ba: list[frozenbitarray]
                List of itemsets for every row in `data`.
                `itemsets_ba[i][j]` shows whether `data[i]` contains`patterns[j]`
        """
        patterns, flags = list(zip(*list(self.iter_bin_attributes(data, min_support))))

        n_rows, n_cols = len(flags[0]), len(flags)
        itemsets_ba = [bazeros(n_cols) for _ in range(n_rows)]
        for j, flag in enumerate(flags):
            for i in flag.itersearch(True):
                itemsets_ba[i][j] = True
        itemsets_ba = [fbarray(ba) for ba in itemsets_ba]
        return list(patterns), itemsets_ba
