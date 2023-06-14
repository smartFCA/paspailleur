from dataclasses import dataclass
from typing import TypeVar, Iterator
from bitarray import frozenbitarray as fbarray
from bitarray.util import zeros as bazeros


@dataclass
class ProjectionNotFoundError(ValueError):
    projection_number: int

    def __str__(self):
        return f"Projection #{self.projection_number} could not be computed"


class AbstractPS:
    PatternType = TypeVar('PatternType')

    def intersect_patterns(self, a: PatternType, b: PatternType) -> PatternType:
        raise NotImplementedError

    def bin_attributes(self, data: list[PatternType]) -> Iterator[tuple[PatternType, fbarray]]:
        raise NotImplementedError

    def is_subpattern(self, a: PatternType, b: PatternType) -> bool:
        return self.intersect_patterns(a, b) == a

    def extent(self, pattern: PatternType, data: list[PatternType]) -> Iterator[int]:
        return (i for i, obj_description in enumerate(data) if self.is_subpattern(pattern, obj_description))

    def intent(self, data: list[PatternType]) -> PatternType:
        intent = None
        for obj_description in data:
            if intent is None:
                intent = obj_description
                continue

            intent = self.intersect_patterns(intent, obj_description)
        return intent

    def n_bin_attributes(self, data: list[PatternType]) -> int:
        return sum(1 for _ in self.bin_attributes(data))

    def binarize(self, data: list[PatternType]) -> tuple[list[PatternType], list[fbarray]]:
        patterns, flags = list(zip(*list(self.bin_attributes(data))))

        n_rows, n_cols = len(flags[0]), len(flags)
        itemsets_ba = [bazeros(n_cols) for _ in range(n_rows)]
        for j, flag in enumerate(flags):
            for i in flag.itersearch(True):
                itemsets_ba[i][j] = True
        itemsets_ba = [fbarray(ba) for ba in itemsets_ba]
        return list(patterns), itemsets_ba
