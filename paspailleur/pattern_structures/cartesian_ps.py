from typing import Iterator
from bitarray import frozenbitarray as fbarray
from .abstract_ps import AbstractPS


class CartesianPS(AbstractPS):
    PatternType = list[list]
    basic_structures: list[AbstractPS]

    def __init__(self, basic_structures: list[AbstractPS]):
        self.basic_structures = basic_structures

    def intersect_patterns(self, a: PatternType, b: PatternType) -> PatternType:
        return [ps.intersect_patterns(a_, b_) for (ps, a_, b_) in zip(self.basic_structures, a, b)]

    def bin_attributes(self, data: list[PatternType]) -> Iterator[tuple[PatternType, fbarray]]:
        for i, ps in enumerate(self.basic_structures):
            ps_data = [data_row[i] for data_row in data]
            for pattern, flag in ps.bin_attributes(ps_data):
                yield (i, pattern), flag

    def is_subpattern(self, a: PatternType, b: PatternType) -> bool:
        return all(ps.is_subpattern(a_, b_) for ps, a_, b_ in zip(self.basic_structures, a, b))

    def n_bin_attributes(self, data: list[PatternType]) -> int:
        n_bin_attrs = 0
        for i, ps in enumerate(self.basic_structures):
            ps_data = [data_row[i] for data_row in data]
            n_bin_attrs += ps.n_bin_attributes(ps_data)
        return n_bin_attrs
