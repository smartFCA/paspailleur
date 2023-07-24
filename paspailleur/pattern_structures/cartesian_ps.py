from typing import Iterator
from bitarray import frozenbitarray as fbarray
from .abstract_ps import AbstractPS


class CartesianPS(AbstractPS):
    PatternType = list[list]
    bottom: list  # Bottom pattern, more specific than any other one
    basic_structures: list[AbstractPS]

    def __init__(self, basic_structures: list[AbstractPS]):
        self.basic_structures = basic_structures
        self.bottom = [ps.bottom for ps in basic_structures]

    def join_patterns(self, a: PatternType, b: PatternType) -> PatternType:
        """Return the most precise common pattern, describing both patterns `a` and `b`"""
        return [ps.join_patterns(a_, b_) for (ps, a_, b_) in zip(self.basic_structures, a, b)]

    def iter_bin_attributes(self, data: list[PatternType]) -> Iterator[tuple[PatternType, fbarray]]:
        """Iterate binary attributes obtained from `data` (from the most general to the most precise ones)

        :parameter
            data: list[PatternType]
             list of object descriptions
        :return
            iterator of (description: PatternType, extent of the description: frozenbitarray)
        """
        for i, ps in enumerate(self.basic_structures):
            ps_data = [data_row[i] for data_row in data]
            for pattern, flag in ps.iter_bin_attributes(ps_data):
                yield (i, pattern), flag

    def is_less_precise(self, a: PatternType, b: PatternType) -> bool:
        """Return True if pattern `a` is less precise than pattern `b`"""
        return all(ps.is_less_precise(a_, b_) for ps, a_, b_ in zip(self.basic_structures, a, b))

    def n_bin_attributes(self, data: list[PatternType]) -> int:
        """Count the number of attributes in the binary representation of `data`"""
        n_bin_attrs = 0
        for i, ps in enumerate(self.basic_structures):
            ps_data = [data_row[i] for data_row in data]
            n_bin_attrs += ps.n_bin_attributes(ps_data)
        return n_bin_attrs
