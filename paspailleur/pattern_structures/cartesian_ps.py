from typing import Iterator
from bitarray import frozenbitarray as fbarray
from .abstract_ps import AbstractPS

from tqdm.autonotebook import tqdm


class CartesianPS(AbstractPS):
    PatternType = tuple[tuple, ...]
    max_pattern: tuple  # Bottom pattern, more specific than any other one
    basic_structures: tuple[AbstractPS, ...]

    def __init__(self, basic_structures: list[AbstractPS]):
        self.basic_structures = tuple(basic_structures)
        self.max_pattern = tuple([ps.max_pattern for ps in basic_structures])

    def join_patterns(self, a: PatternType, b: PatternType) -> PatternType:
        """Return the most precise common pattern, describing both patterns `a` and `b`"""
        return tuple([ps.join_patterns(a_, b_) for (ps, a_, b_) in zip(self.basic_structures, a, b)])

    def is_less_precise(self, a: PatternType, b: PatternType) -> bool:
        """Return True if pattern `a` is less precise than pattern `b`"""
        return all(ps.is_less_precise(a_, b_) for ps, a_, b_ in zip(self.basic_structures, a, b))

    def iter_bin_attributes(self, data: list[PatternType], min_support: int | float = 0) -> Iterator[tuple[PatternType, fbarray]]:
        """Iterate binary attributes obtained from `data` (from the most general to the most precise ones)

        :parameter
            data: list[PatternType]
             list of object descriptions
            min_support: int
             minimal amount of objects an attribute should describe (in natural numbers, not per cents)
        :return
            iterator of (description: PatternType, extent of the description: frozenbitarray)
        """
        for i, ps in enumerate(self.basic_structures):
            ps_data = [data_row[i] for data_row in data]
            for pattern, flag in ps.iter_bin_attributes(ps_data, min_support):
                yield (i, pattern), flag

    def n_bin_attributes(self, data: list[PatternType], min_support: int | float = 0, use_tqdm: bool = False) -> int:
        """Count the number of attributes in the binary representation of `data`"""
        n_bin_attrs = 0
        iterator = enumerate(self.basic_structures)
        if use_tqdm:
            iterator = tqdm(iterator, desc='Iterating basic structures', total=len(self.basic_structures))
        for i, ps in iterator:
            ps_data = [data_row[i] for data_row in data]
            n_bin_attrs += ps.n_bin_attributes(ps_data, min_support=min_support, use_tqdm=use_tqdm)
        return n_bin_attrs
