from collections import deque
from dataclasses import dataclass
from typing import TypeVar, Iterator, Iterable, Union, Optional
from bitarray import frozenbitarray as fbarray
from bitarray.util import zeros as bazeros
from tqdm.autonotebook import tqdm
from deprecation import deprecated


@dataclass
class ProjectionNotFoundError(ValueError):
    projection_number: int

    def __str__(self):
        return f"Projection #{self.projection_number} could not be computed"


class AbstractPS:
    PatternType = TypeVar('PatternType')  # A type that should be hashable and support __eq__ comparison
    min_pattern: PatternType  # Pattern, less specific than any other one. Should be of PatternType, i.e. NOT None
    max_pattern: PatternType  # Pattern, more specific than any other one. Should be of PatternType, i.e. NOT None

    def join_patterns(self, a: PatternType, b: PatternType) -> PatternType:
        """Return the most precise common pattern, describing both patterns `a` and `b`"""
        raise NotImplementedError

    def is_less_precise(self, a: PatternType, b: PatternType) -> bool:
        """Return True if pattern `a` is less precise than pattern `b`"""
        return self.join_patterns(a, b) == a

    def extent(self, data: list[PatternType], pattern: Optional[PatternType] = None) -> Iterator[int]:
        """Return indices of rows in `data` whose description contains `pattern`"""
        pattern = pattern if pattern is not None else self.min_pattern
        if pattern == self.min_pattern:
            return (i for i in range(len(data)))
        return (i for i, obj_description in enumerate(data) if self.is_less_precise(pattern, obj_description))

    def intent(self, data: list[PatternType], indices: Iterable[int] = None) -> PatternType:
        """Return common pattern of all rows in `data`"""
        iterator = (data[i] for i in indices) if indices is not None else data

        intent = self.max_pattern
        for obj_description in iterator:
            intent = self.join_patterns(intent, obj_description)
        return intent

    def iter_attributes(self, data: list[PatternType], min_support: Union[int, float] = 0)\
            -> Iterator[tuple[PatternType, fbarray]]:
        """Iterate binary attributes obtained from `data` (from the most general to the most precise ones)

        :parameter
            data: list[PatternType]
             list of object descriptions
            min_support: int or float
             minimal amount of objects an attribute should describe (in natural numbers, not per cents)
        :return
            iterator of (description: PatternType, extent of the description: frozenbitarray)
        """
        min_support = int(min_support*len(data)) if isinstance(min_support, float) else min_support
        total_extent = fbarray(~bazeros(len(data)))
        assert total_extent.count() >= min_support,\
            f"Provided min support {min_support} covers more objects than there exists (i.e. {len(total_extent)})"
        queue = deque([(self.min_pattern, total_extent)])
        while queue:
            attr, extent = queue.popleft()
            yield attr, extent

            next_attrs = self.closest_more_precise(attr, use_lectic_order=True)
            for next_attr in next_attrs:
                next_extent = bazeros(len(total_extent))
                for g_i in self.extent(data, next_attr):
                    next_extent[g_i] = True

                if next_extent.count() < min_support:
                    continue

                queue.append((next_attr, fbarray(next_extent)))

    @deprecated(deprecated_in='0.1.0', removed_in='0.1.1', details='The function is renamed to `iter_attributes`')
    def iter_bin_attributes(self, data: list[PatternType], min_support: Union[int, float] = 0) \
            -> Iterator[tuple[PatternType, fbarray]]:
        """Iterate binary attributes obtained from `data` (from the most general to the most precise ones)

        :parameter
            data: list[PatternType]
             list of object descriptions
            min_support: int or float
             minimal amount of objects an attribute should describe (in natural numbers, not per cents)
        :return
            iterator of (description: PatternType, extent of the description: frozenbitarray)
        """
        return self.iter_attributes(data, min_support)

    def n_attributes(self, data: list[PatternType], min_support: Union[int, float] = 0, use_tqdm: bool = False)\
            -> int:
        """Count the number of attributes in the binary representation of `data`"""
        iterator = self.iter_attributes(data, min_support)
        if use_tqdm:
            iterator = tqdm(iterator, desc='Counting patterns')
        return sum(1 for _ in iterator)

    @deprecated(deprecated_in='0.1.0', removed_in='0.1.1', details='The function is renamed to `n_attributes`')
    def n_bin_attributes(self, data: list[PatternType], min_support: Union[int, float] = 0, use_tqdm: bool = False) \
            -> int:
        """Count the number of attributes in the binary representation of `data`"""
        return self.n_attributes(data, min_support, use_tqdm)

    def binarize(self, data: list[PatternType], min_support: Union[int, float] = 0)\
            -> tuple[list[PatternType], list[fbarray]]:
        """Binarize the data into Formal Context

        :parameter
            data: list[PatternType]
                List of row descriptions
            min_support: int or float
                minimal amount of objects an attribute should describe (in natural numbers, not per cents)
        :return
            patterns: list[PatternType]
                Patterns corresponding to the attributes in the binarised data (aka binary attribute names)
            itemsets_ba: list[frozenbitarray]
                List of itemsets for every row in `data`.
                `itemsets_ba[i][j]` shows whether `data[i]` contains`patterns[j]`
        """
        patterns, flags = list(zip(*list(self.iter_attributes(data, min_support))))

        n_rows, n_cols = len(flags[0]), len(flags)
        itemsets_ba = [bazeros(n_cols) for _ in range(n_rows)]
        for j, flag in enumerate(flags):
            for i in flag.itersearch(True):
                itemsets_ba[i][j] = True
        itemsets_ba = [fbarray(ba) for ba in itemsets_ba]
        return list(patterns), itemsets_ba

    def preprocess_data(self, data: Iterable) -> Iterator[PatternType]:
        """Preprocess the data into to the format, supported by attrs_order/extent functions"""
        for description in data:
            yield description

    def verbalize(self, description: PatternType) -> str:
        """Convert `description` into human-readable string"""
        return f"{description}"

    def closest_less_precise(self, description: PatternType, use_lectic_order: bool = False) -> Iterator[PatternType]:
        """Return closest descriptions that are less precise than `description`

        Use lectic order for optimisation of description traversal
        """
        raise NotImplementedError

    def closest_more_precise(self, description: PatternType, use_lectic_order: bool = False) -> Iterator[PatternType]:
        """Return closest descriptions that are more precise than `description`

        Use lectic order for optimisation of description traversal
        """
        raise NotImplementedError

    def keys(self, intent: PatternType, data: list[PatternType]) -> list[PatternType]:
        """Return the least precise descriptions equivalent to the given attrs_order"""
        extent = set(self.extent(data, intent))
        outer_data = [data[i] for i in range(len(data)) if i not in extent]

        keys_candidates, keys = deque([intent]), []
        while keys_candidates:
            key_candidate = keys_candidates.popleft()
            subdescriptions = self.closest_less_precise(key_candidate, use_lectic_order=True)
            has_next_descriptions = False
            for next_descr in subdescriptions:
                no_outer_extent = all(0 for _ in self.extent(outer_data, next_descr))
                if no_outer_extent:
                    keys_candidates.append(next_descr)
                    has_next_descriptions = True
            if not has_next_descriptions:
                keys.append(key_candidate)

        keys_candidates, keys = keys, []
        for key_candidate in keys_candidates:
            if not any(self.is_less_precise(key, key_candidate) for key in keys):
                keys = [key for key in keys if not self.is_less_precise(key_candidate, key)]
                keys.append(key_candidate)

        return keys
