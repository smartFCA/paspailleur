from collections import deque
from dataclasses import dataclass
from functools import reduce
from math import ceil
from numbers import Number
from typing import Iterator, TypeVar, Union, Iterable, Container, Hashable
from bitarray import frozenbitarray as fbarray, bitarray
from bitarray.util import zeros as bazeros
from deprecation import deprecated
from caspailleur.base_functions import isets2bas

from .abstract_ps import AbstractPS

from itertools import combinations

T = TypeVar('T')


@dataclass
class ValuesUniverseUndefined(ValueError):
    parameter_name: str

    def __str__(self):
        return f'`{self.parameter_name}` variable should be properly defined. ' \
            'You can use self.__init__ method to pass `all_values` parameter ' \
            'or pass your data through `self.preprocess_data` function'


class DisjunctiveSetPS(AbstractPS):
    """A PS where every description is a set of values. And the bigger is the set, the less precise is the description

    E.g. description {'green', 'yellow', 'red'} is less precise than {'green', 'yellow'}
    as the former describes all the objects that are 'green' OR 'yellow' OR 'red'
    and the latter only describes the objects that are 'green' OR 'yellow'.

    Such Pattern Structure can be applied for categorical values in tabular data.
    """
    PatternType = frozenset[T]
    MIN_PATTERN_PLACEHOLDER = frozenset({'<ALL_VALUES>'})
    min_pattern = MIN_PATTERN_PLACEHOLDER  # Top pattern, less specific than any other one
    max_pattern = frozenset()  # Bottom pattern, more specific than any other one

    def __init__(self, all_values: set = None):
        self.min_pattern = frozenset(all_values) if all_values else self.MIN_PATTERN_PLACEHOLDER

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
        min_support = ceil(len(data) * min_support) if 0 < min_support < 1 else int(min_support)
        n_objects = len(data)
        empty_extent = fbarray(bazeros(n_objects))

        vals_extents: dict[T, bitarray] = {}
        for i, pattern in enumerate(data):
            for v in pattern:
                if v not in vals_extents:
                    vals_extents[v] = bitarray(empty_extent)
                vals_extents[v][i] = True

        for value in reversed(sorted(vals_extents)):
            pattern = frozenset(vals_extents) - {value}
            extent = reduce(fbarray.__or__, (vals_extents[v] for v in pattern), empty_extent)
            if extent.count() < min_support:
                continue
            yield pattern, extent

    def n_attributes(self, data: list[PatternType], min_support: Union[int, float] = 0, use_tqdm: bool = False)\
            -> int:
        """Count the number of attributes in the binary representation of `data`"""
        if min_support == 0:
            unique_values = reduce(set.__or__, data, set())
            return len(unique_values)
        return super().n_attributes(data, min_support)

    def preprocess_data(self, data: Iterable[Union[Number, str, Container[Hashable]]]) -> Iterator[PatternType]:
        """Preprocess the data into to the format, supported by attrs_order/extent functions"""
        all_values = set()
        for description in data:
            if isinstance(description, (Number, str)):
                description = {description}
            if isinstance(description, Container):
                description = frozenset(description)
            else:
                raise ValueError(f'Cannot preprocess this description: {description}. '
                                 f'Provide either a number or a string or a container of hashable values.')

            yield description
            all_values |= description

        if self.min_pattern == self.MIN_PATTERN_PLACEHOLDER:
            self.min_pattern = frozenset(all_values)

    def verbalize(self, description: PatternType, separator: str = ', ', add_curly_braces: bool = False) -> str:
        """Convert `description` into human-readable string"""
        if not description and not add_curly_braces:
            return '∅'

        description_verb = separator.join([f"{v}" for v in sorted(description)])
        if add_curly_braces:
            description_verb = '{' + description_verb + '}'
        return description_verb

    def closest_less_precise(self, description: PatternType, use_lectic_order: bool = False) -> Iterator[PatternType]:
        """Return closest descriptions that are less precise than `description`

        Use lectic order for optimisation of description traversal
        """
        if self.min_pattern == self.MIN_PATTERN_PLACEHOLDER:
            raise ValuesUniverseUndefined('self.min_pattern')

        if description == self.min_pattern:
            return iter([])

        if not use_lectic_order:
            return (description | {attr} for attr in self.min_pattern - description)

        all_values = sorted(self.min_pattern)
        max_attr = next(attr_i for attr_i in reversed(range(len(all_values))) if all_values[attr_i] in description)
        return (description | {attr} for attr in all_values[max_attr+1:])

    def closest_more_precise(
            self, description: PatternType, use_lectic_order: bool = False, intent: PatternType = None
    ) -> Iterator[PatternType]:
        """Return closest descriptions that are more precise than `description`

        Use lectic order for optimisation of description traversal
        """
        if description == self.max_pattern:
            return iter([])

        if not use_lectic_order:
            return (description - {attr} for attr in description)

        if intent is None and self.min_pattern == self.MIN_PATTERN_PLACEHOLDER:
            raise ValuesUniverseUndefined('self.min_pattern')
        all_values = sorted(self.min_pattern if intent is None else intent)
        missing_attrs = (attr_i for attr_i in reversed(range(len(all_values))) if all_values[attr_i] not in description)
        max_missing_attr = next(missing_attrs) if len(all_values) != len(description) else -1
        return (description - {attr} for attr in all_values[max_missing_attr+1:] if attr in description)


class ConjunctiveSetPS(AbstractPS):
    """A PS where every description is a set of values. And the smaller is the set, the less precise is the description

    E.g. description {'green', 'cubic'} is less precise than {'green', 'cubic', 'heavy'}
    as the former describes all the objects that are 'green' AND 'cubic'
    and the latter describes the objects that are 'green' AND 'cubic' AND 'heavy'.

    """
    PatternType = frozenset[T]
    MAX_PATTERN_PLACEHOLDER = frozenset({'<ALL_VALUES>'})
    min_pattern = frozenset()  # Empty set that is always contained in any other set of values
    max_pattern = MAX_PATTERN_PLACEHOLDER  # Maximal pattern that should be more precise than any other pattern

    def __init__(self, all_values: set[T] = None):
        self.max_pattern = frozenset(all_values) if all_values else self.MAX_PATTERN_PLACEHOLDER

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

    def iter_attributes(self, data: list[PatternType], min_support: Union[int, float] = 0)\
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
        n_objects = len(data)
        min_support = ceil(n_objects * min_support) if 0 < min_support < 1 else int(min_support)

        empty_extent = bazeros(len(data))
        vals_extents: dict[T, bitarray] = {}
        for i, pattern in enumerate(data):
            for v in pattern:
                if v not in vals_extents:
                    vals_extents[v] = empty_extent.copy()
                vals_extents[v][i] = True

        for v in sorted(vals_extents):
            extent = vals_extents[v]
            if extent.count() < min_support:
                continue
            yield frozenset({v}), fbarray(extent)

        # bottom_extent = reduce(fbarray.__and__, vals_extents.values(), ~empty_extent)
        # if bottom_extent.count() >= min_support:
        #     yield frozenset(vals_extents), fbarray(bottom_extent)

    def n_attributes(self, data: list[PatternType], min_support: Union[int, float] = 0, use_tqdm: bool = False)\
            -> int:
        """Count the number of attributes in the binary representation of `data`"""
        if min_support == 0:
            unique_values = reduce(set.__or__, data, set())
            return len(unique_values)
        return super().n_attributes(data, min_support)

    def preprocess_data(self, data: Iterable[Union[Number, str, Container[Hashable]]]) -> Iterator[PatternType]:
        """Preprocess the data into to the format, supported by attrs_order/extent functions"""
        all_values = set()
        for description in data:
            if isinstance(description, (Number, str)):
                description = {description}
            if isinstance(description, Container):
                description = frozenset(description)
            else:
                raise ValueError(f'Cannot preprocess this description: {description}. '
                                 f'Provide either a number or a string or a container of hashable values.')

            yield description
            all_values |= description

        if self.max_pattern == self.MAX_PATTERN_PLACEHOLDER:
            self.max_pattern = frozenset(all_values)

    def verbalize(self, description: PatternType, separator: str = ', ', add_curly_braces: bool = False) -> str:
        """Convert `description` into human-readable string"""
        if not description and not add_curly_braces:
            return '∅'

        description_verb = separator.join([f"{v}" for v in sorted(description)])
        if add_curly_braces:
            description_verb = '{' + description_verb + '}'
        return description_verb

    def closest_less_precise(
            self, description: PatternType, use_lectic_order: bool = False, intent: PatternType = None
    ) -> Iterator[PatternType]:
        """Return closest descriptions that are less precise than `description`

        Use lectic order for optimisation of description traversal
        """
        if description == self.min_pattern:
            return iter([])

        if not use_lectic_order:
            return (description - {attr} for attr in description)

        if intent is None and self.max_pattern == self.MAX_PATTERN_PLACEHOLDER:
            raise ValuesUniverseUndefined('self.max_pattern')
        all_values = sorted(self.max_pattern if intent is None else intent)
        missing_attrs = (attr_i for attr_i in reversed(range(len(all_values))) if all_values[attr_i] not in description)
        max_missing_attr = next(missing_attrs) if len(description) != len(all_values) else -1
        return (description - {attr} for attr in all_values[max_missing_attr+1:] if attr in description)

    def closest_more_precise(self, description: PatternType, use_lectic_order: bool = False) -> Iterator[PatternType]:
        """Return closest descriptions that are more precise than `description`

        Use lectic order for optimisation of description traversal
        """
        if self.max_pattern == self.MAX_PATTERN_PLACEHOLDER:
            raise ValuesUniverseUndefined('self.max_pattern')

        if description == self.max_pattern:
            return iter([])

        if not use_lectic_order:
            return (description | {attr} for attr in self.max_pattern - description)

        all_values = sorted(self.max_pattern)
        attr_indices = (attr_i for attr_i in reversed(range(len(all_values))) if all_values[attr_i] in description)
        max_attr = next(attr_indices) if description else -1
        return (description | {attr} for attr in all_values[max_attr+1:])

    def keys(self, intent: PatternType, data: list[PatternType]) -> list[PatternType]:
        """Return the least precise descriptions equivalent to the given attrs_order"""
        extent = set(self.extent(data, intent))
        outer_data = [data[i] for i in range(len(data)) if i not in extent]

        keys_candidates, keys = deque([intent]), []
        while keys_candidates:
            key_candidate = keys_candidates.popleft()
            subdescriptions = self.closest_less_precise(key_candidate, use_lectic_order=True, intent=intent)

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


@deprecated(deprecated_in='0.1.0', removed_in='0.1.1')
class SuperSetPS(DisjunctiveSetPS):
    pass


@deprecated(deprecated_in='0.1.0', removed_in='0.1.1')
class SubSetPS(ConjunctiveSetPS):
    pass
