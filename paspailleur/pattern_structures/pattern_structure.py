from collections import deque, OrderedDict
from functools import reduce
from typing import Type, TypeVar, Union, Collection, Optional
from bitarray import bitarray, frozenbitarray as fbarray
from bitarray.util import zeros as bazeros, subset as basubset

from .pattern import Pattern


class PatternStructure:
    PatternType = TypeVar('PatternType', bound=Pattern)

    def __init__(self, pattern_type: Type[Pattern] = Pattern):
        self.PatternType = pattern_type
        self._object_irreducibles: Optional[dict[pattern_type, fbarray]] = None
        self._object_names: Optional[list[str]] = None
        self._atomic_patterns: Optional[OrderedDict[pattern_type, fbarray]] = None

    def extent(self, pattern: PatternType, return_bitarray: bool = False) -> Union[set[str], fbarray]:
        if not self._object_irreducibles or not self._object_names:
            raise ValueError('The data is unknown. Fit the PatternStructure to your data using .fit(...) method')

        n_objects = len(self._object_names)
        empty_extent = fbarray(bazeros(n_objects))
        sub_extents = (extent for ptrn, extent in self._object_irreducibles.items() if pattern <= ptrn)
        extent = reduce(fbarray.__or__, sub_extents, empty_extent)

        if return_bitarray:
            return fbarray(extent)
        return {self._object_names[g] for g in extent.search(True)}

    def intent(self, objects: Union[Collection[str], fbarray]) -> PatternType:
        if not self._object_irreducibles or not self._object_names:
            raise ValueError('The data is unknown. Fit the PatternStructure to your data using .fit(...) method')

        if not isinstance(objects, bitarray):
            objects_ba = bazeros(len(self._object_names))
            for object_name in objects:
                objects_ba[self._object_names.index(object_name)] = True
        else:
            objects_ba = objects

        super_patterns = [ptrn for ptrn, irr_ext in self._object_irreducibles.items() if basubset(irr_ext, objects_ba)]
        if super_patterns:
            return reduce(self.PatternType.__and__, super_patterns)
        return reduce(self.PatternType.__or__, self._object_irreducibles)

    def fit(self, object_descriptions: dict[str, PatternType], compute_atomic_patterns: bool = None):
        n_objects = len(object_descriptions)
        empty_extent = bazeros(n_objects)

        object_names = []
        object_irreducibles = dict()
        for g, (object_name, object_description) in enumerate(object_descriptions.items()):
            object_names.append(object_name)
            if object_description not in object_irreducibles:
                object_irreducibles[object_description] = empty_extent.copy()
            object_irreducibles[object_description][g] = True
        object_irreducibles = {pattern: fbarray(extent) for pattern, extent in object_irreducibles.items()}

        self._object_names = object_names
        self._object_irreducibles = object_irreducibles

        if compute_atomic_patterns is None:
            # Set to True if the values can be computed
            pattern = list(object_irreducibles)[0]
            try:
                _ = pattern.atomic_patterns
                compute_atomic_patterns = True
            except NotImplementedError:
                compute_atomic_patterns = False
        if compute_atomic_patterns:
            self.init_atomic_patterns()

    @property
    def min_pattern(self) -> PatternType:
        if not self._object_irreducibles:
            raise ValueError('The data is unknown. Fit the PatternStructure to your data using .fit(...) method')
        some_pattern = list(self._object_irreducibles)[0]
        if some_pattern.min_pattern is None:
            min_pattern = reduce(self.PatternType.__and__, self._object_irreducibles, some_pattern)
        else:
            min_pattern = some_pattern.min_pattern
        return min_pattern

    @property
    def max_pattern(self) -> PatternType:
        if not self._object_irreducibles:
            raise ValueError('The data is unknown. Fit the PatternStructure to your data using .fit(...) method')

        some_pattern = list(self._object_irreducibles)[0]
        if some_pattern.max_pattern is None:
            max_pattern = reduce(self.PatternType.__or__, self._object_irreducibles, some_pattern)
        else:
            max_pattern = some_pattern.max_pattern
        return max_pattern

    def init_atomic_patterns(self):
        """Compute the set of all patterns that cannot be obtained by intersection of other patterns"""
        atomic_patterns = reduce(set.__or__, (p.atomic_patterns for p in self._object_irreducibles), set())
        patterns_per_extent: dict[fbarray, deque[Pattern]] = dict()
        for atomic_pattern in atomic_patterns:
            extent: fbarray = self.extent(atomic_pattern, return_bitarray=True)
            if extent not in patterns_per_extent:
                patterns_per_extent[extent] = deque([atomic_pattern])
                continue
            # extent in patterns_per_extent, i.e. there are already some known patterns per extent
            equiv_patterns = patterns_per_extent[extent]
            greater_patterns = (i for i, other in enumerate(equiv_patterns) if atomic_pattern <= other)
            first_greater_pattern = next(greater_patterns, len(equiv_patterns)+1)
            patterns_per_extent[extent].insert(first_greater_pattern-1, atomic_pattern)

        sorted_extents = sorted(patterns_per_extent, key=lambda ext: (-ext.count(), ext.search(True)))
        self._atomic_patterns = OrderedDict([(ptrn, ext) for ext in sorted_extents for ptrn in patterns_per_extent[ext]])
