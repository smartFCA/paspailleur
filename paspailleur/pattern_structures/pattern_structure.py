from functools import reduce
from typing import Type, TypeVar, Union, Collection
from bitarray import bitarray, frozenbitarray as fbarray
from bitarray.util import zeros as bazeros, subset as basubset

from .pattern import Pattern


class PatternStructure:
    PatternType = TypeVar('PatternType', bound=Pattern)

    def __init__(self, pattern_type: Type[Pattern] = Pattern):
        self.PatternType = pattern_type
        self._object_irreducibles: dict[pattern_type, fbarray] = None
        self._object_names: list[str] = None
        self._atomic_patterns: list[pattern_type] = None

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

    def fit(self, object_descriptions: dict[str, PatternType]):
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

    @property
    def min_pattern(self):
        if not self._object_irreducibles:
            raise ValueError('The data is unknown. Fit the PatternStructure to your data using .fit(...) method')
        return reduce(self.PatternType.__and__, self._object_irreducibles, list(self._object_irreducibles)[0])

    def init_atomic_patterns(self):
        """Compute the set of all patterns that cannot be obtained by intersection of other patterns"""
        self._atomic_patterns = list(reduce(set.__or__, (p.atomic_patterns for p in self._object_irreducibles), set()))
