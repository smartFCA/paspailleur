import warnings
from collections import deque, OrderedDict
from functools import reduce
from typing import Type, TypeVar, Union, Collection, Optional, Iterator, Generator, Literal
from bitarray import bitarray, frozenbitarray as fbarray
from bitarray.util import zeros as bazeros

from caspailleur.io import to_absolute_number
from caspailleur.order import sort_intents_inclusion, inverse_order
from .pattern import Pattern

from paspailleur.algorithms import base_functions as bfuncs, mine_equivalence_classes as mec


class PatternStructure:
    PatternType = TypeVar('PatternType', bound=Pattern)

    def __init__(self, pattern_type: Type[Pattern] = Pattern):
        self.PatternType = pattern_type
        # patterns introduced by objects, related to what exact objects they introduce
        self._object_irreducibles: Optional[dict[pattern_type, fbarray]] = None
        self._object_names: Optional[list[str]] = None
        # smallest nontrivial patterns, related to what objects they describe
        self._atomic_patterns: Optional[OrderedDict[pattern_type, fbarray]] = None
        # list of indices of greater atomic patterns per every atomic pattern
        self._atomic_patterns_order: Optional[list[fbarray]] = None

    def extent(self, pattern: PatternType, return_bitarray: bool = False) -> Union[set[str], fbarray]:
        if not self._object_irreducibles or not self._object_names:
            raise ValueError('The data is unknown. Fit the PatternStructure to your data using .fit(...) method')

        extent = bfuncs.extension(pattern, self._object_irreducibles)

        if return_bitarray:
            return fbarray(extent)
        return {self._object_names[g] for g in extent.search(True)}

    def intent(self, objects: Union[Collection[str], fbarray]) -> PatternType:
        if not self._object_irreducibles or not self._object_names:
            raise ValueError('The data is unknown. Fit the PatternStructure to your data using .fit(...) method')

        objects_ba = objects
        if not isinstance(objects_ba, bitarray):
            objects_ba = bazeros(len(self._object_names))
            for object_name in objects:
                objects_ba[self._object_names.index(object_name)] = True

        return bfuncs.intention(objects_ba, self._object_irreducibles)

    def fit(self, object_descriptions: dict[str, PatternType], compute_atomic_patterns: bool = None):
        object_names, objects_patterns = zip(*object_descriptions.items())
        object_irreducibles = bfuncs.group_objects_by_patterns(objects_patterns)

        self._object_names = list(object_names)
        self._object_irreducibles = {k: fbarray(v) for k, v in object_irreducibles.items()}

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
        return bfuncs.minimal_pattern(self._object_irreducibles)

    @property
    def max_pattern(self) -> PatternType:
        if not self._object_irreducibles:
            raise ValueError('The data is unknown. Fit the PatternStructure to your data using .fit(...) method')

        return bfuncs.maximal_pattern(self._object_irreducibles)

    def init_atomic_patterns(self):
        """Compute the set of all patterns that cannot be obtained by intersection of other patterns"""
        atomic_patterns = reduce(set.__or__, (p.atomic_patterns for p in self._object_irreducibles), set())
        atomic_patterns |= self.max_pattern.atomic_patterns

        # Step 1. Group patterns by their extents. For every extent, list patterns in topological sorting
        patterns_per_extent: dict[fbarray, deque[Pattern]] = dict()
        for atomic_pattern in atomic_patterns:
            extent: fbarray = self.extent(atomic_pattern, return_bitarray=True)
            if extent not in patterns_per_extent:
                patterns_per_extent[extent] = deque([atomic_pattern])
                continue
            # extent in patterns_per_extent, i.e. there are already some known patterns per extent
            equiv_patterns = patterns_per_extent[extent]
            greater_patterns = (i for i, other in enumerate(equiv_patterns) if atomic_pattern <= other)
            first_greater_pattern = next(greater_patterns, len(equiv_patterns))
            patterns_per_extent[extent].insert(first_greater_pattern, atomic_pattern)

        # Step 2. Find order on atomic patterns.
        def sort_extents_subsumption(extents):
            empty_extent = extents[0] & ~extents[0]
            if not extents[0].all():
                extents.insert(0, ~empty_extent)
            if extents[-1].any():
                extents.append(empty_extent)
            inversed_extents_subsumption_order = inverse_order(sort_intents_inclusion(extents[::-1], use_tqdm=False, return_transitive_order=True)[1])
            extents_subsumption_order = [ba[::-1] for ba in inversed_extents_subsumption_order[::-1]]
            if ~empty_extent not in patterns_per_extent:
                extents.pop(0)
                extents_subsumption_order = [ba[1:] for ba in extents_subsumption_order[1:]]
            if empty_extent not in patterns_per_extent:
                extents.pop(-1)
                extents_subsumption_order = [ba[:-1] for ba in extents_subsumption_order[:-1]]
            return extents_subsumption_order

        sorted_extents = sorted(patterns_per_extent, key=lambda ext: (-ext.count(), tuple(ext.search(True))))
        extents_order = sort_extents_subsumption(sorted_extents)
        extents_to_idx_map = {extent: idx for idx, extent in enumerate(sorted_extents)}

        atomic_patterns, atomic_extents = zip(*[(ptrn, ext) for ext in sorted_extents for ptrn in patterns_per_extent[ext]])
        pattern_to_idx_map = {pattern: idx for idx, pattern in enumerate(atomic_patterns)}
        n_patterns = len(atomic_patterns)

        # patterns pointing to the bitarray of indices of next greater patterns
        patterns_order: list[bitarray] = [None for _ in range(n_patterns)]
        for pattern in reversed(atomic_patterns):
            idx = pattern_to_idx_map[pattern]
            extent = atomic_extents[idx]
            extent_idx = extents_to_idx_map[extent]

            # select patterns that might be greater than the current one
            patterns_to_test = bazeros(n_patterns)
            n_greater_patterns_same_extent = len(patterns_per_extent[extent]) - patterns_per_extent[extent].index(pattern)-1
            patterns_to_test[idx+1:idx+n_greater_patterns_same_extent+1] = True
            for smaller_extent_idx in extents_order[extent_idx].search(True):
                other_extent = sorted_extents[smaller_extent_idx]
                first_other_pattern_idx = pattern_to_idx_map[patterns_per_extent[other_extent][0]]
                n_patterns_other_extent = len(patterns_per_extent[other_extent])
                patterns_to_test[first_other_pattern_idx:first_other_pattern_idx+n_patterns_other_extent] = True

            # find patterns that are greater than the current one
            super_patterns = bazeros(n_patterns)
            while patterns_to_test.any():
                other_idx = patterns_to_test.find(True)
                patterns_to_test[other_idx] = False

                other = atomic_patterns[other_idx]
                if pattern < other:
                    super_patterns[other_idx] = True
                    super_patterns |= patterns_order[other_idx]
                    patterns_to_test &= ~patterns_order[other_idx]
            patterns_order[idx] = super_patterns

        atomic_patterns = OrderedDict([(ptrn, ext) for ext in sorted_extents for ptrn in patterns_per_extent[ext]])
        self._atomic_patterns = atomic_patterns
        self._atomic_patterns_order = [fbarray(ba) for ba in patterns_order]

    def iter_atomic_patterns(
        self,
        return_extents: bool = True, return_bitarrays: bool = False,
        kind: Literal["bruteforce", "ascending", "ascending controlled"] = 'bruteforce'
    ) -> Union[
        Generator[PatternType, bool, None],
        Generator[tuple[PatternType, set[str]], bool, None],
        Generator[tuple[PatternType, fbarray], bool, None]
    ]:
        assert self._atomic_patterns is not None

        def form_yielded_value(ptrn: PatternStructure.PatternType, ext: bitarray):
            if not return_extents:
                return ptrn
            if not return_bitarrays:
                return ptrn, {self._object_names[g] for g in ext.search(True)}
            return ptrn, ext

        if kind == 'bruteforce':
            for pattern, extent in self._atomic_patterns.items():
                yield form_yielded_value(pattern, extent)

        if kind == 'ascending':
            assert self._atomic_patterns_order is not None
            iterator = bfuncs.iter_patterns_ascending(self._atomic_patterns, self._atomic_patterns_order, False)
            for pattern, extent in iterator:
                yield form_yielded_value(pattern, extent)

        if kind == 'ascending controlled':
            assert self._atomic_patterns_order is not None
            iterator = bfuncs.iter_patterns_ascending(self._atomic_patterns, self._atomic_patterns_order, True)
            next(iterator)  # initialise
            yield
            go_more_precise = True
            while True:
                try:
                    pattern, extent = iterator.send(go_more_precise)
                except StopIteration:
                    break
                go_more_precise = yield form_yielded_value(pattern, extent)

    @property
    def atomic_patterns(self) -> OrderedDict[PatternType, set[str]]:
        return OrderedDict(self.iter_atomic_patterns(return_extents=True, return_bitarrays=False))

    @property
    def atomic_patterns_order(self) -> dict[PatternType, set[PatternType]]:
        if self._atomic_patterns_order is None:
            return None
        atomic_patterns_list = list(self._atomic_patterns)
        return {atomic_patterns_list[idx]: {atomic_patterns_list[v] for v in vs.search(True)}
                for idx, vs in enumerate(self._atomic_patterns_order)}

    @property
    def n_atomic_patterns(self) -> int:
        return sum(1 for _ in self.iter_atomic_patterns(return_extents=False, return_bitarrays=False))

    def iter_premaximal_patterns(self, return_extents: bool = True, return_bitarrays: bool = False) -> Union[
        Iterator[PatternType], Iterator[tuple[PatternType, set[str]]], Iterator[tuple[PatternType, fbarray]]
    ]:
        assert self._object_irreducibles is not None, \
            "Please define object-irreducible patterns (i.e. via .fit() function) " \
            "to be able to define premaximal_patterns"

        border_pattern_extents = {
            pattern: self.extent(pattern=pattern, return_bitarray=True) for pattern in self._object_irreducibles}
        premaximals = sorted(
            border_pattern_extents,
            key=lambda pattern: (border_pattern_extents[pattern].count(),
                                 tuple(border_pattern_extents[pattern].search(True))))
        # now smallest patterns at the start, maximals at the end

        i = 0
        while i < len(premaximals):
            pattern = premaximals[i]
            if any(other >= pattern for other in premaximals[:i]):
                del premaximals[i]
                continue
            # current pattern is premaximal, i.e. exists no bigger nontrivial pattern
            i += 1

            if not return_extents:
                yield pattern
            else:
                extent = border_pattern_extents[pattern]
                if return_bitarrays:
                    yield pattern, extent
                else:
                    yield pattern, {self._object_names[g] for g in extent.search(True)}

    @property
    def premaximal_patterns(self) -> dict[PatternType, set[str]]:
        """Maximal patterns that describe fewest objects (and their extents)"""
        return dict(self.iter_premaximal_patterns(return_extents=True, return_bitarrays=False))

    def mine_concepts(
            self,
            min_support: Union[int, float] = 0, min_delta_stability: Union[int, float] = 0,
            algorithm: Literal['CloseByOne object-wise', 'gSofia'] = None,
            return_objects_as_bitarrays: bool = False,
            use_tqdm: bool = False
    ) -> Union[list[tuple[set[str], PatternType]], list[tuple[fbarray, PatternType]]]:
        SUPPORTED_ALGOS = {'CloseByOne object-wise', 'gSofia'}
        assert algorithm is None or algorithm in SUPPORTED_ALGOS,\
            f"Only the following algorithms are supported: {SUPPORTED_ALGOS}. " \
            f"Either choose of the supported algorithm or set algorithm=None " \
            f"to automatically choose the best algorithm based on the other hyperparameters"

        n_objects = len(self._object_names)
        min_support = to_absolute_number(min_support, n_objects)
        min_delta_stability = to_absolute_number(min_delta_stability, n_objects)

        if algorithm is None:
            if min_delta_stability > 0 or min_support > 0:
                algorithm = 'gSofia'
            else:
                algorithm = 'CloseByOne object-wise'

        if algorithm == 'CloseByOne object-wise':
            unused_parameters = []
            if min_support > 0:
                unused_parameters.append(f"{min_support=}")
            if min_delta_stability > 0:
                unused_parameters.append(f"{min_delta_stability}")
            if unused_parameters:
                warnings.warn(UserWarning(
                    f'The following parameters {", ".join(unused_parameters)} do not affect algorithm {algorithm}'))

            objects_patterns = [next(p for p, objs in self._object_irreducibles.items() if objs[g]) for g in range(n_objects)]
            concepts_generator = mec.iter_intents_via_ocbo(objects_patterns)
            extents_intents_dict: dict[fbarray, Pattern] = {fbarray(extent): intent for intent, extent in concepts_generator}

        if algorithm == 'gSofia':
            atomic_patterns_iterator = self.iter_atomic_patterns(
                return_extents=True, return_bitarrays=True, kind='ascending controlled'
            )
            extents_ba = mec.list_stable_extents_via_gsofia(
                atomic_patterns_iterator,
                min_delta_stability=min_delta_stability,
                min_supp=min_support,
                use_tqdm=use_tqdm,
                n_atomic_patterns=len(self._atomic_patterns)
            )
            extents_intents_dict: dict[fbarray, Pattern] = {fbarray(extent): self.intent(extent) for extent in extents_ba}

        extents_order = sorted(extents_intents_dict, key=lambda extent: (-extent.count(), tuple(extent.search(True))))
        concepts = [(
            extent_ba if return_objects_as_bitarrays else self.verbalise_extent(extent_ba),
            extents_intents_dict[extent_ba]
        ) for extent_ba in extents_order]
        return concepts

    def iter_patterns(
            self, kind: Literal['ascending', 'ascending controlled'] = 'ascending',
            min_support: Union[int, float] = 0,
            depth_first: bool = True,
            return_objects_as_bitarrays: bool = False,
    ) -> Union[Iterator[tuple[PatternType, bitarray]], Generator[tuple[PatternType, bitarray], bool, None]]:
        assert self._atomic_patterns is not None,\
            "Initialise the atomic patterns with PatternStructure.init_atomic_patterns() function " \
            "to be able to iterate through the set of possible patterns"

        min_support = to_absolute_number(min_support, len(self._object_names))
        is_controlled = kind.split()[-1] == 'controlled'

        iterator = None
        if kind.split()[0] == 'ascending':
            iterator = mec.iter_all_patterns_ascending(self._atomic_patterns, min_support, depth_first,
                                                       controlled_iteration=is_controlled)

        if iterator is None:
            raise ValueError(f'Do not know how to treat parameter {kind=} '
                             f'in the PatternStructure.iter_patterns(...) function. '
                             f'See the list of available `kind` values in the type hints of the function.')

        if not is_controlled:
            for pattern, extent in iterator:
                yield pattern, extent if return_objects_as_bitarrays else self.verbalise_extent(extent)

        else:  # is_controlled = True
            go_deeper = True
            yield
            next(iterator)
            while True:
                try:
                    pattern, extent = iterator.send(go_deeper)
                except StopIteration as e:
                    break
                go_deeper = yield pattern, extent if return_objects_as_bitarrays else self.verbalise_extent(extent)

    def verbalise_extent(self, extent: Union[bitarray, set[str]]) -> set[str]:
        if not isinstance(extent, bitarray):
            return extent
        return {self._object_names[g] for g in extent.search(True)}

    def iter_keys(self, patterns: Union[PatternType, Collection[PatternType]]) -> Union[
        Iterator[PatternType], Iterator[tuple[PatternType, PatternType]]
    ]:
        if isinstance(patterns, self.PatternType):
            return mec.iter_keys_of_pattern(patterns, self._atomic_patterns)

        # `patterns` is a collection of patterns
        patterns = list(patterns)
        iterator = mec.iter_keys_of_patterns(patterns, self._atomic_patterns)
        return ((key, patterns[pattern_i]) for key, pattern_i in iterator)
