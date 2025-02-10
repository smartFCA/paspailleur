import warnings
from collections import OrderedDict
from functools import reduce
from operator import itemgetter
from typing import Type, TypeVar, Union, Collection, Optional, Iterator, Generator, Literal, Iterable, Sized
from bitarray import bitarray, frozenbitarray as fbarray
from bitarray.util import zeros as bazeros

from caspailleur.io import to_absolute_number
from tqdm.auto import tqdm

from .pattern import Pattern

from paspailleur.algorithms import base_functions as bfuncs, mine_equivalence_classes as mec, mine_subgroups as msubg


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

    #################################
    # Basic operations on a context #
    #################################
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

    ###########################################
    # Initialisation of the Pattern Structure #
    ###########################################
    def fit(
            self,
            object_descriptions: dict[str, PatternType],
            compute_atomic_patterns: bool = None, min_atom_support: Union[int, float] = 0,
            use_tqdm: bool = True
    ):
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
            self.init_atomic_patterns(use_tqdm=use_tqdm, min_support=min_atom_support)

    def init_atomic_patterns(self, min_support: Union[int, float] = 0, use_tqdm: bool = False):
        """Compute the set of all patterns that cannot be obtained by intersection of other patterns"""
        min_support = to_absolute_number(min_support, len(self._object_names))

        atomic_patterns = reduce(set.__or__, (p.atomic_patterns for p in self._object_irreducibles), set())
        atomic_patterns |= self.max_atoms

        atoms_iterator = atomic_patterns
        if use_tqdm:
            atoms_iterator = tqdm(atoms_iterator, total=len(atomic_patterns), desc='Compute atomic extents')

        atoms_extents: list[tuple[Pattern, fbarray]] = []
        for atom in atoms_iterator:
            extent = self.extent(atom, return_bitarray=True)
            if extent.count() < min_support:
                continue
            atoms_extents.append((atom, extent))

        atoms_order = bfuncs.order_patterns_via_extents(atoms_extents, use_tqdm=use_tqdm)
        assert all(not ba[i] for i, ba in enumerate(atoms_order)), 'Something went wrong during the run of the programm. Ask the developer'

        def topological_key(atom_idx):
            extent: fbarray = atoms_extents[atom_idx][1]
            superatoms = atoms_order[atom_idx]

            return -extent.count(), tuple(extent.search(True)), -superatoms.count(), tuple(superatoms.search(True))
        topologic_indices = sorted(range(len(atoms_extents)), key=topological_key)
        atoms_order = bfuncs.rearrange_indices(atoms_order, atoms_extents, [atoms_extents[i] for i in topologic_indices])
        atoms_extents = [atoms_extents[i] for i in topologic_indices]

        # patterns pointing to the bitarray of indices of next greater patterns
        patterns_order: list[bitarray] = [None for _ in range(n_patterns)]
        patterns_iterator = tqdm(reversed(atomic_patterns), disable=not use_tqdm, desc='Compute order of atoms',
                                 total=len(atomic_patterns))
        for pattern in patterns_iterator:
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

    #######################################
    # Properties that are easy to compute #
    #######################################
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

    @property
    def max_atoms(self) -> set[PatternType]:
        some_pattern = next(p for p in self._object_irreducibles)
        max_atoms = some_pattern.maximal_atoms
        if max_atoms is None:
            return set()
        return max_atoms

    @property
    def atomic_patterns(self) -> OrderedDict[PatternType, set[str]]:
        return OrderedDict(self.iter_atomic_patterns(return_extents=True, return_bitarrays=False))

    @property
    def n_atomic_patterns(self) -> int:
        return sum(1 for _ in self.iter_atomic_patterns(return_extents=False, return_bitarrays=False))

    @property
    def atomic_patterns_order(self) -> dict[PatternType, set[PatternType]]:
        if self._atomic_patterns_order is None:
            return None
        atomic_patterns_list = list(self._atomic_patterns)
        return {atomic_patterns_list[idx]: {atomic_patterns_list[v] for v in vs.search(True)}
                for idx, vs in enumerate(self._atomic_patterns_order)}

    @property
    def premaximal_patterns(self) -> dict[PatternType, set[str]]:
        """Maximal patterns that describe fewest objects (and their extents)"""
        return dict(self.iter_premaximal_patterns(return_extents=True, return_bitarrays=False))

    #####################
    # Pattern Iterators #
    #####################
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

    def iter_keys(
            self,
            patterns: Union[PatternType, Iterable[PatternType]],
            max_length: Optional[int] = None
    ) -> Union[Iterator[PatternType], Iterator[tuple[PatternType, PatternType]]]:
        if isinstance(patterns, self.PatternType):
            return mec.iter_keys_of_pattern(patterns, self._atomic_patterns, max_length=max_length)

        # `patterns` is a collection of patterns
        patterns = list(patterns)
        iterator = mec.iter_keys_of_patterns(patterns, self._atomic_patterns, max_length=max_length)
        return ((key, patterns[pattern_i]) for key, pattern_i in iterator)

    def iter_subgroups(
            self,
            goal_objects: Union[set[str], bitarray],
            quality_measure: Literal['Accuracy', 'Precision', 'Recall', 'Jaccard', 'F1', 'WRAcc'],
            quality_threshold: float,
            kind: Literal["bruteforce"] = 'bruteforce',
            max_length: Optional[int] = None,
            return_objects_as_bitarrays: bool = False
    ) -> Iterator[tuple[Pattern, Union[set[str], bitarray], float]]:
        if not isinstance(goal_objects, bitarray):
            goal_objects = set(goal_objects)
            goal_objects = bitarray([obj_name in goal_objects for obj_name in self._object_names])

        quality_func, tp_min, fp_max = msubg.setup_quality_measure_function(
            quality_measure, quality_threshold, goal_objects.count(), len(goal_objects)
        )

        subgroups_iterator: Optional[Iterator[tuple[Pattern, bitarray, float]]] = None
        if kind == 'bruteforce':
            subgroups_iterator = msubg.iter_subgroups_bruteforce(
                self, goal_objects, quality_threshold, quality_func, tp_min, fp_max, max_length)

        if subgroups_iterator is None:
            raise ValueError(f'Submitted kind of iterator {kind=} is not supported. '
                             f'The only supported type for the moment is "bruteforce"')

        if return_objects_as_bitarrays:
            return subgroups_iterator
        return ((pattern, self.verbalise_extent(extent_ba), quality)
                for pattern, extent_ba, quality in subgroups_iterator)

    ######################
    # High-level FCA API #
    ######################
    def mine_concepts(
            self,
            min_support: Union[int, float] = 0, min_delta_stability: Union[int, float] = 0,
            algorithm: Literal['CloseByOne object-wise', 'gSofia'] = None,
            return_objects_as_bitarrays: bool = False,
            use_tqdm: bool = False
    ) -> Union[list[tuple[set[str], PatternType]], list[tuple[fbarray, PatternType]]]:
        SUPPORTED_ALGOS = {'CloseByOne object-wise', 'gSofia'}
        assert algorithm is None or algorithm in SUPPORTED_ALGOS, \
            f"Only the following algorithms are supported: {SUPPORTED_ALGOS}. " \
            f"Either choose of the supported algorithm or set algorithm=None " \
            f"to automatically choose the best algorithm based on the other hyperparameters"

        n_objects = len(self._object_names)
        min_support = to_absolute_number(min_support, n_objects)
        min_delta_stability = to_absolute_number(min_delta_stability, n_objects)

        algorithm = algorithm if algorithm is not None else 'gSofia'

        if algorithm == 'CloseByOne object-wise':
            unused_parameters = []
            if min_support > 0:
                unused_parameters.append(f"{min_support=}")
            if min_delta_stability > 0:
                unused_parameters.append(f"{min_delta_stability}")
            if unused_parameters:
                warnings.warn(UserWarning(
                    f'The following parameters {", ".join(unused_parameters)} do not affect algorithm {algorithm}'))

            objects_patterns = [next(p for p, objs in self._object_irreducibles.items() if objs[g]) for g in
                                range(n_objects)]
            concepts_generator = mec.iter_intents_via_ocbo(objects_patterns)
            extents_intents_dict: dict[fbarray, Pattern] = {fbarray(extent): intent for intent, extent in
                                                            concepts_generator}

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
            extents_intents_dict: dict[fbarray, Pattern] = dict()
            for extent in tqdm(extents_ba, disable=not use_tqdm, desc='Compute intents'):
                extents_intents_dict[fbarray(extent)] = self.intent(extent)

        extents_order = sorted(extents_intents_dict, key=lambda extent: (-extent.count(), tuple(extent.search(True))))
        concepts = [(
            extent_ba if return_objects_as_bitarrays else self.verbalise_extent(extent_ba),
            extents_intents_dict[extent_ba]
        ) for extent_ba in extents_order]
        return concepts

    def mine_implications(
            self,
            basis_name: Literal["Canonical", "Canonical Direct"] = "Canonical Direct",
            min_support: Union[int, float] = 0, min_delta_stability: Union[int, float] = 0,
            max_key_length: Optional[int] = None,
            algorithm: Literal['CloseByOne object-wise', 'gSofia'] = None,
            reduce_conclusions: bool = False,
            use_tqdm: bool = False,
    ) -> dict[PatternType, PatternType]:
        concepts: list[tuple[fbarray, PatternStructure.PatternType]] = self.mine_concepts(
            min_support=min_support, min_delta_stability=min_delta_stability,
            algorithm=algorithm, return_objects_as_bitarrays=True, use_tqdm=use_tqdm
        )
        intents = map(itemgetter(1), concepts)
        if not concepts[0][0].all():
            intents = [self.intent(concepts[0][0]|~concepts[0][0])] + list(intents)

        PType = PatternStructure.PatternType
        keys: Iterator[PType] = map(itemgetter(0), self.iter_keys(intents, max_length=max_key_length))
        keys = list(tqdm(keys, desc="Mine premise candidates", disable=not use_tqdm))

        pseudo_close_premises = basis_name == 'Canonical'
        return self.mine_implications_from_premises(
            keys,
            pseudo_close_premises=pseudo_close_premises, use_tqdm=use_tqdm, reduce_conclusions=reduce_conclusions
        )

    def mine_implications_from_premises(
            self,
            premises: Iterable[PatternType],
            pseudo_close_premises: bool = False, reduce_conclusions: bool = False,
            use_tqdm: bool = False
    ) -> Union[dict[PatternType, PatternType], OrderedDict[PatternType, PatternType]]:
        # construct implication basis at the first try

        premises = tqdm(
            premises, desc="Construct implications",
            disable=not use_tqdm, total=len(premises) if isinstance(premises, Sized) else None
        )
        implication_basis = OrderedDict()
        for premise in premises:
            intent = self.intent(self.extent(premise, return_bitarray=True))
            premise_saturated = premise
            for p, c in implication_basis.items():
                if p <= premise_saturated:
                    premise_saturated = premise_saturated | c
                if premise_saturated == intent:
                    break
            else:  # if key_saturated != intent
                implication_basis[premise] = intent if not reduce_conclusions else intent - premise_saturated

        if not pseudo_close_premises:
            return implication_basis

        iterator = tqdm(
            implication_basis.items(), desc='Closing Premises',
            disable=not use_tqdm, total=len(implication_basis)
        )
        pseudo_closed_basis = dict()
        for premise, conclusion in iterator:
            premise_saturated = premise
            for p, c in implication_basis.items():
                if p < premise_saturated:
                    premise_saturated = premise_saturated | c
                if premise_saturated == conclusion:
                    break
            else:  # if key_saturated != intent
                pseudo_closed_basis[premise_saturated] = conclusion if not reduce_conclusions else conclusion - premise_saturated
        return pseudo_closed_basis

    def next_patterns(
            self, pattern: PatternType,
            return_extents: bool = False, return_objects_as_bitarrays: bool = False,
    ) -> Union[set[PatternType], dict[PatternType, set[str]], dict[PatternType, fbarray]]:
        atom_iterator = self.iter_atomic_patterns(return_extents=True, return_bitarrays=True, kind='ascending controlled')
        extent = self.extent(pattern, return_bitarray=True)
        next(atom_iterator)  # initialise iterator

        next_patterns = dict()
        go_deeper = True
        while True:
            try:
                atom, atom_extent = atom_iterator.send(go_deeper)
            except StopIteration:
                break

            if atom <= pattern:
                go_deeper = True
                continue

            # atom >= pattern, or atom and pattern are incomparable
            next_patterns[pattern | atom] = extent & atom_extent
            go_deeper = False

        minimal_next_patterns = [p for p in next_patterns
                                 if not any(other <= p for other in next_patterns if other != p)]

        next_patterns = {p: next_patterns[p] for p in minimal_next_patterns}
        if not return_extents:
            return set(next_patterns)
        if return_objects_as_bitarrays:
            return next_patterns
        return {p: self.verbalise_extent(extent) for p, extent in next_patterns.items()}

    ########################
    # Measures of patterns #
    ########################
    def measure_support(self, pattern: Pattern) -> int:
        return self.extent(pattern, return_bitarray=True).count()

    def measure_frequency(self, pattern: Pattern) -> float:
        extent = self.extent(pattern, return_bitarray=True)
        return extent.count()/len(extent)

    def measure_delta_stability(self, pattern: Pattern) -> int:
        extent = self.extent(pattern, return_bitarray=True)

        subextents = self.next_patterns(pattern, return_extents=True, return_objects_as_bitarrays=True).values()
        return extent.count() - max((subext.count() for subext in subextents), default=0)

    #####################
    # Helping functions #
    #####################
    def verbalise_extent(self, extent: Union[bitarray, set[str]]) -> set[str]:
        if not isinstance(extent, bitarray):
            return extent
        return {self._object_names[g] for g in extent.search(True)}