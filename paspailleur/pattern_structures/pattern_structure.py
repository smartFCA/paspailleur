import warnings
from collections import OrderedDict
from functools import reduce, partial
from operator import itemgetter
from typing import Type, TypeVar, Union, Collection, Optional, Iterator, Generator, Literal, Iterable, Sized
from bitarray import bitarray, frozenbitarray as fbarray
from bitarray.util import zeros as bazeros, subset as basubset
from caspailleur import inverse_order

from caspailleur.io import to_absolute_number
from tqdm.auto import tqdm

from .pattern import Pattern

from paspailleur.algorithms import base_functions as bfuncs, mine_equivalence_classes as mec, mine_subgroups as msubg
from paspailleur.algorithms import mine_implication_bases as mib


class PatternStructure:
    """
    A class to process complex datasets where every row (called object) is described by one pattern.

    All patterns should be of the same class defined by PatternType attribute.

    References
    ..........
    Ganter, B., & Kuznetsov, S. O. (2001, July). Pattern structures and their projections. In International conference on conceptual structures (pp. 129-142). Berlin, Heidelberg: Springer Berlin Heidelberg.
    
    Attributes
    ..........
    PatternType: TypeVar
        A type variable bound to the Pattern class.
    
    Private Attributes
    ..................
    _object_irreducibles: Optional[dict[PatternType, fbarray]]
        Patterns introduced by objects.
    _object_names: Optional[list[str]]
        Names of the objects.
    _atomic_patterns: Optional[OrderedDict[PatternType, fbarray]]
        Smallest nontrivial patterns.
    _atomic_patterns_order: Optional[list[fbarray]]
        Partial order on atomic patterns meaning that some atomic patterns can be incomparable.
        _atomic_patterns_order[i][j] == True would mean that i-th atomic pattern is less precise than j-th atomic pattern.
        _atomic_patterns_order[i][j] == False can mean both that j-th atomic pattern is less precise than i-th atomic pattern and that i-th and j-th atomic patterns can not be compared.

    Properties
    ..........
    premaximal_patterns
        Return the premaximal patterns in the structure.
    atomic_patterns_order
        Return the partial order of atomic patterns by extent inclusion.
    n_atomic_patterns
        Return the number of atomic patterns in the structure.
    atomic_patterns
        Return the atomic patterns in the structure.
    max_atoms
        Return the maximal atomic patterns in the structure.
    max_pattern
        Return the maximal pattern in the structure.
    min_pattern
        Return the minimal pattern in the structure.
    """
    PatternType = TypeVar('PatternType', bound=Pattern)

    def __init__(self, pattern_type: Type[Pattern] = None):
        """
        Initialize the PatternStructure with a specific pattern class.

        Parameters
        ----------
        pattern_type: Type[Pattern], optional
            The type of Pattern to use (default is Pattern).
        """
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
        """
        Compute the extent of a given pattern.

        Parameters
        ----------
        pattern: PatternType
            The pattern for which to compute the extent.
        return_bitarray: bool, optional
            If True, returns the extent as a bitarray (default is False).

        Returns
        -------
        extent: Union[set[str], fbarray]
            The extent of the pattern, either as a set of object names or a bitarray.

        Raises
        ------
        ValueError
            If the data is unknown (i.e., not fitted).

        Examples
        --------
        >>> ps = PatternStructure()
        >>> ps.fit(data)  # Assume this fits the structure with object data
        >>> p = Pattern("A")
        >>> ps.extent(p)
        {'obj1', 'obj2'}
        """
        if not self._object_irreducibles or not self._object_names:
            raise ValueError('The data is unknown. Fit the PatternStructure to your data using .fit(...) method')

        extent = None
        if pattern.atomisable and self._atomic_patterns is not None:
            atoms = pattern.atomise('min')
            if all(atom in self._atomic_patterns for atom in atoms):
                total_extent = ~bazeros(len(self._object_names))
                extent = reduce(fbarray.__and__, (self._atomic_patterns[atom] for atom in atoms), total_extent)
        extent = bfuncs.extension(pattern, self._object_irreducibles) if extent is None else extent

        if return_bitarray:
            return fbarray(extent)
        return {self._object_names[g] for g in extent.search(True)}

    def intent(self, objects: Union[Collection[str], fbarray]) -> PatternType:
        """
        Compute the intent of a given set of objects.

        Parameters
        ----------
        objects: Union[Collection[str], fbarray]
            The objects for which to compute the intent.

        Returns
        -------
        intent: PatternType
            The intent of the given objects.

        Raises
        ------
        ValueError
            If the data is unknown (i.e., not fitted).
        
        Examples
        --------
        >>> ps = PatternStructure()
        >>> ps.fit(data)  # Assume this fits the structure with object data
        >>> ps.intent(['obj1', 'obj2'])
        Pattern('A')
        """
        if not self._object_irreducibles or not self._object_names:
            raise ValueError('The data is unknown. Fit the PatternStructure to your data using .fit(...) method')


        if isinstance(objects, bitarray):
            objects_ba = objects
        else:  # not isinstance(objects_ba, bitarray):
            objects = set(objects)
            objects_ba = bitarray([obj in objects for obj in self._object_names])

        if self._atomic_patterns is None:
            return bfuncs.intention(objects_ba, self._object_irreducibles)

        selected_atoms = {i: atom for i, (atom, aextent) in enumerate(self._atomic_patterns.items())
                          if basubset(objects_ba, aextent)}

        if not selected_atoms:
            return self.PatternType.get_min_pattern()
        return reduce(self.PatternType.__or__, selected_atoms.values())

    ###########################################
    # Initialisation of the Pattern Structure #
    ###########################################
    def fit(
            self,
            object_descriptions: dict[str, PatternType],
            compute_atomic_patterns: bool = None, min_atom_support: Union[int, float] = 0,
            use_tqdm: bool = False
        ):
        """
        Initialize the PatternStructure with object descriptions.

        Parameters
        ----------
        object_descriptions: dict[str, PatternType]
            A dictionary mapping object names to their corresponding patterns.
        compute_atomic_patterns: bool, optional
            If True, computes atomic patterns. If None, tries to infer whether computation is possible (default is None).
        min_atom_support: Union[int, float], optional
            Minimum support threshold for an atomic pattern to be retained (default is 0).
            If a float between 0 and 1, it is interpreted as a proportion of total objects.
        use_tqdm: bool, optional
            If True, displays a progress bar when computing atomic patterns (default is True).

        Examples
        --------
        >>> ps = PatternStructure()
        >>> data = {
            "obj1": Pattern("A"),
            "obj2": Pattern("A & B"),
            "obj3": Pattern("B")
        }
        >>> ps.fit(data)
        """
        object_names, objects_patterns = zip(*object_descriptions.items())
        if self.PatternType is None:  # try to automatically identify the PatternType
            PatternTypes = {pattern.__class__ for pattern in objects_patterns if isinstance(pattern, Pattern)}
            if len(PatternTypes) > 1:
                raise ValueError(
                    f'PatternType of the PatternStructure cannot be identified as `object_descriptions` provided into '
                    f'`PatternStructure.fit()` function have multiple pattern types: {PatternTypes}. '
                    f'Convert all objects descriptions to a single pattern type and run fit() function once again.')
            self.PatternType = next(iter(PatternTypes))

        objects_patterns = list(objects_patterns)
        for i in range(len(objects_patterns)):
            if isinstance(objects_patterns[i], self.PatternType): continue

            try:
                objects_patterns[i] = self.PatternType(objects_patterns[i])
            except Exception as e:
                raise ValueError(f"Value of object '{object_names[i]}' cannot be converted to "
                                 f"PatternType '{self.PatternType}'. The raised error is: {e}.")


        object_irreducibles = bfuncs.group_objects_by_patterns(objects_patterns)

        self._object_names = list(object_names)
        self._object_irreducibles = {k: fbarray(v) for k, v in object_irreducibles.items()}

        if compute_atomic_patterns is None:
            compute_atomic_patterns = objects_patterns[0].atomisable

        if compute_atomic_patterns:
            self.init_atomic_patterns(use_tqdm=use_tqdm, min_support=min_atom_support)

    def init_atomic_patterns(self, min_support: Union[int, float] = 0, use_tqdm: bool = False):
        """
        Compute the set of all patterns that cannot be obtained by join of other patterns

        Parameters
        ----------
        min_support: Union[int, float], optional
            Minimum number or proportion of objects a pattern must describe to be considered atomic (default is 0).
        use_tqdm: bool, optional
            If True, displays a progress bar during computation (default is False).

        Raises
        ------
        AssertionError
            If the internal atomic pattern ordering check fails.

        Examples
        --------
        >>> ps = PatternStructure()
        >>> ps.fit({
            "obj1": Pattern("A"),
            "obj2": Pattern("B"),
            "obj3": Pattern("A & B")
        })
        >>> ps.init_atomic_patterns(min_support=1)
        """
        min_support = to_absolute_number(min_support, len(self._object_names))

        atomic_patterns: dict[Pattern, bitarray] = {}
        object_irreducibles_iterator = tqdm(self._object_irreducibles.items(), total=len(self._object_irreducibles),
                                            desc='Compute atomic extents', disable=not use_tqdm)
        for oi_pattern, oi_extent in object_irreducibles_iterator:
            for atomic_pattern in oi_pattern.atomise(atoms_configuration='max'):
                if atomic_pattern not in atomic_patterns:
                    atomic_patterns[atomic_pattern] = bitarray(len(oi_extent))
                atomic_patterns[atomic_pattern] |= oi_extent

        atoms_extents: list[tuple[Pattern, fbarray]] = [
            (atom, fbarray(extent)) for atom, extent in atomic_patterns.items()
            if extent.count() >= min_support
        ]

        atoms_order = bfuncs.order_patterns_via_extents(atoms_extents, use_tqdm=use_tqdm)

        def topological_key(atom_idx):
            extent: fbarray = atoms_extents[atom_idx][1]
            superatoms = atoms_order[atom_idx]

            return -extent.count(), tuple(extent.search(True)), -superatoms.count(), tuple(superatoms.search(True))
        topologic_indices = sorted(range(len(atoms_extents)), key=topological_key)
        atoms_order = bfuncs.rearrange_indices(atoms_order, atoms_extents, [atoms_extents[i] for i in topologic_indices])
        atoms_extents = [atoms_extents[i] for i in topologic_indices]

        atomic_patterns = OrderedDict(atoms_extents)
        self._atomic_patterns = atomic_patterns
        self._atomic_patterns_order = [fbarray(ba) for ba in atoms_order]

    #######################################
    # Properties that are easy to compute #
    #######################################
    @property
    def min_pattern(self) -> PatternType:
        """
        Return the minimal pattern in the structure.

        This is the least precise pattern that describes all objects.

        Returns
        -------
        min: PatternType
            The minimal pattern found in the structure.

        Raises
        ------
        ValueError
            If the data is unknown (i.e., the structure has not been fitted yet).

        Examples
        --------
        >>> ps = PatternStructure()
        >>> ps.fit({"obj1": Pattern("A"), "obj2": Pattern("B")})
        >>> ps.min_pattern
        Pattern("...")  # minimal pattern across objects
        """
        if not self._object_irreducibles:
            raise ValueError('The data is unknown. Fit the PatternStructure to your data using .fit(...) method')
        return bfuncs.minimal_pattern(self._object_irreducibles)

    @property
    def max_pattern(self) -> PatternType:
        """
        Return the maximal pattern in the structure.

        This is the most precise pattern that includes all patterns in the structure.

        Returns
        -------
        max: PatternType
            The maximal pattern found in the structure.

        Raises
        ------
        ValueError
            If the data is unknown (i.e., the structure has not been fitted yet).

        Examples
        --------
        >>> ps = PatternStructure()
        >>> ps.fit({"obj1": Pattern("A"), "obj2": Pattern("B")})
        >>> ps.max_pattern
        Pattern("A | B")
        """
        if not self._object_irreducibles:
            raise ValueError('The data is unknown. Fit the PatternStructure to your data using .fit(...) method')

        return bfuncs.maximal_pattern(self._object_irreducibles)

    @property
    def max_atoms(self) -> set[PatternType]:
        """
        Return the maximal atomic patterns in the structure.

        Maximal atoms are those atomic patterns that cannot be subsumed by any other.

        Returns
        -------
        max_atoms: set[PatternType]
            A set of maximal atomic patterns.

        Examples
        --------
        >>> ps = PatternStructure()
        >>> ps.fit({"obj1": Pattern("A"), "obj2": Pattern("A & B")})
        >>> ps.max_atoms
        {Pattern("A")}
        """
        some_pattern = next(p for p in self._object_irreducibles)
        max_atoms = some_pattern.maximal_atoms
        if max_atoms is None:
            return set()
        return max_atoms

    @property
    def atomic_patterns(self) -> OrderedDict[PatternType, set[str]]:
        """
        Return the atomic patterns in the structure.

        Returns
        -------
        atoms: OrderedDict[PatternType, set[str]]
            An ordered dictionary mapping atomic patterns to their extents (object names).

        Examples
        --------
        >>> ps.atomic_patterns
        OrderedDict({Pattern("A"): {"obj1", "obj3"}, Pattern("B"): {"obj2"}})
        """
        return OrderedDict(self.iter_atomic_patterns(return_bitarrays=False))

    @property
    def n_atomic_patterns(self) -> int:
        """
        Return the number of atomic patterns in the structure.

        Returns
        -------
        count: int
            The number of atomic patterns.

        Examples
        --------
        >>> ps.n_atomic_patterns
        5
        """
        return sum(1 for _ in self.iter_atomic_patterns(return_bitarrays=True))

    @property
    def atomic_patterns_order(self) -> dict[PatternType, set[PatternType]]:
        """
        Return the partial order of atomic patterns, i.e. for every atomic pattern show all its atomic super-patterns.

        Each pattern maps to the set of patterns that strictly subsume it.

        Returns
        -------
        order: dict[PatternType, set[PatternType]]
            A dictionary representing the ordering of atomic patterns.
            Keys are atomic patterns, values are sets of greater atomic patterns.

        Examples
        --------
        >>> ps.atomic_patterns_order
        {
            Pattern("A"): {Pattern("A | B")},
            Pattern("B"): set()
        }
        """
        if self._atomic_patterns_order is None:
            return None
        atomic_patterns_list = list(self._atomic_patterns)
        return {atomic_patterns_list[idx]: {atomic_patterns_list[v] for v in vs.search(True)}
                for idx, vs in enumerate(self._atomic_patterns_order)}

    @property
    def premaximal_patterns(self) -> dict[PatternType, set[str]]:
        """
        Return the premaximal patterns in the structure.

        Premaximal patterns are those just below the maximal pattern in precision.

        Returns
        -------
        premaximals: dict[PatternType, set[str]]
            A dictionary mapping premaximal patterns to their extents.

        Examples
        --------
        >>> ps.premaximal_patterns
        {
            Pattern("A & B"): {"obj1", "obj3"},
            Pattern("B & C"): {"obj2"}
        }
        """
        return dict(self.iter_premaximal_patterns(return_extents=True, return_bitarrays=False))

    #####################
    # Pattern Iterators #
    #####################
    def _filter_atomic_patterns_by_support(self, support_characteristic: Literal["any", "maximal", "minimal"])\
            -> tuple[OrderedDict[PatternType, fbarray], list[bitarray]]:
        if support_characteristic not in {'maximal', 'minimal', 'any'}:
            raise ValueError(f"Unsupported value for `support_characteristic`: {support_characteristic}. "
                             f"The only accepted values are 'any`, 'maximal', and 'minimal'.")

        atomic_patterns, atomic_patterns_order = self._atomic_patterns, self._atomic_patterns_order
        if support_characteristic == 'any':
            return atomic_patterns, atomic_patterns_order

        atomic_supports = [extent.count() for extent in self._atomic_patterns.values()]
        if support_characteristic == 'maximal':
            atoms_to_iterate = [i for i, superatoms in enumerate(self._atomic_patterns_order)
                                if not any(atomic_supports[i] == atomic_supports[j] for j in superatoms.search(True))]
        else:  # if support_characteristic == 'minimal':
            subatoms_order = inverse_order(self._atomic_patterns_order)
            atoms_to_iterate = [i for i, subatoms in enumerate(subatoms_order)
                                if not any(atomic_supports[i] == atomic_supports[j] for j in subatoms.search(True))]

        atomic_order = [bitarray([atomic_patterns_order[i][j] for j in atoms_to_iterate]) for i in atoms_to_iterate]
        atoms_to_iterate = set(atoms_to_iterate)
        atomic_patterns = OrderedDict([(atom, extent) for idx, (atom, extent) in enumerate(atomic_patterns.items())
                                        if idx in atoms_to_iterate])
        return atomic_patterns, atomic_order

    def iter_atomic_patterns(
        self,
        return_bitarrays: bool = False,
        kind: Literal["bruteforce", "ascending", "ascending controlled"] = 'bruteforce',
        support_characteristic: Literal["any", "maximal", "minimal"] = "any",
    ) -> Generator[tuple[PatternType, Union[set[str], fbarray]], Union[bool, None], None]:
        """
        Iterate over atomic patterns in the pattern structure.

        If `controlled` is False, then the function works as a generator that can be directly put into for-in loop,
        If `controlled` is True, then patterns' navigation can be controleld by `pattern_iterator.send()` function.
        After receiving pattern `p`,
        send `True` to the iterator in order to continue iterating over atomic patterns more precise than `p`,
        and send `False` to "early stop" the iteration.

        Important:
        When `controlled` is set to `True`,
        initialise the generator using `next(pattern_generator)` before generating any atomic patterns.


        Parameters
        ----------
        return_bitarrays: bool, optional
            If True, returns extents as bitarrays (default is False).
        kind: Literal, optional
            Iteration strategy: 'bruteforce', 'ascending', or 'ascending controlled' (default is 'bruteforce').
        support_characteristic: Literal, optional
            Defines what type of atomic patterns to output.
            If 'any' (which is the default), then output all the atomic patterns.
            If 'maximal' then output the most precise atomic patterns w.r.t. their support.
            (i.e. if atomic pattern is support-maximal, then any more precise atomic pattern would describe fewer objects).
            If 'minimal' then output the least precise atomic patterns w.r.t. their support.
            (i.e. if atomic pattern is support-minimal, then any less precise atomic pattern would describe more objects).

        Yields
        ------
        pattern_generator: Generator[tuple[PatternType, Union[set[str], fbarray]], Union[bool, None], None]
            Generator of pairs of atomic patterns and their extents (either as a set of objects' names or as a bitarray).
            When `controlled` parameter is set to `True`, one can send boolean `refine_atomic_pattern` parameter
            to the generator in order to control the patterns' navigation.

        Examples
        --------
        >>> for atom, extent in ps.iter_atomic_patterns():
            print(atom, extent)
        """
        assert self._atomic_patterns is not None, \
            "Please initialise atomic patterns first using `ps.init_atomic_patterns()` function."

        def form_yielded_value(ptrn: PatternStructure.PatternType, ext: bitarray) -> tuple[Pattern, Union[set[str], fbarray]]:
            """
            Format the yielded result based on configuration.

            Parameters
            ----------
            ptrn : PatternType
                The atomic pattern.
            ext : bitarray
                The extent of the pattern.

            Returns
            -------
            Union[PatternType, tuple[PatternType, set[str]], tuple[PatternType, fbarray]]
            """
            return ptrn, self.verbalise_extent(ext) if not return_bitarrays else ext

        if kind == 'bruteforce' and support_characteristic == 'any':
            for pattern, extent in self._atomic_patterns.items():
                yield form_yielded_value(pattern, extent)
            return

        assert self._atomic_patterns_order is not None, \
            ("Please initialise the order of atomic patterns (together with the atomic patterns themselves) "
             "using `ps.init_atomic_patterns()` function.")

        atomic_patterns, atomic_order = self._filter_atomic_patterns_by_support(support_characteristic)

        if kind == 'bruteforce':
            for pattern, extent in atomic_patterns.items():
                yield form_yielded_value(pattern, extent)

        if kind == 'ascending':
            iterator = bfuncs.iter_patterns_ascending(atomic_patterns, atomic_order, False)
            for pattern, extent in iterator:
                yield form_yielded_value(pattern, extent)

        if kind == 'ascending controlled':
            iterator = bfuncs.iter_patterns_ascending(atomic_patterns, atomic_order, True)
            next(iterator)  # initialise
            yield
            go_more_precise = True
            while True:
                try:
                    pattern, extent = iterator.send(go_more_precise)
                except StopIteration:
                    break
                go_more_precise = yield form_yielded_value(pattern, extent)

    def iter_premaximal_patterns(self, return_extents: bool = True, return_bitarrays: bool = False
        ) -> Union[
        Iterator[PatternType],
        Iterator[tuple[PatternType, set[str]]],
        Iterator[tuple[PatternType, fbarray]]]:
        """
        Iterate over premaximal patterns.

        A premaximal pattern is not strictly subsumed by any other in the irreducible set.

        Parameters
        ----------
        return_extents: bool, optional
            If True, returns extents along with patterns (default is True).
        return_bitarrays: bool, optional
            If True, returns extents as bitarrays (default is False).

        Yields
        ------
        premaximal_data: Union[PatternType, tuple[PatternType, set[str]], tuple[PatternType, fbarray]]
            Premaximal patterns with optional extent representations.

        Examples
        --------
        >>> for pattern, extent in ps.iter_premaximal_patterns():
            print(pattern, extent)
        """
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
        ) -> Union[
        Iterator[tuple[PatternType, bitarray]], 
        Generator[tuple[PatternType, bitarray], bool, None]]:
        """
        Iterate through all patterns based on selected criteria.

        Parameters
        ----------
        kind: Literal, optional
            Strategy for traversal: 'ascending' or 'ascending controlled' (default is 'ascending').
        min_support: Union[int, float], optional
            Minimum support required for a pattern to be yielded (default is 0).
        depth_first: bool, optional
            If True, performs a depth-first traversal (default is True).
        return_objects_as_bitarrays: bool, optional
            If True, extents are returned as bitarrays; otherwise as sets (default is False).

        Yields
        ------
        pattern_info: Iterator or Generator
            Yields patterns and their extents.

        Examples
        --------
        >>> for pattern, extent in ps.iter_patterns(min_support=2):
            print(pattern, extent)
        """
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
            patterns: Union[PatternType, Iterable[PatternType]] = None,
            max_length: int = None,
            min_support: Union[int, float] = 1,
            use_tqdm: bool = False
    ) -> Union[
            Iterator[PatternType], 
            Iterator[tuple[PatternType, PatternType]]]:
        """
        Iterate over the keys that describe given patterns.

        Keys (also known as minimal generators) of a pattern are the least precise subpatterns that describe the very same objects as the pattern.

        Reference
        ---------
        Buzmakov, A., Dudyrev, E., Kuznetsov, S. O., Makhalova, T., & Napoli, A. (2024). Data complexity: An FCA-based approach. International Journal of Approximate Reasoning, 165, 109084.

        Parameters
        ----------
        patterns: Union[PatternType, Iterable[PatternType]]
            A pattern or list of patterns to decompose into atomic keys.
        max_length: Optional[int], optional
            Maximum length of key combinations (default is None).
        min_support: int, default = 0
            Minimal number of objects described by a key. Only used when `patterns` is set to `None`.
        use_tqdm: bool, default = False
            Flag whether to show tqdm progress bar or not.

        Yields
        ------
        keys: Union[PatternType, tuple[PatternType, PatternType]]
            Keys of the input patterns or (key, original pattern) pairs.

        Examples
        --------
        >>> for key in ps.iter_keys(Pattern("A | B")):
        ...    print(key)
        """
        min_support = to_absolute_number(min_support, len(self._object_names))
        sup_min_atoms, sup_min_order = self._filter_atomic_patterns_by_support("minimal")
        descending_pattern_order = inverse_order(sup_min_order)

        if patterns is None:
            keys_iterator: Iterator[tuple[Pattern, fbarray]] = mec.iter_keys_via_talky_gi(
                sup_min_atoms, sup_min_order,
                min_support=min_support, yield_pattern_keys=False, max_key_length=max_length, test_subsets=True
            )
            if use_tqdm:
                keys_iterator = list(tqdm(keys_iterator, desc="Compute binarised keys", unit_scale=True))

            def key_intent_iterator(keys_iterator_) -> Iterator[tuple[Pattern, Pattern]]:
                extent_intent_map = dict()
                for key, extent in tqdm(keys_iterator_, desc="Compute pattern keys", disable=not use_tqdm, unit_scale=True):
                    pattern_key = bfuncs.patternise_description(key, list(sup_min_atoms), descending_pattern_order,
                                                                trusted_input=True)

                    if extent not in extent_intent_map:
                        extent_intent_map[extent] = self.intent(extent)
                    intent = extent_intent_map[extent]
                    yield pattern_key, intent

            return key_intent_iterator(keys_iterator)

        is_single_pattern = isinstance(patterns, self.PatternType)
        patterns = [patterns] if is_single_pattern else patterns
        patterns_extents = [(pattern, self.extent(pattern, return_bitarray=True)) for pattern in patterns]

        keys_iterator: Iterator[tuple[Pattern, int]] = mec.iter_keys_of_patterns_via_atoms(
            patterns_extents,
            sup_min_atoms, descending_pattern_order, max_length, use_tqdm=use_tqdm,
        )

        if is_single_pattern:
            return (key for key, _ in keys_iterator)
        return ((key, patterns_extents[pattern_i][0]) for key, pattern_i in keys_iterator)


    def iter_subgroups(
            self,
            goal_objects: Union[set[str], bitarray],
            quality_measure: Literal['Accuracy', 'Precision', 'Recall', 'Jaccard', 'F1', 'WRAcc'],
            quality_threshold: float,
            kind: Literal["bruteforce"] = 'bruteforce',
            max_length: Optional[int] = None,
            min_support: Optional[Union[int, float]] = 0,
            return_objects_as_bitarrays: bool = False,
            use_tqdm: bool = False,
            atomic_support_characteristic: Literal["maximal", "minimal"] = "maximal",
        ) -> Iterator[tuple[Pattern, Union[set[str], bitarray], float]]:
        """
        Iterate over subgroups that satisfy a quality threshold.

        A subgroup is a pattern whose extent is very similar to the set of goal_objects.
        The similarity is measured by quality_measure and is lower-bounded by qulity_threshold.

        References
        ----------
        Atzmueller, M. (2015). Subgroup discovery. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 5(1), 35-49.

        Parameters
        ----------
        goal_objects: Union[set[str], bitarray]
            Set or bitarray of target objects.
        quality_measure: Literal
            Metric to evaluate subgroups (e.g., 'Accuracy', 'F1', 'WRAcc').
        quality_threshold: float
            Minimum value for the selected quality measure.
        kind: Literal, optional
            Subgroup mining strategy (currently only 'bruteforce' supported).
        max_length: Optional[int], optional
            Maximum length of subgroups (default is None).
        min_support: Union[int, float], optional
            The minimal amount of objects that a subgroup should describe. Defaults to 0.
        return_objects_as_bitarrays: bool, optional
            If True, extents are returned as bitarrays (default is False).
        use_tqdm: bool, optional
            Flag whether to show tqdm progress bar to visualise the number of passed subgroup candidates.
            Defaults to False.
        atomic_support_characteristic: Literal["maximal", "minimal"], optional
            What type of atomic patterns to use to find subgroups.
            If "maximal" (the default) then prefer more precise atomic patterns (called support-maximal),
            and if "minimal" then prefer less precise atomic patterns (called support-minimal).
            For example, say there are two interval patterns "Age >= 20" and "Age > 10" that describe *the same* objects.
            Then the former pattern is support-maximal (as it is more precise) and the latter is support-minimal
            (as it is less precise).

        Yields
        ------
        subs: Iterator[tuple[Pattern, Union[set[str], bitarray], float]]
            Tuples of (pattern, extent, quality).

        Examples
        --------
        >>> for p, o, q in ps.iter_subgroups({"obj1", "obj2"}, "Precision", 0.7):
            print(p, o, q)
        """
        if not isinstance(goal_objects, bitarray):
            goal_objects = set(goal_objects)
            goal_objects = bitarray([obj_name in goal_objects for obj_name in self._object_names])

        min_support = to_absolute_number(min_support, len(self._object_names))

        quality_func, tp_min, fp_max = msubg.setup_quality_measure_function(
            quality_measure, quality_threshold, goal_objects.count(), len(goal_objects)
        )

        atomic_patterns, atomic_order = self._filter_atomic_patterns_by_support(atomic_support_characteristic)

        subgroups_iterator: Optional[Iterator[tuple[Pattern, bitarray, float]]] = None
        if kind == 'bruteforce':
            subgroups_iterator = msubg.iter_subgroups_via_atoms(
                atomic_patterns, goal_objects, quality_threshold, quality_func, max(tp_min, min_support), max_length,
                inverse_order(atomic_order),
                use_tqdm=use_tqdm
            )

        if subgroups_iterator is None:
            raise ValueError(f'Submitted kind of iterator {kind=} is not supported. '
                             f'The only supported type for the moment is "bruteforce"')

        if return_objects_as_bitarrays:
            return subgroups_iterator
        return ((pattern, self.verbalise_extent(extent), score) for pattern, extent, score in subgroups_iterator)

    ######################
    # High-level FCA API #
    ######################
    def mine_concepts(
            self,
            min_support: Union[int, float] = 0, min_delta_stability: Union[int, float] = 0,
            algorithm: Literal['CloseByOne object-wise', 'gSofia', 'CbOI'] = None,
            return_objects_as_bitarrays: bool = False,
            use_tqdm: bool = False
        ) -> Union[
            list[tuple[set[str], PatternType]], 
            list[tuple[fbarray, PatternType]]]:
        """
        Mine pattern concepts (extent-intent pairs) from the pattern structure.

        Parameters
        ----------
        min_support: Union[int, float], optional
            Minimum support required for concepts (default is 0).
        min_delta_stability: Union[int, float], optional
            Minimum delta stability for concept filtering (default is 0).
        algorithm: Literal, optional
            Algorithm used for mining: 'CloseByOne object-wise', 'gSofia', and 'CbOI'.
        return_objects_as_bitarrays: bool, optional
            If True, returns extents as bitarrays (default is False).
        use_tqdm: bool, optional
            If True, displays a progress bar (default is False).

        Returns
        -------
        concepts: Union[list[tuple[set[str], PatternType]], list[tuple[fbarray, PatternType]]]
            A list of concept tuples, each containing an extent and its corresponding intent.
        
        Examples
        --------
        >>> ps.mine_concepts(min_support=1)
        [({"obj1", "obj2"}, Pattern("A"))]
        """
        SUPPORTED_ALGOS = {'CloseByOne object-wise', 'gSofia', 'CbOI'}
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
            atomic_patterns_iterator = self.iter_atomic_patterns(return_bitarrays=True, kind='ascending controlled',
                                                                 support_characteristic='maximal')
            n_atomic_patterns = len(self._filter_atomic_patterns_by_support('maximal')[0]) if self._atomic_patterns_order is not None else None
            extents_ba = mec.list_stable_extents_via_gsofia(
                atomic_patterns_iterator,
                min_delta_stability=min_delta_stability,
                min_supp=min_support,
                use_tqdm=use_tqdm,
                n_atomic_patterns=n_atomic_patterns
            )
            extents_intents_dict: dict[fbarray, Pattern] = dict()
            for extent in tqdm(extents_ba, disable=not use_tqdm, desc='Compute intents'):
                extents_intents_dict[fbarray(extent)] = self.intent(extent)

        if algorithm == 'CbOI':
            unused_parameters = []
            if min_delta_stability > 0:
                unused_parameters.append(f"{min_delta_stability=}")
            if unused_parameters:
                warnings.warn(UserWarning(
                    f'The following parameters {", ".join(unused_parameters)} do not affect algorithm {algorithm}'))

            sup_max_patterns, sup_max_order = self._filter_atomic_patterns_by_support('maximal')
            sup_max_subpattern_order = inverse_order(sup_max_order)

            concepts_generator = mec.iter_intents_via_cboi(
                sup_max_patterns, sup_max_order, min_support=min_support, yield_pattern_intents=False
            )
            concepts_generator = tqdm(concepts_generator, disable=not use_tqdm, desc='Compute binarised concepts')
            if use_tqdm:
                concepts_generator = list(concepts_generator)

            pattern_intent = partial(
                bfuncs.patternise_description,
                atomic_patterns=list(sup_max_patterns), subatoms_order=sup_max_subpattern_order, trusted_input=True)
            concepts_generator = ((pattern_intent(intent), extent) for intent, extent in
                                  tqdm(concepts_generator, disable=not use_tqdm, desc='Compute pattern concepts'))

            extents_intents_dict: dict[fbarray, Pattern] = {extent: intent  for intent, extent in concepts_generator}

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
            algorithm: Literal['CloseByOne object-wise', 'gSofia', 'Talky-GI'] = None,
            reduce_conclusions: bool = False,
            use_tqdm: bool = False,
        ) -> dict[PatternType, PatternType]:
        """
        Mine implications from the pattern structure using a selected basis.

        Parameters
        ----------
        basis_name: Literal, optional
            Type of basis used for implications (default is "Canonical Direct").
        min_support: Union[int, float], optional
            Minimum support for implications (default is 0).
        min_delta_stability: Union[int, float], optional
            Minimum delta stability (default is 0).
        max_key_length: Optional[int], optional
            Maximum length of keys (default is None).
        algorithm: Literal, optional
            Concept mining algorithm to use (default is None).
        reduce_conclusions: bool, optional
            If True, reduces the size of implication conclusions (default is False).
        use_tqdm: bool, optional
            If True, displays a progress bar (default is False).

        Returns
        -------
        implications: dict[PatternType, PatternType]
            A dictionary mapping premises to conclusions in the implication basis.
        
        Examples
        --------
        >>> ps.mine_implications(min_support=2)
        {Pattern("A"): Pattern("B")}
        """
        if algorithm == 'Talky-GI':
            supmin_atoms, supmin_order = self._filter_atomic_patterns_by_support('minimal')
            supmax_atoms, supmax_order = self._filter_atomic_patterns_by_support('maximal')
            supmin_order_dim, supmax_order_dim = inverse_order(supmin_order), inverse_order(supmax_order)

            keys_extent_iterator = mec.iter_keys_via_talky_gi(
                supmin_atoms, supmin_order,
                min_support=min_support, yield_pattern_keys=False, max_key_length=max_key_length
            )
            keys_extent_iterator = tqdm(keys_extent_iterator, disable=not use_tqdm, desc='Iter premise candidates')
            ppremise_iterator = mib.iter_proper_premises_from_atomised_premises(
                keys_extent_iterator,
                supmin_atoms, supmin_order_dim, supmax_atoms, supmax_order_dim,
                yield_patterns=False, reduce_conclusions=reduce_conclusions
            )
            if use_tqdm:
                ppremise_iterator = list(tqdm(ppremise_iterator, desc='Iter atomised premises'))

            supmin_atoms, supmax_atoms = list(supmin_atoms), list(supmax_atoms)

            if basis_name == 'Canonical Direct':
                pattern_premise_iterator = (
                    (bfuncs.patternise_description(premise, supmin_atoms, supmin_order_dim),
                     bfuncs.patternise_description(conclusion, supmax_atoms, supmax_order_dim))
                    for premise, conclusion in ppremise_iterator
                )
                pattern_premise_iterator = tqdm(pattern_premise_iterator, disable=not use_tqdm, desc='Iter pattern premises')
                return OrderedDict(list(pattern_premise_iterator))

            # basis_name == 'Canonical'
            atoms, superatoms_order = self._filter_atomic_patterns_by_support('any')
            subatoms_order = inverse_order(superatoms_order)

            atom_to_idx = {atom: i for i, atom in enumerate(atoms)}
            n_atoms = len(atoms)
            premises: list[bitarray] = []
            for premise_minsup, _ in ppremise_iterator:
                premise = bazeros(n_atoms)
                for i_minsup in premise_minsup.search(True): premise[atom_to_idx[supmin_atoms[i_minsup]]] = True
                premises.append(premise)

            pseudo_intents = mib.iter_pseudo_intents_from_atomised_premises(
                premises, atoms, subatoms_order, yield_patterns=False, reduce_conclusions=reduce_conclusions)
            atoms = list(atoms)
            if use_tqdm:
                pseudo_intents = list(tqdm(pseudo_intents, desc='Iter atomised p.intents'))
            pattern_pintents_iterator = (
                (bfuncs.patternise_description(premise, atoms, subatoms_order),
                 bfuncs.patternise_description(conclusion, atoms, subatoms_order))
                for premise, conclusion in pseudo_intents
            )
            return dict(tqdm(pattern_pintents_iterator, desc='Iter pattern p.intents', disable=not use_tqdm))

        else:
            concepts: list[tuple[fbarray, PatternStructure.PatternType]] = self.mine_concepts(
                min_support=min_support, min_delta_stability=min_delta_stability,
                algorithm=algorithm, return_objects_as_bitarrays=True, use_tqdm=use_tqdm
            )
            intents = map(itemgetter(1), concepts)
            if not concepts[0][0].all():
                intents = [self.intent(concepts[0][0]|~concepts[0][0])] + list(intents)

            keys: Iterator[Pattern] = map(itemgetter(0), self.iter_keys(intents, max_length=max_key_length))
        keys: list[Pattern] = list(tqdm(keys, desc="Mine premise candidates", disable=not use_tqdm))

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
        ) -> Union[
            dict[PatternType, PatternType], 
            OrderedDict[PatternType, PatternType]]:
        """
        Construct implications from a set of given premises.

        Parameters
        ----------
        premises: Iterable[PatternType]
            The premises to base implications on.
        pseudo_close_premises: bool, optional
            Whether to pseudo-close the premises (default is False).
        reduce_conclusions: bool, optional
            If True, reduces conclusions to minimal additions (default is False).
        use_tqdm: bool, optional
            If True, displays progress bars (default is False).

        Returns
        -------
        premises: Union[dict[PatternType, PatternType], OrderedDict[PatternType, PatternType]]
            A mapping from premises to conclusions.
        
        Examples
        --------
        >>> premises = [Pattern("A")]
        >>> ps.mine_implications_from_premises(premises)
        {Pattern("A"): Pattern("B")}
        """
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
        ) -> Union[
            set[PatternType],
            dict[PatternType, set[str]], 
            dict[PatternType, fbarray]]:
        """
        Compute the immediate successor patterns of a given pattern.

        The immediate successor patternsare the next more precise patterns.

        Parameters
        ----------
        pattern: PatternType
            The pattern to compute successors for.
        return_extents: bool, optional
            If True, returns extents along with patterns (default is False).
        return_objects_as_bitarrays: bool, optional
            If True, extents are returned as bitarrays instead of sets (default is False).

        Returns
        -------
        next_patterns: Union[set[PatternType], dict[PatternType, set[str]], dict[PatternType, fbarray]]
            Either a set of successor patterns or a mapping of successors to extents.

        Examples
        --------
        >>> ps.next_patterns(Pattern("A"))
        {Pattern("A & B")}
        """
        atom_iterator = self.iter_atomic_patterns(return_bitarrays=True, kind='ascending controlled')
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
        """
        Measure the support (number of objects) of a given pattern.

        Support is the size of the pattern's extent, i.e., the number of objects that satisfy the pattern.

        Parameters
        ----------
        pattern: Pattern
            The pattern for which to compute support.

        Returns
        -------
        support: int
            The number of objects (support count) covered by the pattern.

        Examples
        --------
        >>> ps.measure_support(Pattern("A"))
        3
        """
        return self.extent(pattern, return_bitarray=True).count()

    def measure_frequency(self, pattern: Pattern) -> float:
        """
        Measure the frequency of a given pattern.

        Frequency is the proportion of objects satisfying the pattern.

        Parameters
        ----------
        pattern: Pattern
            The pattern for which to compute frequency.

        Returns
        -------
        frequency: float
            The frequency of the pattern as a fraction between 0 and 1.

        Examples
        --------
        >>> ps.measure_frequency(Pattern("A"))
        0.6
        """
        extent = self.extent(pattern, return_bitarray=True)
        return extent.count()/len(extent)

    def measure_delta_stability(self, pattern: Pattern) -> int:
        """
        Measure the delta stability of a given pattern.

        Delta stability is defined as the difference in support between the pattern and
        its most specific generalization (i.e., next pattern with largest shared extent).

        Parameters
        ----------
        pattern: Pattern
            The pattern for which to compute delta stability.

        Returns
        -------
        delta_s: int
            The delta stability value.

        Examples
        --------
        >>> ps.measure_delta_stability(Pattern("A"))
        2  # pattern support is 5, next largest overlapping pattern support is 3
        """
        extent = self.extent(pattern, return_bitarray=True)

        subextents = self.next_patterns(pattern, return_extents=True, return_objects_as_bitarrays=True).values()
        return extent.count() - max((subext.count() for subext in subextents), default=0)

    #####################
    # Helping functions #
    #####################
    def verbalise_extent(self, extent: Union[bitarray, set[str]]) -> set[str]:
        """
        Convert an extent to a human-readable format (set of object names).

        This function takes an extent represented as a bitarray (internal format) or a set of object names,
        and returns the corresponding object names in a readable form.

        Parameters
        ----------
        extent: Union[bitarray, set[str]]
            The extent to convert. If a bitarray is provided, each set bit is mapped to its corresponding object name.

        Returns
        -------
        readable: set[str]
            The human-readable extent as a set of object names.

        Examples
        --------
        >>> ba = bitarray('1010')
        >>> ps._object_names = ["obj1", "obj2", "obj3", "obj4"]
        >>> ps.verbalise_extent(ba)
        {'obj1', 'obj3'}

        >>> ps.verbalise_extent({'obj2', 'obj4'})
        {'obj2', 'obj4'}
        """
        if not isinstance(extent, bitarray):
            return extent
        return {self._object_names[g] for g in extent.search(True)}

    @property
    def objects(self) -> set[str]:
        """Return the names of objects in the Pattern Structure"""
        return set(self._object_names)
