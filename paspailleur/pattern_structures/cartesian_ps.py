from collections import deque
from functools import reduce
import itertools as itools
from typing import Iterator, Union, Iterable, Any, Sequence
from bitarray import frozenbitarray as fbarray
from caspailleur.base_functions import isets2bas
from .abstract_ps import AbstractPS

from tqdm.autonotebook import tqdm


class CartesianPS(AbstractPS):
    PatternType = tuple[tuple, ...]
    min_pattern: tuple  # Top pattern, less specific than any other one
    max_pattern: tuple  # Bottom pattern, more specific than any other one
    basic_structures: tuple[AbstractPS, ...]

    def __init__(self, basic_structures: list[AbstractPS]):
        self.basic_structures = tuple(basic_structures)
        self.min_pattern = tuple([ps.min_pattern for ps in basic_structures])
        self.max_pattern = tuple([ps.max_pattern for ps in basic_structures])

    def join_patterns(self, a: PatternType, b: PatternType) -> PatternType:
        """Return the most precise common pattern, describing both patterns `a` and `b`"""
        return tuple([ps.join_patterns(a_, b_) for (ps, a_, b_) in zip(self.basic_structures, a, b)])

    def is_less_precise(self, a: PatternType, b: PatternType) -> bool:
        """Return True if pattern `a` is less precise than pattern `b`"""
        return all(ps.is_less_precise(a_, b_) for ps, a_, b_ in zip(self.basic_structures, a, b))

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
        for i, ps in enumerate(self.basic_structures):
            ps_data = [data_row[i] for data_row in data]
            for pattern, flag in ps.iter_attributes(ps_data, min_support):
                yield (i, pattern), flag

    def n_attributes(self, data: list[PatternType], min_support: Union[int, float] = 0, use_tqdm: bool = False)\
            -> int:
        """Count the number of attributes in the binary representation of `data`"""
        n_bin_attrs = 0
        iterator = enumerate(self.basic_structures)
        if use_tqdm:
            iterator = tqdm(iterator, desc='Iterating basic structures', total=len(self.basic_structures))
        for i, ps in iterator:
            ps_data = [data_row[i] for data_row in data]
            n_bin_attrs += ps.n_attributes(ps_data, min_support=min_support, use_tqdm=use_tqdm)
        return n_bin_attrs

    def preprocess_data(self, data: Iterable[Sequence[Any]]) -> Iterator[PatternType]:
        """Preprocess the data into to the format, supported by attrs_order/extent functions"""
        vals_per_structures = [[] for _ in self.basic_structures]
        for description in data:
            for column, value in enumerate(description):
                vals_per_structures[column].append(value)

        processed_per_structures = [list(bs.preprocess_data(vals))
                                    for bs, vals in zip(self.basic_structures, vals_per_structures)]
        return zip(*processed_per_structures)

    def verbalize(
        self, description: PatternType,
        separator=', ', pattern_names: list[str] = None,
        basic_structures_params: dict[int, dict[str, Any]] = None
    ) -> str:
        """Convert `description` into human-readable string"""
        if pattern_names is None:
            pattern_names = [f"{i}" for i in range(len(self.basic_structures))]

        if basic_structures_params is None:
            basic_structures_params = dict()

        basic_strs = [f"{pattern_names[i]}: {bps.verbalize(v, **basic_structures_params.get(i, {}))}"
                      for i, (v, bps) in enumerate(zip(description, self.basic_structures))]
        return separator.join(basic_strs)

    def closest_less_precise(self, description: PatternType, use_lectic_order: bool = False) -> Iterator[PatternType]:
        """Return closest descriptions that are less precise than `description`

        Use lectic order for optimisation of description traversal
        """
        if description == self.min_pattern:
            return iter([])

        for i, bs in enumerate(self.basic_structures):
            for next_coord in bs.closest_less_precise(description[i], use_lectic_order=use_lectic_order):
                next_description = list(description)
                next_description[i] = next_coord
                yield tuple(next_description)

    def closest_more_precise(self, description: PatternType, use_lectic_order: bool = False) -> Iterator[PatternType]:
        """Return closest descriptions that are more precise than `description`

        Use lectic order for optimisation of description traversal
        """
        if description == self.max_pattern:
            return iter([])

        for i, bs in enumerate(self.basic_structures):
            for next_coord in bs.closest_more_precise(description[i], use_lectic_order=use_lectic_order):
                next_description = list(description)
                next_description[i] = next_coord
                yield tuple(next_description)

#    def keys(self, intent: PatternType, data: list[PatternType]) -> list[PatternType]:
#        """Return the least precise descriptions equivalent to the given attrs_order"""
#        pass

    def passkeys(self, intent: PatternType, data: list[PatternType]) -> list[PatternType]:
        n_objs, n_attrs = len(data), len(self.basic_structures)
        extent_final = next(isets2bas([self.extent(data, intent)], n_objs))
        if extent_final.all():
            return [self.min_pattern]
        total_extent = extent_final | (~extent_final)

        data_per_structure = [[] for _ in range(n_attrs)]
        for item in data:
            for j, v in enumerate(item):
                data_per_structure[j].append(v)

        extents_per_structure: list[fbarray] = [
            next(isets2bas([bs.extent(values, coord)], n_objs))
            for (bs, coord, values) in zip(self.basic_structures, intent, data_per_structure)
        ]

        protokeys_per_structure = []
        for bs_i, (bs, coord, values) in enumerate(zip(self.basic_structures, intent, data_per_structure)):
            base_extent = reduce(fbarray.__and__, map(extents_per_structure.__getitem__, set(range(n_attrs))-{bs_i}), total_extent)
            protokey = bs.keys(coord, [values[i] for i in base_extent.search(True)])
            protokeys_per_structure.append(tuple(protokey))

        keys = []
        for n_coords in range(1, n_attrs+1):
            for combination in itools.combinations(range(n_attrs), n_coords):
                extent = reduce(fbarray.__and__, map(extents_per_structure.__getitem__, combination), total_extent)
                if extent != extent_final:
                    continue

                subdata = [tuple([row[j] for j in combination]) for i, row in enumerate(data)]
                subintent = tuple([intent[j] for j in combination])
                subps = CartesianPS([self.basic_structures[j] for j in combination])

                protokeys = itools.product(*map(protokeys_per_structure.__getitem__, combination))
                key_candidates, new_keys, visited = deque(protokeys), [], set()
                while key_candidates:
                    candidate = key_candidates.popleft()
                    if candidate in visited:
                        continue
                    visited.add(candidate)

                    extent = next(isets2bas([subps.extent(subdata, candidate)], n_objs))
                    if extent == extent_final:
                        new_keys.append(candidate)
                        continue
                    next_candidates = [
                        next_candidate
                        for next_candidate in subps.closest_more_precise(candidate, use_lectic_order=False)
                        if subps.is_less_precise(next_candidate, subintent) or next_candidate == subintent
                    ]
                    key_candidates.extend(next_candidates)

                while True:
                    for new_key in list(new_keys):
                        preciser_keys = {other_key for other_key in new_keys
                                         if new_key != other_key and subps.is_less_precise(new_key, other_key)}
                        if preciser_keys:
                            new_keys = [key for key in new_keys if key not in preciser_keys]
                            break
                    else:  # no break => no preciser keys found for any key
                        break  # the list of keys is maximal

                for new_key in new_keys:
                    descr = list(self.min_pattern)
                    for j, v in zip(combination, new_key):
                        descr[j] = v
                    keys.append(tuple(descr))

            if keys:
                break
        return keys
