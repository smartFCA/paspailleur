import math
from numbers import Number
from typing import Iterator, Optional, Union, Iterable, Sequence
from bitarray import frozenbitarray as fbarray
from .abstract_ps import AbstractPS
from math import inf, ceil


class IntervalPS(AbstractPS):
    ndigits: int  # Number of digits after the comma
    precision: float  # Precision of the number in pattern
    PatternType = Optional[tuple[float, float]]
    min_pattern = (-inf, inf)  # The pattern that always describes all objects
    max_pattern = (inf, -inf)  # The pattern that always describes no objects

    def __init__(self, ndigits: int = 6):
        self.ndigits = ndigits
        self.precision = 10**(-ndigits)

    def join_patterns(self, a: PatternType, b: PatternType) -> PatternType:
        """Return the most precise common pattern, describing both patterns `a` and `b`"""
        return min(a[0], b[0]), max(a[1], b[1])

    def is_less_precise(self, a: PatternType, b: PatternType) -> bool:
        """Return True if pattern `a` is less precise than pattern `b`"""
        if b == self.max_pattern:
            return True
        if a == self.max_pattern:  # and b != max_pattern
            return False

        return a[0] <= b[0] <= b[1] <= a[1]

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

        lower_bounds, upper_bounds = [sorted(set(bounds)) for bounds in zip(*data)]
        min_, max_ = lower_bounds[0], upper_bounds[-1]
        lower_bounds.pop(0)
        upper_bounds.pop(-1)

        yield (min_, max_), fbarray([True]*len(data))

        for lb in lower_bounds:
            extent = fbarray((lb <= x for x, _ in data))
            if extent.count() < min_support:
                break
            yield (lb, max_), extent

        for ub in upper_bounds[::-1]:
            extent = fbarray((x <= ub for _, x in data))
            if extent.count() < min_support:
                break
            yield (min_, ub), extent

        if min_support == 0:
            yield self.max_pattern, fbarray([False]*len(data))

    def n_attributes(self, data: list[PatternType], min_support: Union[int, float] = 0, use_tqdm: bool = False)\
            -> int:
        """Count the number of attributes in the binary representation of `data`"""
        if min_support == 0:
            return len({lb for lb, ub in data}) + len({ub for ub in data})
        return super().n_attributes(data, min_support)

    def preprocess_data(self, data: Iterable[Union[Number, Sequence[Number]]]) -> Iterator[PatternType]:
        """Preprocess the data into to the format, supported by attrs_order/extent functions"""
        for description in data:
            if isinstance(description, Number):
                description = (description, description)
            if isinstance(description, range):
                start, stop = description.start, description.stop
                if start < stop:
                    description = (start, stop-1)
                elif stop < start:
                    description = (stop+1, start)
                else:  # if start == stop, then there is not closed interval inside [start, stop) == [start, start)
                    description = (inf, -inf)

            if isinstance(description, Sequence)\
                    and len(description) == 2 and all(isinstance(x, Number) for x in description):
                description = (float(description[0]), float(description[1]))
            else:
                raise ValueError(f'Cannot preprocess this description: {description}. '
                                 f'Provide either a number or a sequence of two numbers.')

            description = tuple([round(x, self.ndigits) if x != inf and x != -inf else x for x in description])
            yield description

    def verbalize(self, description: PatternType, number_format: str = '.2f') -> str:
        """Convert `description` into human-readable string"""
        if tuple(description) == self.max_pattern:
            return '∅'
        if description == (-inf, inf):
            return '[-∞, ∞]'
        if description[0] == -inf:
            return f'<= {description[1]:{number_format}}'
        if description[1] == inf:
            return f'>= {description[0]:{number_format}}'
        return f'[{description[0]:{number_format}}, {description[1]:{number_format}}]'

    def closest_less_precise(self, description: PatternType, use_lectic_order: bool = False) -> Iterator[PatternType]:
        """Return closest descriptions that are less precise than `description`

        Use lectic order for optimisation of description traversal
        """
        if description == self.min_pattern:
            return

        l, r = description

        yield l, r+self.precision
        if not use_lectic_order:
            yield l-self.precision, r

    def closest_more_precise(self, description: PatternType, use_lectic_order: bool = False) -> Iterator[PatternType]:
        """Return closest descriptions that are more precise than `description`

        Use lectic order for optimisation of description traversal
        """
        if description == self.max_pattern:
            return

        l, r = description
        if r - l <= self.precision:
            yield self.max_pattern
            return

        yield l, r-self.precision
        if not use_lectic_order:
            yield l+self.precision, r

    def keys(self, intent: PatternType, data: list[PatternType]) -> list[PatternType]:
        """Return the least precise descriptions equivalent to the given attrs_order"""
        out_l = max((l + self.precision for l, _ in data if l < intent[0]), default=self.min_pattern[0])
        out_r = min((r - self.precision for _, r in data if r > intent[1]), default=self.min_pattern[1])
        return [(out_l, out_r)]
