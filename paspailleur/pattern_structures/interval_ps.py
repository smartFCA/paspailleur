from enum import Flag
from numbers import Number
from typing import Iterator, Optional, Union, Iterable, Sequence
from bitarray import frozenbitarray as fbarray
from .abstract_ps import AbstractPS
from math import inf, ceil


class BoundStatus(Flag):
    OPEN = 0  # binary 00 meaning open-open
    RCLOSED = 1  # binary 01 meaning open-closed
    LCLOSED = 2  # binary 10 meaning closed-open
    CLOSED = 3  # binary 11 meaning closed-closed


class IntervalPS(AbstractPS):
    ndigits: int  # Number of digits after the comma
    PatternType = tuple[float, float, BoundStatus]
    min_pattern = (-inf, inf, BoundStatus.OPEN)  # The pattern that always describes all objects
    max_pattern = (None, None, BoundStatus.CLOSED)  # The pattern that always describes no objects
    min_bounds: Optional[tuple[float]] = None  # Left bounds of intervals in the data. Sorted in ascending order
    max_bounds: Optional[tuple[float]] = None  # Right bounds of intervals in the data. Sorted in ascending order
    only_closed_flg: bool = False  # Use only closed intervals

    def __init__(
            self, ndigits: int = 2,
            values: list[float] = None, min_bounds: list[float] = None, max_bounds: list[float] = None,
            only_closed_flag: bool = False
    ):
        self.ndigits = ndigits

        if values is not None and not (min_bounds is None and max_bounds is None):
            raise ValueError('Please specify either `values` or the pair `min_bounds`, `max_bounds`,'
                             ' but not all the parameters at the same time')
        if (min_bounds is None) != (max_bounds is None):
            raise ValueError('Please specify both `min_bounds` and `max_bounds` parameters.'
                             ' If the bounds are the same, you can use `values` parameter of the __init__ function')

        if values is not None:
            min_bounds = max_bounds = tuple(values)
        self.min_bounds = tuple(sorted({round(x, ndigits) for x in min_bounds})) if min_bounds else None
        self.max_bounds = tuple(sorted({round(x, ndigits) for x in max_bounds})) if max_bounds else None

        self.only_closed_flg = only_closed_flag

    @property
    def precision(self) -> float:
        return 10**(-self.ndigits)

    def join_patterns(self, a: PatternType, b: PatternType) -> PatternType:
        """Return the most precise common pattern, describing both patterns `a` and `b`"""
        if a == self.max_pattern:
            return b
        if b == self.max_pattern:
            return a

        l, r = min(a[0], b[0]), max(a[1], b[1])
        if self.only_closed_flg:
            return l, r, BoundStatus.CLOSED
        lbound = BoundStatus.LCLOSED & (a[2] if a[0] < b[0] else b[2] if b[0] < a[0] else a[2]|b[2])
        rbound = BoundStatus.RCLOSED & (b[2] if a[1] < b[1] else a[2] if b[1] < a[1] else a[2]|b[2])
        return l, r, lbound | rbound

    def is_less_precise(self, a: PatternType, b: PatternType) -> bool:
        """Return True if pattern `a` is less precise than pattern `b`"""
        if b == self.max_pattern:
            return True
        if a == self.max_pattern:  # and b != max_pattern
            return False

        if not (a[0] <= b[0]):
            return False
        if not (b[1] <= a[1]):
            return False
        if self.only_closed_flg:
            return True

        if a[0] == b[0] and BoundStatus.LCLOSED not in a[2] and BoundStatus.LCLOSED in b[2]:
            return False
        if a[1] == b[1] and BoundStatus.RCLOSED not in a[2] and BoundStatus.RCLOSED in b[2]:
            return False

        return True

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
        # TODO: Add support for open and half-open intervals
        min_support = ceil(len(data) * min_support) if 0 < min_support < 1 else int(min_support)

        lower_bounds = sorted({lb for lb, _, _ in data})
        upper_bounds = sorted({ub for _, ub, _ in data})
        min_, max_ = lower_bounds.pop(0), upper_bounds.pop(-1)

        yield (min_, max_, BoundStatus.CLOSED), fbarray([True]*len(data))

        for lb in lower_bounds:
            extent = fbarray((lb <= x for x, _, _ in data))
            if extent.count() < min_support:
                break
            yield (lb, max_, BoundStatus.CLOSED), extent

        for ub in upper_bounds[::-1]:
            extent = fbarray((x <= ub for _, x, _ in data))
            if extent.count() < min_support:
                break
            yield (min_, ub, BoundStatus.CLOSED), extent

        if min_support == 0:
            yield self.max_pattern, fbarray([False]*len(data))

    def n_attributes(self, data: list[PatternType], min_support: Union[int, float] = 0, use_tqdm: bool = False)\
            -> int:
        """Count the number of attributes in the binary representation of `data`"""
        if min_support == 0:
            return len({lb for lb, _, _ in data}) + len({ub for _, ub, _ in data})
        return super().n_attributes(data, min_support)

    def preprocess_data(self, data: Iterable[Union[Number, Sequence[Number]]]) -> Iterator[PatternType]:
        """Preprocess the data into to the format, supported by attrs_order/extent functions"""
        lbounds, rbounds = [], []
        for descr in data:
            if isinstance(descr, Number):
                descr = (descr, descr)
            if isinstance(descr, range):
                start, stop = descr.start, descr.stop
                if start < stop:
                    descr = (start, stop-1)
                elif stop < start:
                    descr = (stop+1, start)
                else:  # if start == stop, then there is not closed interval inside [start, stop) == [start, start)
                    descr = self.max_pattern

            if isinstance(descr, Sequence)\
                    and len(descr) == 2 and all(isinstance(x, Number) for x in descr):
                descr = (descr[0], descr[1], BoundStatus.CLOSED)
            if isinstance(descr, Sequence) and len(descr) == 3 \
                    and all(isinstance(x, Number) for x in descr[:2]) and isinstance(descr[2], BoundStatus):
                pass
            else:
                raise ValueError(f'Cannot preprocess this description: {descr}. '
                                 f'Provide either a number, or a sequence of two numbers,'
                                 f' or a sequence of two numbers + border status'
                                 f' (0 for open, 1 for right-closed, 2 for left-closed, 3 for closed interval).')

            descr = (float(round(descr[0], self.ndigits)), float(round(descr[1], self.ndigits)), descr[2])
            yield descr
            lbounds.append(descr[0])
            rbounds.append(descr[1])

        self.min_bounds = tuple(sorted(set(lbounds)))
        self.max_bounds = tuple(sorted(set(rbounds)))

    def verbalize(self, description: PatternType, number_format: str = '.2f') -> str:
        """Convert `description` into human-readable string"""
        if tuple(description) == self.max_pattern:
            return '∅'
        lb = '[' if BoundStatus.LCLOSED in description[2] else '('
        ub = ']' if BoundStatus.RCLOSED in description[2] else ')'

        l = f"{description[0]:{number_format}}" if description[0] > -inf else '-∞'
        u = f"{description[1]:{number_format}}" if description[1] < inf else '∞'
        return f"{lb}{l}, {u}{ub}"

    def closest_less_precise(
            self,
            description: PatternType,
            use_lectic_order: bool = False, use_data_values: bool = True
    ) -> Iterator[PatternType]:
        """Return closest descriptions that are less precise than `description`

        Use lectic order for optimisation of description traversal.
        If use_data_values=True, then the function only returns the closest descriptions that can be found in the data
        (in case the values from the data are provided).
        """
        if description == self.min_pattern:
            return iter([])

        l, r, bound = description

        next_right = r
        if BoundStatus.RCLOSED in bound:  # find next bigger value and make the bound open
            if use_data_values and self.max_bounds:
                next_right = next(x for x in self.max_bounds if x > r) if r < self.max_bounds[-1] else self.min_pattern[1]
            else:
                next_right += self.precision
        next_right_bound = (~bound) & BoundStatus.RCLOSED if not self.only_closed_flg else BoundStatus.CLOSED

        next_right_descr = l, round(next_right, self.ndigits), (bound & BoundStatus.LCLOSED) | next_right_bound
        if use_lectic_order:
            return iter([next_right_descr])

        next_left = l
        if BoundStatus.LCLOSED in bound:  # find next smaller value and make the bound open
            if use_data_values and self.min_bounds:
                next_left = next(x for x in self.min_bounds[::-1] if x < l)\
                    if self.min_bounds[0] < l else self.min_pattern[0]
            else:
                next_left -= self.precision
        next_left_bound = (~bound) & BoundStatus.LCLOSED if not self.only_closed_flg else BoundStatus.CLOSED

        next_left_descr = round(next_left, self.ndigits), r, (bound & BoundStatus.RCLOSED) | next_left_bound
        return iter([next_right_descr, next_left_descr])

    def closest_more_precise(
            self,
            description: PatternType,
            intent: PatternType = None,
            use_lectic_order: bool = False, use_data_values: bool = True
    ) -> Iterator[PatternType]:
        """Return closest descriptions that are more precise than `description`

        Use lectic order for optimisation of description traversal
        If use_data_values=True, then the function only returns the closest descriptions that can be found in the data
        (in case the values from the data are provided).
        """
        if description == self.max_pattern:
            return iter([])

        l, r, bound = description
        if r == l:
            return iter([self.max_pattern])

        intent = description if intent is None else intent

        next_right = r
        if BoundStatus.RCLOSED not in bound:  # if right bound is open, find next smaller value
            if use_data_values and self.max_bounds:
                next_right = next(x for x in self.max_bounds[::-1] if x < r)
            else:
                next_right -= self.precision
        next_right_bound = (~bound) & BoundStatus.RCLOSED if not self.only_closed_flg else BoundStatus.CLOSED

        next_right_descr = l, round(next_right, self.ndigits), (bound & BoundStatus.LCLOSED) | next_right_bound
        if use_lectic_order and description[1] < intent[1]:
            return iter([next_right_descr])

        next_left = l
        if BoundStatus.LCLOSED not in bound:
            if use_data_values and self.min_bounds:
                next_left = next(x for x in self.min_bounds if x > l)
            else:
                next_left += self.precision
        next_left_bound = (~bound) & BoundStatus.LCLOSED if not self.only_closed_flg else BoundStatus.CLOSED

        next_left_descr = round(next_left, self.ndigits), r, (bound & BoundStatus.RCLOSED) | next_left_bound
        return iter([next_right_descr, next_left_descr])

    def keys(self, intent: PatternType, data: list[PatternType]) -> list[PatternType]:
        """Return the least precise descriptions equivalent to the given attrs_order"""
        assert intent[2] == BoundStatus.CLOSED, 'Only closed descriptions can be intents'
        if self.only_closed_flg:
            out_l = intent[0] if any(l < intent[0] for l, _, _ in data) else self.min_pattern[0]
            out_r = intent[1] if any(intent[1] < r for _, r, _ in data) else self.min_pattern[1]
            return [(out_l, out_r, BoundStatus.OPEN)]
        out_l = max((l for l, _, _ in data if l < intent[0]), default=self.min_pattern[0])
        out_r = min((r for _, r, _ in data if r > intent[1]), default=self.min_pattern[1])
        return [(out_l, out_r, BoundStatus.OPEN)]
