from typing import Self, Union

from .pattern import Pattern


class ItemSetPattern(Pattern):
    PatternValueType = set

    def __init__(self, value):
        super().__init__(frozenset(value))

    @property
    def value(self) -> PatternValueType:
        return set(self._value)

    def __repr__(self) -> str:
        return f"ItemSetPattern({self.value})"


class IntervalPattern(Pattern):
    # PatternValue semantics: ((lower_bound, is_closed), (upper_bound, is_closed))
    PatternValueType = tuple[tuple[float, bool], tuple[float, bool]]

    def __init__(self, value: Union[PatternValueType, str]):
        super().__init__(None)
        if isinstance(value, str):
            lb, ub = map(str.strip, value[1:-1].replace('∞', 'inf').split(','))
            closed_lb, closed_ub = value[0] == '[', value[-1] == ']'
        else:
            lb, closed_lb = value[0]
            ub, closed_ub = value[1]
        self._lower_bound = float(lb)
        self._upper_bound = float(ub)
        self._is_closed_lower_bound = bool(closed_lb)
        self._is_closed_upper_bound = bool(closed_ub)

    @property
    def value(self) -> PatternValueType:
        return (self._lower_bound, self._is_closed_lower_bound), (self._upper_bound, self._is_closed_upper_bound)

    def __repr__(self) -> str:
        lbound_sign = '[' if self._is_closed_lower_bound else '('
        ubound_sign = ']' if self._is_closed_upper_bound else ')'
        return f"IntervalPattern({lbound_sign}{self._lower_bound}, {self._upper_bound}{ubound_sign})"

    def __and__(self, other: Self) -> Self:
        """Return self & other, i.e. the most precise pattern that is less precise than both self and other"""
        if self._lower_bound < other._lower_bound:
            lbound, closed_lb = self._lower_bound, self._is_closed_lower_bound
        elif other._lower_bound < self._lower_bound:
            lbound, closed_lb = other._lower_bound, other._is_closed_lower_bound
        else:  # self._lower_bound == other._lower_bound
            lbound = self._lower_bound
            closed_lb = self._is_closed_lower_bound or other._is_closed_lower_bound

        if self._upper_bound > other._upper_bound:
            ubound, closed_ub = self._upper_bound, self._is_closed_upper_bound
        elif other._upper_bound > self._upper_bound:
            ubound, closed_ub = other._upper_bound, other._is_closed_upper_bound
        else:  # self._upper_bound == other._upper_bound
            ubound = self._upper_bound
            closed_ub = self._is_closed_upper_bound or other._is_closed_upper_bound

        new_value = (lbound, closed_lb), (ubound, closed_ub)
        return self.__class__(new_value)

    def __or__(self, other: Self) -> Self:
        """Return self | other, i.e. the least precise pattern that is more precise than both self and other"""
        if self._lower_bound < other._lower_bound:
            lbound, closed_lb = other._lower_bound, other._is_closed_lower_bound
        elif other._lower_bound < self._lower_bound:
            lbound, closed_lb = self._lower_bound, self._is_closed_lower_bound
        else:  # self._lower_bound == other._lower_bound
            lbound = self._lower_bound
            closed_lb = self._is_closed_lower_bound and other._is_closed_lower_bound

        if self._upper_bound > other._upper_bound:
            ubound, closed_ub = other._upper_bound, other._is_closed_upper_bound
        elif other._upper_bound > self._upper_bound:
            ubound, closed_ub = self._upper_bound, self._is_closed_upper_bound
        else:  # self._upper_bound == other._upper_bound
            ubound = self._upper_bound
            closed_ub = self._is_closed_upper_bound and other._is_closed_upper_bound

        new_value = (lbound, closed_lb), (ubound, closed_ub)
        return self.__class__(new_value)

    def __sub__(self, other: Self) -> Self:
        """Return self - other, i.e. the least precise pattern s.t. (self-other)|other == self"""
        # TODO: Find out how to implement this. And should it be implemented
        raise NotImplementedError
