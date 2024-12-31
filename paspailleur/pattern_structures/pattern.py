from typing import TypeVar, Self, Optional


class Pattern:
    PatternValueType = TypeVar('PatternValueType')

    def __init__(self, value: PatternValueType):
        self._value = value

    @property
    def value(self) -> PatternValueType:
        return self._value

    def __and__(self, other: Self) -> Self:
        """Return self & other, i.e. the most precise pattern that is less precise than both self and other"""
        if self.min_pattern is not None and other.min_pattern is not None:
            assert self.min_pattern == other.min_pattern, \
                "Minimal patterns defined for `self` (left-hand side value) and `other` (right-hand side value) differ." \
                " That should not be possible"

            if self.min_pattern == self or self.min_pattern == other:
                return self.min_pattern

        return self.__class__(self.value & other.value)

    def __or__(self, other: Self) -> Self:
        """Return self | other, i.e. the least precise pattern that is more precise than both self and other"""
        if self.max_pattern is not None and other.max_pattern is not None:
            assert self.max_pattern == other.max_pattern, \
                "Maximal patterns defined for `self` (left-hand side value) and `other` (right-hand side value) differ." \
                " That should not be possible"

            if self.max_pattern == self or self.max_pattern == other:
                return self.max_pattern

        return self.__class__(self.value | other.value)

    def __sub__(self, other: Self) -> Self:
        """Return self - other, i.e. the least precise pattern s.t. (self-other)|other == self"""
        return self.__class__(self.value - other.value)

    def __repr__(self) -> str:
        """String representation of the pattern"""
        return f"Pattern({self.value})"

    def __eq__(self, other: Self) -> bool:
        """Return self==other"""
        return self.value == other.value

    def __le__(self, other: Self) -> bool:
        """Return self<=other, i.e. whether self is less precise or equal to other"""
        if self == other:
            return True

        return self & other == self

    def __lt__(self, other: Self) -> bool:
        """Return self<other, i.e. whether self is less precise than other"""
        return (self != other) and (self & other == self)

    def intersection(self, other: Self) -> Self:
        """Return self & other, i.e. the most precise pattern that is less precise than both self and other"""
        return self & other

    def union(self, other: Self) -> Self:
        """Return self | other, i.e. the least precise pattern that is more precise than both self and other"""
        return self | other

    def meet(self, other: Self) -> Self:
        """Return self & other, i.e. the most precise pattern that is less precise than both self and other"""
        return self & other

    def join(self, other: Self) -> Self:
        """Return self | other, i.e. the least precise pattern that is more precise than both self and other"""
        return self | other

    def difference(self, other: Self) -> Self:
        """Return self - other, i.e. the least precise pattern s.t. (self-other)|other == self"""
        return self - other

    def issubpattern(self, other: Self) -> Self:
        """Return self<=other, i.e. whether self is less precise or equal to other"""
        return self <= other

    def issuperpattern(self, other: Self) -> Self:
        """Return self>=other, i.e. whether self is more precise or equal to other"""
        return self >= other

    def __hash__(self):
        return hash(self.value)

    @property
    def atomic_patterns(self) -> set[Self]:
        """Return the set of all less precise patterns that cannot be obtained by intersection of other patterns"""
        raise NotImplementedError

    @property
    def min_pattern(self) -> Optional[Self]:
        """Minimal possible pattern, the sole one per Pattern class. `None` if undefined"""
        return None

    @property
    def max_pattern(self) -> Optional[Self]:
        """Minimal possible pattern, the sole one per Pattern class. `None` if undefined"""
        return None
