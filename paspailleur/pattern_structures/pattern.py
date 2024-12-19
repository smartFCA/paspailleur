from typing import TypeVar, Self


class Pattern:
    PatternValueType = TypeVar('PatternValueType')

    def __init__(self, value: PatternValueType):
        self._value = value

    @property
    def value(self) -> PatternValueType:
        return self._value

    def __and__(self, other: Self) -> Self:
        """Return self & other, i.e. the most precise pattern that is less precise than both self and other"""
        return self.__class__(self.value & other.value)

    def __or__(self, other: Self) -> Self:
        """Return self | other, i.e. the least precise pattern that is more precise than both self and other"""
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
