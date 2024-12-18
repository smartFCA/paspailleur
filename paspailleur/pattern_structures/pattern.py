class Pattern:
    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value

    def __and__(self, other):
        """Return self & other, i.e. the most precise pattern that is less precise than both self and other"""
        return self.__class__(self.value & other.value)

    def __or__(self, other):
        """Return self | other, i.e. the least precise pattern that is more precise than both self and other"""
        return self.__class__(self.value | other.value)

    def __sub__(self, other):
        """Return self - other, i.e. the least precise pattern s.t. (self-other)|other == self"""
        return self.__class__(self.value - other.value)

    def __repr__(self):
        """String representation of the pattern"""
        return f"Pattern({self.value})"

    def __eq__(self, other):
        """Return self==other"""
        return self.value == other.value

    def __le__(self, other):
        """Return self<=other, i.e. whether self is less precise or equal to other"""
        return self.value <= other.value

    def __lt__(self, other):
        """Return self<other, i.e. whether self is less precise than other"""
        return (self != other) and (self <= other)

    def intersection(self, other):
        """Return self & other, i.e. the most precise pattern that is less precise than both self and other"""
        return self & other

    def union(self, other):
        """Return self | other, i.e. the least precise pattern that is more precise than both self and other"""
        return self | other

    def meet(self, other):
        """Return self & other, i.e. the most precise pattern that is less precise than both self and other"""
        return self & other

    def join(self, other):
        """Return self | other, i.e. the least precise pattern that is more precise than both self and other"""
        return self | other

    def difference(self, other):
        """Return self - other, i.e. the least precise pattern s.t. (self-other)|other == self"""
        return self - other

    def issubpattern(self, other):
        """Return self<=other, i.e. whether self is less precise or equal to other"""
        return self <= other

    def issuperpattern(self, other):
        """Return self>=other, i.e. whether self is more precise or equal to other"""
        return self >= other
