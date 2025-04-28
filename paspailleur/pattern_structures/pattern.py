from typing import TypeVar, Self, Optional


class Pattern:
    """
    A class representing a pattern with various operations for pattern manipulation.

    This class allows for the creation and manipulation of patterns, including operations
    such as intersection, union, and difference. It also provides properties to check
    the characteristics of the pattern, such as whether it can be met, joined, or atomized.

    Attributes
    ----------
    PatternValueType: 
        A type variable representing the type of the pattern's value.
    _value: 
        The processed value of the pattern.
    """
    PatternValueType = TypeVar('PatternValueType')

    def __init__(self, value: PatternValueType):
        """
        Initializes a Pattern instance with a given value.

        Parameters
        ----------
        value: PatternValueType
            The value of the pattern, which can be a string or other types.

        Raises
        ------
        ValueError
            If the value cannot be parsed from a string.
        """
        if isinstance(value, str):
            try:
                value = self.parse_string_description(value)
            except Exception as e:
                raise ValueError(f"The Pattern's value cannot be parsed from a string {value}. "
                                 f"The following exception is raised: {e}")

        self._value = self.preprocess_value(value)  # if not isinstance(value, self.__class__) else value.value

    @property
    def value(self) -> PatternValueType:
        """
        Returns the value of the pattern.

        Returns
        -------
        PatternValueType
            The value of the pattern.
        
        Examples
        --------
        >>> p = Pattern("42")
        >>> p.value
        42
        """
        return self._value

    @property
    def meetable(self) -> bool:
        """
        Checks if the pattern can be met.

        Returns
        -------
        bool
            True if the pattern can be met, False otherwise.
        
        Examples
        --------
        >>> p = Pattern("example")
        >>> p.meetable
        True
        """
        try:
            self.meet(self)
        except NotImplementedError:
            return False
        return True

    @property
    def joinable(self) -> bool:
        """
        Checks if the pattern can be joined.

        Returns
        -------
        bool
            True if the pattern can be joined, False otherwise.
        
        Examples
        --------
        >>> p = Pattern("example")
        >>> p.joinable
        True
        """
        try:
            self.join(self)
        except NotImplementedError:
            return False
        return True

    @property
    def atomisable(self) -> bool:
        """
        Checks if the pattern can be atomized.

        Returns
        -------
        bool
            True if the pattern can be atomized, False otherwise.
        
        Examples
        --------
        >>> p = Pattern("example")
        >>> p.atomisable
        True
        """
        try:
            _ = self.atomic_patterns
        except NotImplementedError:
            return False
        return True

    @property
    def substractable(self) -> bool:
        """
        Checks if the pattern can be subtracted.

        Returns
        -------
        bool
            True if the pattern can be subtracted, False otherwise.
        
        Examples
        --------
        >>> p = Pattern("a")
        >>> p.substractable
        True
        """
        try:
            self.difference(self)
        except NotImplementedError:
            return False
        return True

    def __and__(self, other: Self) -> Self:
        """
        Returns the most precise pattern that is less precise than both self and other.

        Parameters
        ----------
        other : Self
            The other pattern to intersect with.

        Returns
        -------
        Self
            The intersected pattern.
        
        Examples
        --------
        >>> p1 = Pattern("A")
        >>> p2 = Pattern("B")
        >>> p3 = p1 & p2
        >>> p3.value
        'A & B'
        """ 
        if self.min_pattern is not None and other.min_pattern is not None:
            assert self.min_pattern == other.min_pattern, \
                "Minimal patterns defined for `self` (left-hand side value) and `other` (right-hand side value) differ." \
                " That should not be possible"

            if self.min_pattern == self or self.min_pattern == other:
                return self.min_pattern

        return self.__class__(self.value & other.value)

    def __or__(self, other: Self) -> Self:
        """
        Returns the least precise pattern that is more precise than both self and other.

        Parameters
        ----------
        other : Self
            The other pattern to union with.

        Returns
        -------
        Self
            The unioned pattern.
        
        Examples
        --------
        >>> p1 = Pattern("A")
        >>> p2 = Pattern("B")
        >>> p3 = p1 | p2
        >>> p3.value
        'A | B'
        """        
        if self.max_pattern is not None and other.max_pattern is not None:
            assert self.max_pattern == other.max_pattern, \
                "Maximal patterns defined for `self` (left-hand side value) and `other` (right-hand side value) differ." \
                " That should not be possible"

            if self.max_pattern == self or self.max_pattern == other:
                return self.max_pattern

        return self.__class__(self.value | other.value)

    def __sub__(self, other: Self) -> Self:
        """
        Returns the difference between self and another pattern.

        Parameters
        ----------
        other: Self
            Another pattern to subtract from self.

        Returns
        -------
        Self
            The least precise pattern such that (self - other) | other == self.
            If it's not possible, returns self.
        
        Examples
        --------
        >>> p1 = Pattern("A")
        >>> p2 = Pattern("B")
        >>> p3 = p1 - p2
        >>> p3.value
        'A'
        """
        if self.min_pattern is not None and self == other:
            return self.min_pattern
        return self.__class__(self.value)

    def __repr__(self) -> str:
        """
        Returns a string representation of the pattern
        
        Examples
        --------
        >>> p = Pattern("text")
        >>> repr(p)
        'text'
        """
        return str(self.value)

    def __len__(self) -> int:
        """
        Returns the minimal number of atomic patterns required to generate the pattern.

        Returns
        -------
        int
            The number of atomic patterns.
        
        Examples
        --------
        >>> p = Pattern("A & B")
        >>> len(p)
        2  # Assuming A and B are atomic patterns
        """
        if self.min_pattern is not None and self == self.min_pattern:
            return 0

        atoms = self.atomic_patterns
        atoms = {atom for atom in self.atomic_patterns if not any(other > atom for other in atoms - {atom})}
        return len(atoms)

    @classmethod
    def parse_string_description(cls, value: str) -> PatternValueType:
        """
        Parses a string description into a pattern value.

        Parameters
        ----------
        value: str
            The string description of the pattern.

        Returns
        -------
        PatternValueType
            The parsed pattern value.

        Examples
        --------
        >>> Pattern.parse_string_description("3 + 4")
        7
        """
        return eval(value)

    @classmethod
    def preprocess_value(cls, value) -> PatternValueType:
        """
        Preprocesses the value before storing it in the pattern.

        Parameters
        ----------
        value: 
            The value to preprocess.

        Returns
        -------
        PatternValueType
            The preprocessed value.
        
        Examples
        --------
        >>> Pattern.preprocess_value("raw_value")
        'raw_value'
        """
        return value

    def __eq__(self, other: Self) -> bool:
        """
        Checks if self is equal to another pattern.

        Parameters
        ----------
        other: Self
            Another pattern to compare with.

        Returns
        -------
        bool
            True if self is equal to other, False otherwise.
        
        Examples
        --------
        >>> p1 = Pattern("A")
        >>> p2 = Pattern("A")
        >>> p1 == p2
        True

        >>> p1= Pattern("A")
        >>> p2= Pattern("B")
        p1 == p2
        False
        """
        return self.value == other.value

    def __le__(self, other: Self) -> bool:
        """
        Checks if self is less precise or equal to another pattern.

        Parameters
        ----------
        other: Self
            Another pattern to compare with.

        Returns
        -------
        bool
            True if self is less precise or equal to other, False otherwise.
        
        Examples
        --------
        >>> p1 = Pattern("A")
        >>> p2 = Pattern("A | B")
        >>> p1 <= p2
        True
        """
        if self == other:
            return True

        return self & other == self

    def __lt__(self, other: Self) -> bool:
        """
        Checks if self is less precise than another pattern.

        Parameters
        ----------
        other: Self
            Another pattern to compare with.

        Returns
        -------
        bool
            True if self is less precise than other, False otherwise.
        
        Examples
        --------
        >>> p1 = Pattern("A")
        >>> p2 = Pattern("A | B")
        >>> p1 < p2
        True

        >>> p1 = Pattern("A")
        >>> p2 = Pattern("A")
        >>> p1 < p2
        False
        """
        return (self != other) and (self & other == self)

    def intersection(self, other: Self) -> Self:
        """
        Returns the intersection of self and another pattern.

        Parameters
        ----------
        other: Self
            Another pattern to intersect with.

        Returns
        -------
        Self
            The most precise pattern that is less precise than both self and other.
        
        Examples
        --------
        >>> p1 = Pattern("A")
        >>> p2 = Pattern("A & B")
        >>> p3 = p1.intersection(p2)
        >>> p3.value
        'A'
        """
        return self & other

    def union(self, other: Self) -> Self:
        """
        Returns the union of self and another pattern.

        Parameters
        ----------
        other: Self
            Another pattern to union with.

        Returns
        -------
        Self
            The least precise pattern that is more precise than both self and other.
        
        Examples
        --------
        >>> p1 = Pattern("A")
        >>> p2 = Pattern("B")
        >>> p3 = p1.union(p2)
        >>> p3.value
        'A | B'
        """
        return self | other

    def meet(self, other: Self) -> Self:
        """
        Returns the meeting of self and another pattern.

        Parameters
        ----------
        other: Self
            Another pattern to meet with.

        Returns
        -------
        Self
            The most precise pattern that is less precise than both self and other.
        
        Examples
        --------
        >>> p1 = Pattern("A")
        >>> p2 = Pattern("A & B")
        >>> p3 = p1.meet(p2)
        >>> p3.value
        'A'
        """
        return self & other

    def join(self, other: Self) -> Self:
        """
        Returns the joining of self and another pattern.

        Parameters
        ----------
        other: Self
            Another pattern to join with.

        Returns
        -------
        Self
            The least precise pattern that is more precise than both self and other.
        
        Examples
        --------
        >>> p1 = Pattern("A")
        >>> p2 = Pattern("B")
        >>> p3 = p1.union(p2)
        >>> p3.value
        'A | B'
        """
        return self | other

    def difference(self, other: Self) -> Self:
        """
        Returns the difference between self and another pattern.

        Parameters
        ----------
        other: Self
            Another pattern to subtract from self.

        Returns
        -------
        Self
            The least precise pattern such that (self - other) | other == self.
        
        Examples
        --------
        >>> p1 = Pattern("A | B")
        >>> p2 = Pattern("B")
        >>> p3 = p1.difference(p2)
        >>> p3.value
        'A'
        """
        return self - other

    def issubpattern(self, other: Self) -> Self:
        # shouldn't the -> to boolean instead of self?
        """
        Checks if self is less precise or equal to another pattern.

        Parameters
        ----------
        other: Self
            Another pattern to compare with.

        Returns
        -------
        Self
            True if self is less precise or equal to other.
        
        Examples
        --------
        >>> p1 = Pattern("A")
        >>> p2 = Pattern("A | B")
        >>> p1.issubpattern(p2)
        True
        """
        return self <= other

    def issuperpattern(self, other: Self) -> Self:
        # same question as the issubpattern function just above
        """
        Checks if self is more precise or equal to another pattern.

        Parameters
        ----------
        other: Self
            Another pattern to compare with.

        Returns
        -------
        Self
            True if self is more precise or equal to other.
        
        Examples
        --------
        >>> p1 = Pattern("A | B")
        >>> p2 = Pattern("A")
        >>> p1.issuperpattern(p2)
        True
        """
        return self >= other

    def __hash__(self):
        """
        Returns the hash of the pattern based on its value.

        Returns
        -------
        int
            The hash value of the pattern.
        
        Examples
        --------
        >>> p = Pattern("a")
        >>> hash(p)
        12345678  # (example hash)
        """
        return hash(self.value)

    @property
    def atomic_patterns(self) -> set[Self]:
        """
        Returns the set of all less precise patterns that cannot be obtained by intersection of other patterns.

        Raises
        ------
        NotImplementedError
            This method should be implemented in subclasses.
        
        Examples
        --------
        >>> p = Pattern("A & B")
        >>> p.atomic_patterns
        {'A', 'B'} # Assuming A and B are atomic patterns
        """
        raise NotImplementedError

    @property
    def min_pattern(self) -> Optional[Self]:
        """
        Returns the minimal possible pattern, the sole one per Pattern class.

        Returns
        -------
        Optional[Self]
            The minimal pattern or None if undefined.
        
        Examples
        --------
        >>> p = Pattern("A")
        >>> p.min_pattern
        None  # Assuming no minimal pattern is defined
        """
        return None

    @property
    def max_pattern(self) -> Optional[Self]:
        """
        Returns the maximal possible pattern, the sole one per Pattern class.

        Returns
        -------
        Optional[Self]
            The maximal pattern or None if undefined.
        
        Examples
        --------
        >>> p = Pattern("A")
        >>> p.max_pattern
        None  # Assuming no maximal pattern is defined
        """
        return None

    @property
    def maximal_atoms(self) -> Optional[set[Self]]:
        """
        Returns the maximal atomic patterns.

        Returns
        -------
        Optional[set[Self]]
            The set of maximal atomic patterns or None if undefined.
        
        Examples
        --------
        >>> p = Pattern("A & B")
        >>> p.maximal_atoms
        None  # Assuming no maximal atomic patterns are defined
        """
        return None
