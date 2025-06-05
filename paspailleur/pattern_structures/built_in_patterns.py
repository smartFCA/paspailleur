import math
from builtins import set
from typing import Self, Collection, Optional, Sequence, Type, Literal
from numbers import Number
from frozendict import frozendict
import re


from .pattern import Pattern


class ItemSetPattern(Pattern):
    """
    A class representing a set of items as a pattern.

    This class allows for the creation and manipulation of patterns that consist of a set of items.
    It supports operations such as union, intersection, and difference.

    References
    ..........
    Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules in large databases.

    Attributes
    ..........
    PatternValueType:
        The type of the pattern's value, which is a frozenset.

    Properties
    ..........
    value
        Return the value of the item set pattern
    atomic_patterns
        Return the set of all less precise patterns that cannot be obtained by intersection of other patterns.
    min_pattern
        Return the minimal possible pattern for the item set pattern.
    """
    PatternValueType = frozenset

    @property
    def value(self) -> PatternValueType:
        """
        Return the value of the item set pattern.

        Returns
        -------
        value: PatternValueType
            The value of the itemset pattern, so the set of items itself.

        Examples
        --------
        >>> p = ItemSetPattern({1, 2, 3})
        >>> p.value
        frozenset({1, 2, 3})
        """
        return self._value

    def __repr__(self) -> str:
        """
        Return a string representation of the item set pattern.

        Returns
        -------
        representation: str

            The string representation of the item set pattern.

        Examples
        --------
        >>> p = ItemSetPattern({1, 2})
        >>> repr(p)
        '{1, 2}'
        """
        return repr(set(self.value))

    def __len__(self) -> int:
        """
        Return the minimal number of atomic patterns required to generate the item set pattern.

        Returns
        -------
        length: int
            The number of items in the item set pattern.

        Examples
        --------
        >>> p = ItemSetPattern({1, 2, 3})
        >>> len(p)
        3
        """
        return len(self.value)

    def __sub__(self, other: Self) -> Self:
        """
        Return the itemSetPattern that contains the items that can be found in self but not in other.

        Parameters
        ----------
        other: Self
            Another item set pattern to subtract from self.

        Returns
        -------
        Self
            The least precise pattern such that (self - other) | other == self.
            If it's not possible, returns self.

        Examples
        --------
        >>> p1 = ItemSetPattern({1, 2, 3})
        >>> p2 = ItemSetPattern({2})
        >>> p3 = p1 - p2
        >>> p3
        {1, 3}
        """
        return self.__class__(self.value - other.value)

    @classmethod
    def parse_string_description(cls, value: str) -> PatternValueType:
        """
        Parse a string description into an item set pattern value.

        Parameters
        ----------
        value: str
            The string description of the item set pattern.

        Returns
        -------
        parsed: PatternValueType
            The parsed item set pattern value.

        Raises
        ------
        ValueError
            If the value cannot be parsed into an ItemSetPattern object.

        Examples
        --------
        >>> ItemSetPattern.parse_string_description('{1, 2, 3}')
        frozenset({1, 2, 3})
        """
        parsed_value = None

        try:
            parsed_value = eval(value)
        except Exception as e:
            pass

        if parsed_value is not None and isinstance(parsed_value, Collection):
            return frozenset(parsed_value)

        is_bounded = value.startswith(('(', '[', '{')) and value.endswith((')', ']', '}'))
        value_iterator = value[1:-1].split(',') if is_bounded else value.split(',')
        parsed_value = []
        for v in value_iterator:
            try:
                new_v = eval(v)
            except:
                new_v = v
            parsed_value.append(new_v)

        if parsed_value is not None:
            return frozenset(parsed_value)

        raise ValueError(f'Value {value} cannot be parsed into {cls.__name__} object')

    @classmethod
    def preprocess_value(cls, value: Collection) -> PatternValueType:
        """
        Preprocess the value before storing it in the item set pattern.

        Parameters
        ----------
        value: Collection
            The value to preprocess.

        Returns
        -------
        value: frozenset
            The preprocessed value as a frozenset.

        Examples
        --------
        >>> ItemSetPattern.preprocess_value([1, 2, 2])
        frozenset({1, 2})
        """
        return frozenset(value)


    def atomise(self, atoms_configuration: Literal['min', 'max'] = 'min') -> set[Self]:
        """
        Split the pattern into atomic patterns, i.e. ItemSets containing just one item.

        Parameters
        ----------
        atoms_configuration: Literal['min', 'max']
            Specifically for ItemSetPattern, the value of `atoms_configuration` parameter *does not affect* the output
            of the function.

        Returns
        -------
        atomic_patterns: set[Self]
            The set of atomic patterns, i.e. the set of unsplittable patterns whose join equals to the pattern.


        Notes
        -----
        Speaking in terms of Ordered Set Theory:
        We say that every pattern can be represented as the join of a subset of atomic patterns,
        that are join-irreducible elements of the lattice of all patterns.

        Considering the set of atomic patterns as a partially ordered set (where the order follows the order on patterns),
        every pattern can be represented by an _antichain_ of atomic patterns (when `atoms_configuration` = 'min'),
        and by an *order ideal* of atomic patterns (when `atoms_configuration` = 'max').

        """
        return {self.__class__({v}) for v in self.value}


    @property
    def atomic_patterns(self) -> set[Self]:
        """
        Return the set of all less precise patterns that cannot be obtained by intersection of other patterns.

        For an ItemSetPattern an atomic pattern is a pattern containing one itme.

        Returns
        -------
        atoms: set[Self]
            A set of atomic patterns, each containing a single item from the item set pattern.

        Examples
        --------
        >>> p = ItemSetPattern({1, 2})
        >>> p.atomic_patterns
        {ItemSetPattern({1}), ItemSetPattern({2})}
        """
        return super().atomic_patterns

    @classmethod
    def get_min_pattern(cls) -> Self:
        """
        Return the minimal possible pattern, i.e. empty ItemSetPattern

        Returns
        -------
        Self
            The minimal item set pattern, which is an empty frozenset.
            `None` if undefined

        Examples
        --------
        >>> ItemSetPattern.get_min_pattern()
        ItemSetPattern(frozenset())
        """
        return cls(frozenset())


class CategorySetPattern(ItemSetPattern):
    """
    A class representing a set of categories as a pattern.

    This class extends ItemSetPattern to include operations specific to category sets, such as handling a universe of categories.
    But it's also an inversion of it, meaning for an object to fall under the ItemSetPattern class, it has to fit the the descriptions of all items in the set. On the other hand for an object to fall under the CategorySetPattern, it needs only fit to one item in the set.
    
    Attributes
    ..........
    PatternValueType:
        The type of the pattern's value, which is a frozenset.
    Universe: Optional[frozenset]
        The set of all possible categories.
    
    Properties
    ..........
    atomic_patterns
        Return the set of all less precise patterns that cannot be obtained by intersection of other patterns.
    min_pattern
        Return the minimal possible pattern for the category set pattern.
    max_pattern
        Return the maximal possible pattern for the category set pattern.
    """
    PatternValueType = frozenset
    Universe: Optional[frozenset] = None  # The set of all possible categories

    def __and__(self, other):
        """
        Return the union of self and another category set pattern.

        Parameters
        ----------
        other: Self
            Another category set pattern to intersect with.

        Returns
        -------
        Self
            The union of the two category set patterns.

        Examples
        --------
        >>> p1 = CategorySetPattern({"A", "B"})
        >>> p2 = CategorySetPattern({"B", "C"})
        >>> p1 & p2
        {'A', 'B', 'C'}
        """
        return self.__class__(self.value | other.value)

    def __or__(self, other):
        """
        Return the intersection of self and another category set pattern.

        Parameters
        ----------
        other: Self
            Another category set pattern to intersect with.

        Returns
        -------
        Self
            The intersection of the two category set patterns.

        Examples
        --------
        >>> p1 = CategorySetPattern({"A", "B"})
        >>> p2 = CategorySetPattern({"B", "C"})
        >>> p1 | p2
        {'B'}
        """
        return self.__class__(self.value & other.value)

    def __repr__(self) -> str:
        """
        Return a string representation of the category set pattern.

        Returns
        -------
        representation: str
            The string representation of the category set pattern.

        Examples
        --------
        >>> CategorySetPattern({"A", "B"}).__repr__()
        '{'A', 'B'}'
        """
        repr_negative = self.Universe is not None and len(self.Universe) / 2 < len(self.value) < len(self.Universe)
        s = set(self.Universe) - self.value if repr_negative else self.value
        s = repr(set(s))
        if repr_negative:
            s = f"NOT({s})"
        return s

    def __sub__(self, other):
        """
        Return the CategorySetPattern that contains the items that can be found in self but not in other.

        Parameters
        ----------
        other: Self
            Another category set pattern to subtract from self.

        Returns
        -------
        Self
            The least precise pattern such that (self - other) | other == self.
            If it's not possible, returns self.

        Examples
        --------
        >>> p1 = CategorySetPattern({"A", "B"})
        >>> p2 = CategorySetPattern({"B"})
        >>> p1 - p2
        {'A'}
        """
        if self.min_pattern is not None and self == other:
            return self.min_pattern
        return self.__class__(self.value)

    def atomise(self, atoms_configuration: Literal['min', 'max'] = 'min') -> set[Self]:
        """
        Split the pattern into atomic patterns, i.e. CategorySets that exclude just one category from the minimal pattern

        Parameters
        ----------
        atoms_configuration: Literal['min', 'max']
            Specifically for CategorySetPattern, the value of `atoms_configuration` parameter *does not affect* the
            output of the function.

        Returns
        -------
        atomic_patterns: set[Self]
            The set of atomic patterns, i.e. the set of unsplittable patterns whose join equals to the pattern.


        Notes
        -----
        Speaking in terms of Ordered Set Theory:
        We say that every pattern can be represented as the join of a subset of atomic patterns,
        that are join-irreducible elements of the lattice of all patterns.

        Considering the set of atomic patterns as a partially ordered set (where the order follows the order on patterns),
        every pattern can be represented by an •antichain* of atomic patterns (when `atoms_configuration` = 'min'),
        and by an *order ideal* of atomic patterns (when `atoms_configuration` = 'max').

        """
        assert self.min_pattern is not None, \
            f"Pattern of  {self.__class__} class cannot be splitted without predefined min_pattern value. " \
            f"The proposed solution is to inherit a new class from the current class and " \
            f"explicitly specify the value of min_pattern."

        leftout_vals = self.min_pattern.value - self.value
        return {self.__class__(self.min_pattern.value - {v}) for v in leftout_vals}

    @property
    def atomic_patterns(self) -> set[Self]:
        """
        Return the set of all less precise patterns that cannot be obtained by intersection of other patterns.

        For a CategorySetPattern an atomic pattern is a set containing all-but-one categories.

        Returns
        -------
        atoms: set[Self]
            A set of atomic patterns derived from the universe of categories.

        Raises
        ------
        AssertionError
            If the min_pattern is not defined.

        Examples
        --------
        >>> p = CategorySetPattern({"A"})
        >>> p.atomic_patterns
        {CategorySetPattern({'B'}), CategorySetPattern({'C'})} # assuming Universe={'A','B','C'}
        """
        assert self.min_pattern is not None,\
            f"Atomic patterns of {self.__class__} class cannot be computed without predefined min_pattern value. " \
            f"The proposed solution is to inherit a new class from the current class and " \
            f"explicitly specify the value of min_pattern."
        return super().atomic_patterns

    @classmethod
    def get_min_pattern(cls) -> Optional[Self]:
        """
        Return the minimal possible pattern for the category set pattern, i.e. CategorySet containing all categories

        Returns
        -------
        min: Optional[Self]
            The minimal category set pattern or None if undefined.

        Examples
        --------
        >>> CategorySetPattern.Universe = {"A", "B", "C"}
        >>> CategorySetPattern.get_min_pattern()
        CategorySetPattern({'A', 'B', 'C'})
        """
        return cls(cls.Universe) if cls.Universe is not None else None

    @classmethod
    def get_max_pattern(cls) -> Self:
        """
        Return the maximal possible pattern for the category set pattern, i.e. empty CategorySet

        Empty CategorySet is the maximal possible pattern, because it does not allow for any category.
        That is, it is so maximal and so precise, that it should never occur in the data.

        Returns
        -------
        max: Self
            The maximal category set pattern, which is an empty CategorySet

        Examples
        --------
        >>> CategorySetPattern.get_max_pattern()
        CategorySetPattern(frozenset())
        """
        return cls(frozenset())

    def __len__(self) -> int:
        """
        Return the minimal number of atomic patterns required to generate the category set pattern.

        Returns
        -------
        length: int
            The number of categories not included in the category set pattern.
            The length of a pattern is the categories the pattern.value does _not_ include

        Raises
        ------
        AssertionError
            If the min_pattern is not defined.

        Examples
        --------
        >>> CategorySetPattern.Universe = {"A", "B", "C"}
        >>> p = CategorySetPattern({"A"})
        >>> len(p)
        2
        """
        assert self.min_pattern is not None, f"Length of pattern of {self.__class__} " \
                                             f"class cannot be computed without the predefined min_pattern value."
        return len(self.min_pattern.value) - len(self.value)


class IntervalPattern(Pattern):
    """
    A class representing an interval as a pattern.

    This class allows for the creation and manipulation of patterns that represent intervals,
    including operations such as intersection, union, and difference.

    Attributes
    ..........
    PatternValueType:
        The type of the pattern's value, which is a tuple of bounds.
    BoundsUniverse: list[float]
        A list representing the universe of bounds for the interval.
    
    Properties
    ..........
    lower_bound
        Return the lower bound of the interval.
    is_lower_bound_closed
        Check if the lower bound is closed.
    upper_bound
        Return the upper bound of the interval.
    is_upper_bound_closed
        Check if the upper bound is closed.
    atomic_patterns
        Return the set of all less precise patterns that cannot be obtained by intersection of other patterns.
    min_pattern
        Return the minimal possible pattern for the interval pattern.
    max_pattern
        Return the maximal possible pattern for the interval pattern.
    maximal_atoms
        Return the maximal atomic patterns for the interval pattern.
    
    Notes
    -----
    The IntervalPattern can be defined by a string representation of the interval, or by an pair of two numbers, or by just one number.

    Examples
    --------
    >>> p1 = IntervalPattern( ((5, True), (10, False)) )  # The explicit way to define half-open interval [5, 10)
    >>> p2 = IntervalPattern( "[5, 10)" )  # A more readable way to define interval [5, 10)
    >>> print(p1 == p2)
    True

    >>> p3 = IntervalPattern( ((1, True), (1, True)) )  # The explicit way to define closed interval [1, 1]
    >>> p4 = IntervalPattern( "[1, 1]" )
    >>> p5 = IntervalPattern( [1, 1]  )
    >>> p6 = IntervalPattern( 1 )
    >>> print(p3 == p4, p3 == p5, p3 == p6)
    True, True, True

    >>> p7 = IntervalPattern( "[1, ∞]" )
    >>> p8 = IntervalPattern( ">=1" )
    >>> print(p7 == p8)
    True
    """
    # PatternValue semantics: ((lower_bound, is_closed), (upper_bound, is_closed))
    PatternValueType = tuple[tuple[float, bool], tuple[float, bool]]
    BoundsUniverse: list[float] = None

    @property
    def lower_bound(self) -> float:
        """
        Return the lower bound of the interval.

        Returns
        -------
        bound: float
            The lower bound of the interval.

        Examples
        --------
        >>> IntervalPattern( "[1, 5)" ).lower_bound
        1.0
        """
        return self.value[0][0]

    @property
    def is_lower_bound_closed(self) -> bool:
        """
        Check if the lower bound is closed.

        Returns
        -------
        closed: bool
            True if the lower bound is closed, False otherwise.

        Examples
        --------
        >>> IntervalPattern( "[1, 5)" ).is_lower_bound_closed
        True
        """
        return self.value[0][1]

    @property
    def upper_bound(self) -> float:
        """
        Return the upper bound of the interval.

        Returns
        -------
        bound: float
            The upper bound of the interval.

        Examples
        --------
        >>> IIntervalPattern( "[1, 5)" ).upper_bound
        5.0
        """
        return self.value[1][0]

    @property
    def is_upper_bound_closed(self) -> bool:
        """
        Check if the upper bound is closed.

        Returns
        -------
        closed: bool
            True if the upper bound is closed, False otherwise.

        Examples
        --------
        >>> IntervalPattern( "[1, 5)" ).is_upper_bound_closed
        False
        """
        return self.value[1][1]

    def __repr__(self) -> str:
        """
        Return a string representation of the interval pattern.

        Returns
        -------
        representation: str
            The string representation of the interval pattern.

        Examples
        --------
        >>> repr(IntervalPattern( "[1, 5)" ))
        '[1.0, 5.0)'
        """
        if self == self.max_pattern:
            return 'ø'

        if self.lower_bound == self.upper_bound:
            return f"{self.lower_bound}"

        if self.lower_bound == -math.inf and self.is_lower_bound_closed and self.upper_bound < math.inf:
            return '<'+('=' if self.is_upper_bound_closed else '')+f' {self.upper_bound}'
        if self.upper_bound == math.inf and self.is_upper_bound_closed and self.lower_bound > -math.inf:
            return '>'+('=' if self.is_lower_bound_closed else '')+f' {self.lower_bound}'

        lbound_sign = '[' if self.is_lower_bound_closed else '('
        ubound_sign = ']' if self.is_upper_bound_closed else ')'
        return f"{lbound_sign}{self.lower_bound}, {self.upper_bound}{ubound_sign}"

    def __len__(self) -> int:
        """
        Return the minimal number of atomic patterns required to generate the interval pattern.

        Returns
        -------
        count: int
            The number of non-infinite bounds in the interval.

        Examples
        --------
        >>> len(IntervalPattern( "[1, 5]" ))
        2
        """
        return int(self.lower_bound != -math.inf) + int(self.upper_bound != math.inf)

    @classmethod
    def parse_string_description(cls, value: str) -> PatternValueType:
        """
        Parse a string description into an interval pattern value.

        Parameters
        ----------
        value: str
            The string description of the interval pattern.

        Returns
        -------
        parsed: PatternValueType
            The parsed interval pattern value.

        Examples
        --------
        >>> IntervalPattern.parse_string_description('[1, 5)')
        ((1.0, True), (5.0, False))
        """
        value = value.strip()
        if value == 'ø':
            return (0, False), (0, False)

        if value.startswith('>'):
            if value.startswith('>='):
                return (eval(value[2:]), True), (math.inf, True)
            return (eval(value[1:]), False), (math.inf, True)

        if value.startswith('<'):
            if value.startswith('<='):
                return (-math.inf, True), (eval(value[2:]), True)
            return (-math.inf, True), (eval(value[1:]), False)

        if ',' not in value:
            try:
                return (float(value), True), (float(value), True)
            except Exception as e:
                pass

        lb, ub = map(str.strip, value[1:-1].replace('∞', 'inf').split(','))
        closed_lb, closed_ub = value[0] == '[', value[-1] == ']'
        return (float(lb), bool(closed_lb)), (float(ub), bool(closed_ub))

    @classmethod
    def preprocess_value(cls, value) -> PatternValueType:
        """
        Preprocess the value before storing it in the interval pattern.

        Parameters
        ----------
        value:
            The value to preprocess.

        Returns
        -------
        value: PatternValueType
            The preprocessed value as a tuple of bounds.

        Examples
        --------
        >>> IntervalPattern.preprocess_value((1, 5))
        ((1.0, True), (5.0, True))
        """
        if isinstance(value, Number):
            value = (value, value)

        if len(value) == 2 and all(isinstance(v, Number) for v in value):
            value = (float(value[0]), True), (float(value[1]), True)

        lb, closed_lb = value[0]
        rb, closed_rb = value[1]

        is_contradictive = (rb < lb) or (lb == rb and not (closed_lb and closed_rb))
        if is_contradictive:
            lb, rb = 0, 0

        if cls.BoundsUniverse is not None and not is_contradictive:
            lb = max(b for b in cls.BoundsUniverse if b <= lb) if lb > -math.inf else lb
            rb = min(b for b in cls.BoundsUniverse if rb <= b) if rb < math.inf else rb

        return (float(lb), bool(closed_lb)), (float(rb), bool(closed_rb))

    def __and__(self, other: Self) -> Self:
        """
        Return the **union** of self and another interval pattern.

        Parameters
        ----------
        other: Self
            Another interval pattern to union with.

        Returns
        -------
        Self
            The most precise pattern that is less precise than both self and other.

        Notes
        -----
        Instead of having a normal intersection of intervals, we have an intersection of the restrictions meaning a union.
        Because the union is the only way to get the most presice interval pattern that is less precise than both self and other

        Examples
        --------
        >>> p1 = IntervalPattern( "[1, 5)" )
        >>> p2 = IntervalPattern( "[3, 6)" )
        >>> p1 & p2
        [1.0, 6.0) 
        """
        if self == self.min_pattern or other == self.min_pattern:
            return self.min_pattern

        if self == self.max_pattern:
            return other
        if other == self.max_pattern:
            return self

        if self.lower_bound < other.lower_bound:
            lbound, closed_lb = self.lower_bound, self.is_lower_bound_closed
        elif other.lower_bound < self.lower_bound:
            lbound, closed_lb = other.lower_bound, other.is_lower_bound_closed
        else:  # self._lower_bound == other._lower_bound
            lbound = self.lower_bound
            closed_lb = self.is_lower_bound_closed or other.is_lower_bound_closed

        if self.upper_bound > other.upper_bound:
            ubound, closed_ub = self.upper_bound, self.is_upper_bound_closed
        elif other.upper_bound > self.upper_bound:
            ubound, closed_ub = other.upper_bound, other.is_upper_bound_closed
        else:  # self._upper_bound == other._upper_bound
            ubound = self.upper_bound
            closed_ub = self.is_upper_bound_closed or other.is_upper_bound_closed

        new_value = (lbound, closed_lb), (ubound, closed_ub)
        return self.__class__(new_value)

    def __or__(self, other: Self) -> Self:
        """
        Return the **intersection** of self and another interval pattern.

        Parameters
        ----------
        other: Self
            Another interval pattern to union with.

        Returns
        -------
        Self
            The least precise pattern that is more precise than both self and other.

        Examples
        --------
        >>> p1 = IntervalPattern( "[1, 5)" )
        >>> p2 = IntervalPattern( "[3, 6]" )
        >>> p1 | p2
        [1.0, 5.0)
        """
        if self == self.max_pattern or other == self.max_pattern:
            return self.max_pattern

        if self == self.min_pattern:
            return other
        if other == self.min_pattern:
            return self

        if self.lower_bound < other.lower_bound:
            lbound, closed_lb = other.lower_bound, other.is_lower_bound_closed
        elif other.lower_bound < self.lower_bound:
            lbound, closed_lb = self.lower_bound, self.is_lower_bound_closed
        else:  # self._lower_bound == other._lower_bound
            lbound = self.lower_bound
            closed_lb = self.is_lower_bound_closed and other.is_lower_bound_closed

        if self.upper_bound > other.upper_bound:
            ubound, closed_ub = other.upper_bound, other.is_upper_bound_closed
        elif other.upper_bound > self.upper_bound:
            ubound, closed_ub = self.upper_bound, self.is_upper_bound_closed
        else:  # self._upper_bound == other._upper_bound
            ubound = self.upper_bound
            closed_ub = self.is_upper_bound_closed and other.is_upper_bound_closed

        new_value = (lbound, closed_lb), (ubound, closed_ub)
        if (lbound > ubound) \
                or (lbound == ubound and not (closed_lb and closed_ub)):
            return self.max_pattern
        return self.__class__(new_value)

    def __sub__(self, other: Self) -> Self:
        """
        Return the IntervalPattern that contains the items that can be found in self but not in other.

        Parameters
        ----------
        other: Self
            Another interval pattern to subtract from self.

        Returns
        -------
        Self
            The least precise pattern such that (self - other) | other == self.
            If it's not possible, returns self.

        Examples
        --------
        [1] p2 included in p1 
        >>> p1 = IntervalPattern( "[1, 5]" )
        >>> p2 = IntervalPattern( "[2, 3]" )
        >>> p1 - p2
        [1.0, 5.0]
        [2] p1 included in p2
        >>> p1 = IntervalPattern( "[1, 5]" )
        >>> p2 = IntervalPattern( "[0, 6]" )
        >>> p1 - p2
        [1.0, 5.0]
        [3]  shared lower bound p1 higher upper bound
        >>> p1 = IntervalPattern( "[1, 5]" )
        >>> p2 = IntervalPattern( "[1, 3]" )
        >>> p1 - p2
        [1.0, 5.0]
        [4] shared upper bound p1 lower lower bound
        >>> p1 = IntervalPattern( "[0, 5]" )
        >>> p2 = IntervalPattern( "[1, 5]" )
        >>> p1 - p2
        [0.0, 5.0]
        [5] shared lower bound p2 higher upper bound
        >>> p1 = IntervalPattern( "[1, 3]" )
        >>> p2 = IntervalPattern( "[1, 5]" )
        >>> p1 - p2
        <=3.0
        [6] shared upper bound p2 lower lower bound
        >>> p1 = IntervalPattern( "[1, 5]" )
        >>> p2 = IntervalPattern( "[0, 5]" )
        >>> p1 - p2
        >=1.0
        [7] p1==p2
        >>> p1 = IntervalPattern( "[1, 5]" )
        >>> p2 = IntervalPattern( "[1, 5]" )
        >>> p1 - p2
        [-inf, inf]

        Warning
        -------
        The behavior of the function might change soon
        """
        if self == other:
            return self.min_pattern

        same_lower_bound = self.lower_bound == other.lower_bound and self.is_lower_bound_closed == other.is_lower_bound_closed
        smaller_upper_bound = (self.upper_bound < other.upper_bound) or (self.upper_bound == other.upper_bound and not self.is_upper_bound_closed and other.is_upper_bound_closed)
        if same_lower_bound and smaller_upper_bound:
            return self.__class__((self.min_pattern.value[0], (self.upper_bound, self.is_upper_bound_closed)))

        same_upper_bound = self.upper_bound == other.upper_bound and self.is_upper_bound_closed == other.is_upper_bound_closed
        greater_lower_bound = (other.lower_bound < self.lower_bound) or (self.lower_bound == other.lower_bound and self.is_lower_bound_closed and not other.is_lower_bound_closed)
        if same_upper_bound and greater_lower_bound:
            return self.__class__(((self.lower_bound, self.is_lower_bound_closed), self.min_pattern.value[1]))

        # subtraction is impossible
        return self.__class__(self.value)

    def atomise(self, atoms_configuration: Literal['min', 'max'] = 'min') -> set[Self]:
        """
        Split the pattern into atomic patterns, i.e. the set of one-sided intervals

        Parameters
        ----------
        atoms_configuration: Literal['min', 'max']
            If equals to 'min', return up to 2 atomic patterns each representing a bound of the original interval.
            If equals to 'max', return _all_ less precise one-sided intervals, where the bounds are defined by
            `BoundUniverse` class attribute.

        Returns
        -------
        atomic_patterns: set[Self]
            The set of atomic patterns, i.e. the set of unsplittable patterns whose join equals to the pattern.


        Notes
        -----
        Speaking in terms of Ordered Set Theory:
        We say that every pattern can be represented as the join of a subset of atomic patterns,
        that are join-irreducible elements of the lattice of all patterns.

        Considering the set of atomic patterns as a partially ordered set (where the order follows the order on patterns),
        every pattern can be represented by an _antichain_ of atomic patterns (when `atoms_configuration` = 'min'),
        and by an *order ideal* of atomic patterns (when `atoms_configuration` = 'max').

        """
        if self.value == self.max_pattern.value:
            return {self.max_pattern}

        if atoms_configuration == 'min':
            atoms = []
            if not (self.lower_bound == -math.inf and self.is_lower_bound_closed):
                atoms.append( ((self.lower_bound, self.is_lower_bound_closed), (math.inf, True)) )
            if not (self.upper_bound == math.inf and self.is_upper_bound_closed):
                atoms.append( ((-math.inf, True), (self.upper_bound, self.is_upper_bound_closed)) )
            return {self.__class__(v) for v in atoms}

        # atoms_configuration == 'max'
        assert self.BoundsUniverse is not None, ("Please define BoundsUniverse class attribute in order to "
                                                 "be able to compute all atomic patterns for a specified pattern.")


        atoms = []
        for bound in self.BoundsUniverse:
            for is_bound_closed in [False, True]:
                if bound >= self.upper_bound:
                    atoms.append( ((-math.inf, True), (bound, is_bound_closed)) )
                if bound <= self.lower_bound:
                    atoms.append( ((bound, is_bound_closed), (math.inf, True))  )
        if self.is_upper_bound_closed:
            excessive_atom = ((-math.inf, True), (self.upper_bound, False))
            if excessive_atom in atoms: atoms.remove(excessive_atom)
        if self.is_lower_bound_closed:
            excessive_atom = ((self.lower_bound, False), (math.inf, True))
            if excessive_atom in atoms: atoms.remove(excessive_atom)

        return {self.__class__(v) for v in atoms}

    @property
    def atomic_patterns(self) -> set[Self]:
        """
        Return the set of all less precise patterns that cannot be obtained by intersection of other patterns.

        For an IntervalPattern an atomic pattern is a half-bounded interval.

        Returns
        -------
        atoms: set[Self]
            A set of atomic patterns derived from the interval.

        Examples
        --------
        >>> IntervalPattern( "[1, 5]" )atomic_patterns
        {IntervalPattern("[-∞, ∞]"), IntervalPattern(">=1"), IntervalPattern("<=5") }
        """
        return super().atomic_patterns

    @property
    def atomisable(self) -> bool:
        """
        Check if the pattern can be atomized. IntervalPattern can only be atomised when `BoundsUniverse` is defined

        Returns
        -------
        flag: bool
            True if the pattern can be atomized, False otherwise.

        Examples
        --------
        >>> p = Pattern("example")
        >>> p.atomisable
        True

        """
        if self.BoundsUniverse is None:
            return False

        return super().atomisable

    @classmethod
    def get_min_pattern(cls) -> Self:
        """
        Return the minimal possible pattern for the interval pattern, i.e. interval [-inf, +inf]

        Returns
        -------
        min: Self
            The minimal interval pattern, which is defined as [-inf, inf].

        Examples
        --------
        >>> IntervalPattern.get_min_pattern()
        IntervalPattern('[-inf, +inf]')
        """
        return cls(((-math.inf, True), (math.inf, True)))

    @classmethod
    def get_max_pattern(cls) -> Self:
        """
        Return the maximal possible pattern for the interval pattern, i.e. the empty interval.

        Empty interval is the maximal pattern, because it does not cover any other interval. So it describes no objects.


        Returns
        -------
        max: Self
            The maximal interval pattern, which is represented as 'ø'.

        Examples
        --------
        >>> IntervalPattern.get_max_pattern()
        IntervalPattern('ø')
        """
        return cls("ø")

    @property
    def maximal_atoms(self) -> Optional[set[Self]]:
        """
        Return the maximal atomic patterns for the interval pattern.

        Returns
        -------
        max_atoms: Optional[set[Self]]
            A set of maximal atomic patterns or None if undefined.

        Examples
        --------
        >>> IntervalPattern().maximal_atoms
        {...}
        """
        return {self.max_pattern}


class ClosedIntervalPattern(IntervalPattern):
    """
    A class representing a closed interval as a pattern.

    This class extends IntervalPattern to specifically handle closed intervals.

    Attributes
    ..........
    PatternValueType :
        The type of the pattern's value, which is a tuple of two floats.
    
    Properties
    ..........
    lower_bound
        Return the lower bound of the closed interval.
    upper_bound
        Return the upper bound of the closed interval.
    is_lower_bound_closed
        Check if the lower bound is closed.
    is_upper_bound_closed
        Check if the upper bound is closed.
    """
    PatternValueType = tuple[float, float]

    @property
    def lower_bound(self) -> float:
        """
        Return the lower bound of the closed interval.

        Returns
        -------
        bound: float
            The lower bound of the closed interval.

        Examples
        --------
        >>> ClosedIntervalPattern((1.0, 5.0)).lower_bound
        1.0
        """
        return self._value[0]

    @property
    def upper_bound(self) -> float:
        """
        Return the upper bound of the closed interval.

        Returns
        -------
        bound: float
            The upper bound of the closed interval.

        Examples
        --------
        >>> ClosedIntervalPattern((1.0, 5.0)).upper_bound
        5.0
        """
        return self._value[1]

    @property
    def is_lower_bound_closed(self) -> bool:
        """
        Check if the lower bound is closed.

        Returns
        -------
        closed: bool
            True, indicating that the lower bound is always closed.

        Examples
        --------
        >>> ClosedIntervalPattern((1.0, 5.0)).is_lower_bound_closed
        True
        """
        return True

    @property
    def is_upper_bound_closed(self) -> bool:
        """
        Check if the upper bound is closed.

        Returns
        -------
        closed: bool
            True, indicating that the upper bound is always closed.

        Examples
        --------
        >>> ClosedIntervalPattern((1.0, 5.0)).is_upper_bound_closed
        True
        """
        return True

    @classmethod
    def parse_string_description(cls, value: str) -> PatternValueType:
        """
        Parse a string description into a closed interval pattern value.

        Parameters
        ----------
        value: str
            The string description of the closed interval pattern.

        Returns
        -------
        parsed: PatternValueType
            The parsed closed interval pattern value.

        Raises
        ------
        AssertionError
            If the bounds are not enclosed in square brackets.

        Examples
        --------
        >>> ClosedIntervalPattern.parse_string_description('[1,5]')
        (1.0, 5.0)
        """
        if value != 'ø' and ',' in value:
            assert value[0] == '[' and value[-1] == ']', \
                'Only closed intervals are supported within ClosedIntervalPattern. ' \
                f'Change the bounds of interval "{value}" to square brackets to make it close'

        parsed_value = super(ClosedIntervalPattern, cls).parse_string_description(value)
        return parsed_value[0][0], parsed_value[1][0]

    @classmethod
    def preprocess_value(cls, value) -> PatternValueType:
        """
        Preprocess the value before storing it in the closed interval pattern.

        Parameters
        ----------
        value:
            The value to preprocess.

        Returns
        -------
        value: PatternValueType
            The preprocessed value as a tuple of two floats.

        Raises
        ------
        ValueError
            If the value cannot be preprocessed into a ClosedIntervalPattern.

        Examples
        --------
        >>> ClosedIntervalPattern.preprocess_value([1, 5])
        (1.0, 5.0)
        """
        if isinstance(value, Sequence) and len(value) == 2 and all(isinstance(v, Number) for v in value):
            return float(value[0]), float(value[1])

        try:
            processed_value = super(ClosedIntervalPattern, cls).preprocess_value(value)
            return processed_value[0][0], processed_value[1][0]
        except Exception as e:
            pass

        raise ValueError(f'Value {value} cannot be preprocessed into {cls.__name__}')

    def atomise(self, atoms_configuration: Literal['min', 'max'] = 'min') -> set[Self]:
        """
        Split the pattern into atomic patterns, i.e. the set of one-sided intervals

        Parameters
        ----------
        atoms_configuration: Literal['min', 'max']
            If equals to 'min', return up to 2 atomic patterns each representing a bound of the original interval.
            If equals to 'max', return _all_ less precise one-sided intervals, where the bounds are defined by
            `BoundUniverse` class attribute.

        Returns
        -------
        atomic_patterns: set[Self]
            The set of atomic patterns, i.e. the set of unsplittable patterns whose join equals to the pattern.


        Notes
        -----
        Speaking in terms of Ordered Set Theory:
        We say that every pattern can be represented as the join of a subset of atomic patterns,
        that are join-irreducible elements of the lattice of all patterns.

        Considering the set of atomic patterns as a partially ordered set (where the order follows the order on patterns),
        every pattern can be represented by an _antichain_ of atomic patterns (when `atoms_configuration` = 'min'),
        and by an *order ideal* of atomic patterns (when `atoms_configuration` = 'max').

        """
        return {self.__class__(atom) for atom in super().atomise(atoms_configuration)
                if atom.is_lower_bound_closed and atom.is_upper_bound_closed}


class NgramSetPattern(Pattern):
    """
    A class representing a set of n-grams as a pattern.

    Attributes
    ..........
    PatternValueType:
        The type of the pattern's value, which is a frozenset of tuples.
    StopWords: set[str]
        A set of exclusively stop words to be excluded from the n-grams.
        But if a set has both stop words and non stop words it is kept in the analysis

    Properties
    ..........
    atomic_patterns
        Return the set of all less precise patterns that cannot be obtained by intersection of other patterns.
    min_pattern
        Return the minimal possible pattern for the n-gram set pattern.    
    
    Examples
    --------
    >>> p1 = NgramSetPattern({('hello', 'world')})  # explicit way to define a pattern with a single "hello world" ngram
    >>> p2 = NgramSetPattern('hello world')  # simplified way to define {'hello world'} pattern
    >>> print(p1 == p2)
    True
    >>> p3 = NgramSetPattern( {('hello', 'world'), ('foo',), ('bar',) } )  # explicit way to define a pattern with 3 ngrams
    >>> p4 = NgramSetPattern(['hello world', 'foo', 'bar'])  # simplified way to define a pattern with 3 ngrams 
    >>> print(p3 == p4)
    True
    """
    PatternValueType = frozenset[tuple[str, ...]]
    StopWords: set[str] = frozenset()

    def __repr__(self) -> str:
        """
        Return a string representation of the n-gram set pattern.

        Returns
        -------
        representation: str
            The string representation of the n-gram set pattern.

        Examples
        --------
        >>> p = NgramSetPattern({('hello', 'world')})
        >>> repr(p)
        "{'hello world'}"
        """
        ngrams = sorted(self.value, key=lambda ngram: (-len(ngram), ngram))
        ngrams_verb = [' '.join(ngram) for ngram in ngrams]
        pattern_verb = "{'" + "', '".join(ngrams_verb) + "'}"
        return pattern_verb

    @classmethod
    def parse_string_description(cls, value: str) -> PatternValueType:
        """
        Parse a string description into an n-gram set pattern value.

        Parameters
        ----------
        value: str
            The string description of the n-gram set pattern.

        Returns
        -------
        parsed: PatternValueType
            The parsed n-gram set pattern value.

        Raises
        ------
        ValueError
            If the value cannot be parsed into an NgramSetPattern.

        Examples
        --------
        >>> NgramSetPattern.parse_string_description("{'hello world', 'foo bar'}")
        frozenset({('hello', 'world'), ('foo', 'bar')})
        """
        try:
            parsed_value = super().parse_string_description(value)
        except Exception as e:
            parsed_value = None

        if parsed_value is None and isinstance(value, str):
            parsed_value = [value]

        if parsed_value is not None and isinstance(parsed_value, Collection):
            parsed_value = [v.strip().split(' ') for v in parsed_value]
            return frozenset(map(tuple, parsed_value))

        raise ValueError(f'Value {value} cannot be preprocessed into {cls.__name__}')

    @classmethod
    def preprocess_value(cls, value) -> PatternValueType:
        """
        Preprocess the value before storing it in the n-gram set pattern.

        Parameters
        ----------
        value:
            The value to preprocess.

        Returns
        -------
        value: PatternValueType
            The preprocessed value as a frozenset of tuples.

        Examples
        --------
        >>> NgramSetPattern.preprocess_value(['hello world', 'foo'])
        frozenset({('hello', 'world'), ('foo',)})
        """
        value = [re.sub(r" +", " ", v).strip().split(' ') if isinstance(v, str) else v for v in value]
        value = [ngram for ngram in value if not set(ngram) <= cls.StopWords]
        return frozenset(map(tuple, value))

    def __len__(self) -> int:
        """
        Return the number of n-grams in the n-gram set pattern.

        Returns
        -------
        count: int
            The number of n-grams.

        Examples
        --------
        >>> len(NgramSetPattern([('hello', 'world')]))
        1
        >>> len(NgramSetPattern(['hello', 'world']))
        2
        >>> len(NgramSetPattern([('hello', 'world', 'foo')]))
        1
        >>> len(NgramSetPattern([('hello', 'world'), ('foo')]))
        2
        >>> len(NgramSetPattern(['hello', 'world', 'foo']))
        3
        """
        return len(self.value)

    def __and__(self, other: Self) -> Self:
        """
        Return the intersection of self and another n-gram set pattern.

        Parameters
        ----------
        other: Self
            Another n-gram set pattern to intersect with.

        Returns
        -------
        Self
            The intersected n-gram set pattern.
            The most precise pattern that is less precise than both self and other

        Examples
        --------
        >>> p1 = NgramSetPattern({('hello', 'world')})
        >>> p2 = NgramSetPattern({('hello', 'world'), ('foo',)})
        >>> p1 & p2
        {'hello world'}
        """
        common_ngrams: list[tuple[str, ...]] = []
        for ngram_a in self.value:
            words_pos_a = dict()
            for i, word in enumerate(ngram_a):
                words_pos_a[word] = words_pos_a.get(word, []) + [i]

            for ngram_b in other.value:
                if ngram_a == ngram_b:
                    common_ngrams.append(ngram_a)
                    continue

                for j, word in enumerate(ngram_b):
                    if word not in words_pos_a:
                        continue
                    # word in words_a
                    for i in words_pos_a[word]:
                        ngram_size = next(
                            s for s in range(len(ngram_b)+1)
                            if i+s >= len(ngram_a) or j+s >= len(ngram_b) or ngram_a[i+s] != ngram_b[j+s]
                        )

                        common_ngrams.append(ngram_a[i:i+ngram_size])

        # Delete common n-grams contained in other common n-grams
        common_ngrams = sorted(common_ngrams, key=lambda ngram: len(ngram), reverse=True)
        for i in range(len(common_ngrams)):
            n_ngrams = len(common_ngrams)
            if i == n_ngrams:
                break

            ngram = common_ngrams[i]
            ngrams_to_pop = (j for j in reversed(range(i + 1, n_ngrams))
                             if self._issubngram(common_ngrams[j], ngram))
            for j in ngrams_to_pop:
                common_ngrams.pop(j)

        return self.__class__(frozenset(common_ngrams))

    def __or__(self, other: Self) -> Self:
        """
        Return the union of self and another n-gram set pattern.

        Parameters
        ----------
        other: Self
            Another n-gram set pattern to union with.

        Returns
        -------
        Self
            The union of the two n-gram set patterns.
            The least precise pattern that is more precise than both self and other

        Examples
        --------
        >>> p1 = NgramSetPattern({('hello',)})
        >>> p2 = NgramSetPattern({('world',)})
        >>> p1 | p2
        {'hello', 'world'}
        """
        return self.__class__(self.filter_max_ngrams(self.value | other.value))

    def __sub__(self, other: Self) -> Self:
        """
        Return the NgramSetPattern that contains the items that can be found in self but not in other.

        Parameters
        ----------
        other: Self
            Another n-gram set pattern to subtract from self.

        Returns
        -------
        Self
            The resulting n-gram set pattern after subtraction.

        Examples
        --------
        >>> p1 = NgramSetPattern({('hello',)}, {('world',)})
        >>> p2 = NgramSetPattern({('world',)})
        >>> p1 - p2
        {'hello'}
        """
        if self == other:
            return self.min_pattern

        return self.__class__(self.filter_max_ngrams(self.value - other.value))

    @staticmethod
    def _issubngram(ngram_a: tuple[str], ngram_b: tuple[str])-> bool:
        """
        Check if ngram_a is a sub-ngram of ngram_b.

        Parameters
        ----------
        ngram_a: tuple[str]
            The n-gram to check if it is a sub-ngram.
        ngram_b: tuple[str]
            The n-gram to check against.

        Returns
        -------
        is_sub: bool
            True if ngram_a is a sub-ngram of ngram_b, False otherwise.

        Examples
        --------
        >>> NgramSetPattern._issubngram(('hello',), ('hello', 'world'))
        True
        """
        if len(ngram_a) > len(ngram_b):
            return False
        return any(ngram_b[i:i + len(ngram_a)] == ngram_a for i in range(len(ngram_b) - len(ngram_a) + 1))

    @classmethod
    def filter_max_ngrams(self, ngrams: PatternValueType) -> PatternValueType:
        """
        Filter maximal n-grams from the set of n-grams.

        Parameters
        ----------
        ngrams: PatternValueType
            The set of n-grams to filter.

        Returns
        -------
        maximal_ngrams: PatternValueType
            The filtered set of maximal n-grams, excluding the rest.

        Examples
        --------
        >>> NgramSetPattern.filter_max_ngrams({('hello',), ('hello', 'world')})
        frozenset({('hello', 'world')}) # it filters out hello beacause there is a greater N-gram
        """
        ngrams = sorted(ngrams, key=lambda ngram: len(ngram), reverse=True)
        i = 0
        while i < len(ngrams):
            if any(self._issubngram(ngrams[i], other) for other in ngrams[:i]):
                ngrams.pop(i)
                continue
            i += 1
        return frozenset(ngrams)

    def atomise(self, atoms_configuration: Literal['min', 'max'] = 'min') -> set[Self]:
        """
        Split the pattern into atomic patterns, i.e. the singleton sets of ngrams

        Parameters
        ----------
        atoms_configuration: Literal['min', 'max']
            If set to 'min', then return the set of individual ngrams from the value of the original pattern.
            If set to 'max', the return all sub-ngrams that can only be found in the original pattern.
            Defaults to 'min'.

        Returns
        -------
        atomic_patterns: set[Self]
            The set of atomic patterns, i.e. the set of unsplittable patterns whose join equals to the pattern.


        Notes
        -----
        Speaking in terms of Ordered Set Theory:
        We say that every pattern can be represented as the join of a subset of atomic patterns,
        that are join-irreducible elements of the lattice of all patterns.

        Considering the set of atomic patterns as a partially ordered set (where the order follows the order on patterns),
        every pattern can be represented by an _antichain_ of atomic patterns (when `atoms_configuration` = 'min'),
        and by an *order ideal* of atomic patterns (when `atoms_configuration` = 'max').

        """
        if atoms_configuration == 'min':
            return {self.__class__([ngram]) for ngram in self.value}

        # atoms_configuration == 'max':
        atoms = set()
        for ngram in self.value:
            for atom_size in range(1, len(ngram)+1):
                atoms |= {ngram[i:i+atom_size] for i in range(len(ngram)-atom_size+1)}

        return {self.__class__([v]) for v in atoms}


    @property
    def atomic_patterns(self) -> set[Self]:
        """
        Return the set of every individual sub-ngram for each ngram from the given pattern.

        For an NgramSetPattern an atomic pattern is a set containing just one ngram.

        Returns
        -------
        atoms: set[Self]
            A set of atomic patterns derived from the n-grams.

        Examples
        --------
        >>> p1 = NgramSetPattern({('hello', 'world')})
        >>> p1.atomic_patterns
        {{'hello world'}, {'hello'}, {'world'}}
        >>> p2 = NgramSetPattern(["hello world !", "foo"])
        >>> p2.atomic_patterns
        {{'hello world !'}, {'foo'}, {'hello'}, {'!'}, {'hello world'}, {'world !'}, {'world'}}
        """
        return super().atomic_patterns

    @classmethod
    def get_min_pattern(cls) -> Optional[Self]:
        """
        Return the minimal possible pattern for the n-gram set pattern, i.e. empty NgramSet

        Returns
        -------
        min: Optional[Self]
            The minimal n-gram set pattern, which is an empty NgramSet
        
        Examples
        --------
        >>> NgramSetPattern.get_min_pattern
        NgramSetPattern(set())
        """
        return cls([])


class CartesianPattern(Pattern):
    """
    A class representing a Cartesian product of multiple dimensions as a pattern.

    Attributes
    ..........
    PatternValueType:
        The type of the pattern's value, which is a frozendict mapping dimension names to Pattern instances.
    DimensionTypes: dict[str, Type[Pattern]]
        Optional mapping from dimension names to specific Pattern types for parsing.
    
    Properties
    ..........
    atomic_patterns
        Return the set of all less precise patterns that cannot be obtained by intersection of other patterns.
    min_pattern
        Return the minimal possible pattern for the Cartesian pattern.
    max_pattern
        Return the maximal atomic patterns of the Cartesian pattern.
    maximal_atoms
        Return the maximal atomic patterns of the Cartesian pattern.
    """
    PatternValueType = frozendict[str, Pattern]
    DimensionTypes: dict[str, Type[Pattern]] = None  # required for parsing stings of dimensional patterns

    def __repr__(self) -> str:
        """
        Return a string representation of the Cartesian pattern.

        Returns
        -------
        representation: str
            A string representing the dictionary form of the Cartesian pattern.

        Examples
        --------
        >>> repr(CartesianPattern({'x': Pattern('A'), 'y': Pattern('B')}))
        "{'x': Pattern('A'), 'y': Pattern('B')}"
        """
        return repr(dict(self.value))

    @classmethod
    def preprocess_value(cls, value) -> PatternValueType:
        """
        Preprocess the value before storing it in the Cartesian pattern.

        Parameters
        ----------
        value: dict
            A dictionary mapping dimension names to pattern descriptions or Pattern instances.

        Returns
        -------
        value: PatternValueType
            The preprocessed frozendict of dimension names to Pattern instances.

        Raises
        ------
        AssertionError
            If required dimension types are missing or unprocessable.

        Examples
        --------
        >>> class PersonPattern(CartesianPattern):
        ...     DimensionTypes = {
        ...         'age': IntervalPattern,
        ...         'name': NgramSetPattern,
        ...         'personal qualities': ItemSetPattern
        ...     }
        >>> PersonPattern.preprocess_value({
        ...    'age': 20,
        ...    'name': 'Jean-Francois Martin',
        ...    'personal qualities': ['Ambitious', 'Approachable', 'Articulate']
        ... })
        frozendict({
            'age': IntervalPattern((20.0, 20.0)),
            'name': NgramSetPattern({('Jean-Francois',), ('Martin',)}),
            'personal qualities': ItemSetPattern({'Ambitious', 'Approachable', 'Articulate'})
        })
        """
        if cls.DimensionTypes is not None:
            value = {k: v if isinstance(v, cls.DimensionTypes[k]) else cls.DimensionTypes[k](v)
                     for k, v in value.items()}
        else:  # cls.DimensionTypes are not defined
            non_processed_dimensions = {k for k, v in value.items() if not isinstance(v, Pattern)}
            assert not non_processed_dimensions, \
                f"Cannot preprocess dimensions {non_processed_dimensions} of CartesianPattern given by {value}. " \
                f"Either convert these dimensional descriptions to Pattern classes " \
                f"or define `CartesianPattern.DimensionTypes` class variable."

        value = dict(value)
        for k in list(value):
            if value[k].min_pattern is not None and value[k] == value[k].min_pattern:
                del value[k]
        keys_order = sorted(value)
        return frozendict({k: value[k] for k in keys_order})

    def __and__(self, other: Self) -> Self:
        """
        Return the intersection of self and another Cartesian pattern. Or the intersection of patterns for every dimension.

        Parameters
        ----------
        other: Self
            Another Cartesian pattern to intersect with.

        Returns
        -------
        Self
            The most precise pattern that is less precise than both self and other.

        Examples
        --------
        >>> class PersonPattern(CartesianPattern):
            DimensionTypes = {
                'age': IntervalPattern,
                'name': NgramSetPattern
                }
        >>> p1 = PersonPattern({'age': "[20, 40]", 'name': "John Smith"})
        >>> p2 = PersonPattern({'age': "[30, 50]", 'name': "Smith"})
        >>> p1 & p2
        {'age': [20.0, 50.0], 'name': {'Smith'}}
        """
        return self.__class__({k: self.value[k] & other.value[k] for k in set(self.value) & set(other.value)})

    def __or__(self, other: Self) -> Self:
        """
        Return the union of self and another Cartesian pattern.

        Parameters
        ----------
        other: Self
            Another Cartesian pattern to union with.

        Returns
        -------
        Self
            The least precise pattern that is more precise than both self and other.

        Examples
        --------
        >>> class PersonPattern(CartesianPattern):
        ...    DimensionTypes = {
        ...        'age': IntervalPattern,
        ...        'name': NgramSetPattern
        ...    }
        >>> p1 = PersonPattern({'age': "[20, 40]", 'name': "John Smith"})
        >>> p2 = PersonPattern({'age': "[30, 50]", 'name': "Smith"})
        >>> p1 | p2
        {'age': [30.0, 40.0], 'name': {'John Smith'}}
        """
        keys_a, keys_b = set(self.value), set(other.value)
        left_keys, common_keys, right_keys = keys_a-keys_b, keys_a & keys_b, keys_b - keys_a
        join = {k: self.value[k] | other.value[k] for k in common_keys}
        join |= {k: self.value[k] for k in left_keys} | {k: other.value[k] for k in right_keys}
        return self.__class__(join)

    def __sub__(self, other: Self) -> Self:
        """
        Return the CartesianPattern that contains the items that can be found in self but not in other.

        Parameters
        ----------
        other: Self
            Another Cartesian pattern to subtract from self.

        Returns
        -------
        Self
            The least precise pattern such that (self - other) | other == self.

        Examples
        --------
        >>> class PersonPattern(CartesianPattern):
        ...    DimensionTypes = {
        ...        'age': IntervalPattern,
        ...        'name': NgramSetPattern
        ...    }
        >>> p1 = PersonPattern({'age': "[20, 40]", 'name': "John Smith"})
        >>> p2 = PersonPattern({'age': "[20, 40]", 'name': "John Smith"}) 
        >>> p1 - p2
        {'name': {'John Smith'}}
        """
        if self == other:
            return self.min_pattern

        return self.__class__({k: (v - other.value[k]) if k in other.value else v for k, v in self.value.items()})


    def atomise(self, atoms_configuration: Literal['min', 'max'] = 'min') -> set[Self]:
        """
        Split the pattern into atomic patterns, i.e. the union of atomic patterns computed for each dimension

        Parameters
        ----------
        atoms_configuration: Literal['min', 'max']
            Whether to output the maximal set of atomic patterns for each dimension (if set to 'max') or
            the minimal set of atomic patterns for each dimension (if set to 'min').
            Defaults to 'min'.

        Returns
        -------
        atomic_patterns: set[Self]
            The set of atomic patterns, i.e. the set of unsplittable patterns whose join equals to the pattern.


        Notes
        -----
        Speaking in terms of Ordered Set Theory:
        We say that every pattern can be represented as the join of a subset of atomic patterns,
        that are join-irreducible elements of the lattice of all patterns.

        Considering the set of atomic patterns as a partially ordered set (where the order follows the order on patterns),
        every pattern can be represented by an _antichain_ of atomic patterns (when `atoms_configuration` = 'min'),
        and by an *order ideal* of atomic patterns (when `atoms_configuration` = 'max').

        """
        return {self.__class__({k: atom}) for k, pattern in self.value.items()
                for atom in pattern.atomise(atoms_configuration)}

    @property
    def atomic_patterns(self) -> set[Self]:
        """
        Return the set of atomic patterns of a CartesianPattern which is the union of sets of atomic patterns per its every dimension

        Returns
        -------
        atoms: set[Self]
            A set of atomic Cartesian patterns.

        Examples
        --------
        >>> class PersonPattern(CartesianPattern):
        ...    DimensionTypes = {
        ...        'age': IntervalPattern,
        ...        'name': NgramSetPattern
        ...    }
        >>> p1 = PersonPattern({'age': "[20, 40]", 'name': "John Smith"})
        >>> atoms = p1.atomic_patterns
        >>> for atom in atoms:
        ...     print(atom)
        {'name': {'John Smith'}}
        {'name': {'Smith'}}
        {'name': {'John'}}
        {'age': <= 40.0}
        {'age': >= 20.0}
        {}
        {'name': {'John Smith'}}
        """
        return super().atomic_patterns

    def __len__(self) -> int:
        """
        Return the minimal number of atomic patterns required to generate the Cartesian pattern.

        Returns
        -------
        count: int
            The total number of atomic subpatterns across all dimensions.

        Examples
        --------
        >>> class PersonPattern(CartesianPattern):
        ...    DimensionTypes = {
        ...        'age': IntervalPattern,
        ...        'name': NgramSetPattern
        ...    }
        >>> p1 = PersonPattern({'age': "[20, 40]", 'name': "John Smith"})
        >>> len(p1)
        3
        """
        return sum(len(subpattern) for subpattern in self.value.values())

    @classmethod
    def get_min_pattern(cls) -> Optional[Self]:
        """
        Return the minimal possible pattern for the Cartesian pattern that contains min patterns per every dimension.

        Returns
        -------
        min: Optional[Self]
            The minimal Cartesian pattern or None if any subpattern has no minimum.

        Examples
        --------
        >>> class PersonPattern(CartesianPattern):
        ...    DimensionTypes = {
        ...        'age': IntervalPattern,
        ...        'name': NgramSetPattern
        ... }
        >>> PersonPattern.get_min_pattern()
        {}  # Stands for `PersonPattern({'age': '[-inf, +inf]', 'name': set()})`
        """
        if cls.DimensionTypes is None:
            return None

        min_patterns = {dim: dtype.get_min_pattern() for dim, dtype in cls.DimensionTypes.items()}
        if any(v is None for v in min_patterns.values()):
            return None

        return cls(min_patterns)

    @classmethod
    def get_max_pattern(cls) -> Optional[Self]:
        """
        Return the maximal possible pattern for the Cartesian pattern that contains max patterns per every dimension

        Returns
        -------
        max: Optional[Self]
            The maximal Cartesian pattern or None if any subpattern has no maximum.

        Examples
        --------
        >>> class PersonPattern(CartesianPattern):
        ...    DimensionTypes = {
        ...        'age': IntervalPattern,
        ...        'name': NgramSetPattern
        ...    }
        >>> PersonPattern.get_max_pattern()
        None  # because max_pattern for dimension 'name' is not defined
        """
        if cls.DimensionTypes is None:
            return None
        max_patterns_per_dim = {dimension: dtype.get_max_pattern() for dimension, dtype in cls.DimensionTypes.items()}
        if any(max_pattern is None for max_pattern in max_patterns_per_dim.values()):
            return None

        return cls(max_patterns_per_dim)


    @property
    def maximal_atoms(self) -> Optional[set[Self]]:
        """
        Return the maximal atomic patterns of the Cartesian pattern.

        Returns
        -------
        max_atoms: Optional[set[Self]]
            A set of maximal atomic Cartesian patterns.

        Examples
        --------
        >>> class PersonPattern(CartesianPattern):
        ...     DimensionTypes = {
        ...         'age': IntervalPattern,
        ...         'name': NgramSetPattern,
        ...         'personal qualities': ItemSetPattern
        ...     }
        >>> p = PersonPattern({
        ...    'age': "[25, 35]",
        ...    'name': "Alice Johnson",
        ...    'personal qualities': ['Thoughtful', 'Curious']
        ... })
        >>> for atom in p.maximal_atoms:
        ...     print(atom)
        {'age': ø}
        """
        max_atoms = set()
        for k, pattern in self.value.items():
            if pattern.maximal_atoms is None:
                continue
            max_atoms |= {self.__class__({k: atom}) for atom in pattern.maximal_atoms}
        return max_atoms

    def __getitem__(self, item: str) -> Pattern:
        """
        Access a sub-pattern in a specific dimension.

        Parameters
        ----------
        item: str
            The dimension name.

        Returns
        -------
        pattern: Pattern
            The Pattern instance corresponding to the dimension.

        Examples
        --------
        >>> class PersonPattern(CartesianPattern):
        ...    DimensionTypes = {
        ...    'age': IntervalPattern,
        ...    'name': NgramSetPattern,
        ...    'personal qualities': ItemSetPattern
        ...    }
        >>> p = PersonPattern({
        ...    'age': "[25, 35]",
        ...    'name': "Alice Johnson",
        ...    'personal qualities': ['Thoughtful', 'Curious']
        ... })
        >>> print(p['age'])
        [25.0, 35.0]
        >>> print(p['name'])
        {'Alice Johnson'}
        """
        return self.value[item]
