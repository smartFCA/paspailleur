import math
from typing import Self, Union, Collection, Optional
from numbers import Number
import re

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

    @property
    def atomic_patterns(self) -> set[Self]:
        """Return the set of all less precise patterns that cannot be obtained by intersection of other patterns"""
        return {self.__class__({v}) for v in self.value}

    @property
    def min_pattern(self) -> Optional[Self]:
        """Minimal possible pattern, the sole one per Pattern class. `None` if undefined"""
        return self.__class__(frozenset())

    def __hash__(self):
        return hash(self._value)


class IntervalPattern(Pattern):
    # PatternValue semantics: ((lower_bound, is_closed), (upper_bound, is_closed))
    PatternValueType = tuple[tuple[float, bool], tuple[float, bool]]

    def __init__(self, value: Union[PatternValueType, str]):
        super().__init__(None)
        if isinstance(value, str):
            if value != 'ø':
                lb, ub = map(str.strip, value[1:-1].replace('∞', 'inf').split(','))
                closed_lb, closed_ub = value[0] == '[', value[-1] == ']'
            else:
                lb, ub = 0, 0
                closed_lb, closed_ub = False, False
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
        if self == self.max_pattern:
            str_descr = 'ø'
        else:
            lbound_sign = '[' if self._is_closed_lower_bound else '('
            ubound_sign = ']' if self._is_closed_upper_bound else ')'
            str_descr = f"{lbound_sign}{self._lower_bound}, {self._upper_bound}{ubound_sign}"
        return f"IntervalPattern({str_descr})"

    def __and__(self, other: Self) -> Self:
        """Return self & other, i.e. the most precise pattern that is less precise than both self and other"""
        if self == self.min_pattern or other == self.min_pattern:
            return self.min_pattern

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
        if self == self.max_pattern or other == self.max_pattern:
            return self.max_pattern

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
        if (lbound > ubound) \
                or (lbound == ubound and not (closed_lb and closed_ub)):
            return self.max_pattern
        return self.__class__(new_value)

    def __sub__(self, other: Self) -> Self:
        """Return self - other, i.e. the least precise pattern s.t. (self-other)|other == self"""
        # TODO: Find out how to implement this. And should it be implemented
        raise NotImplementedError

    @property
    def atomic_patterns(self) -> set[Self]:
        """Return the set of all less precise patterns that cannot be obtained by intersection of other patterns"""
        atoms = [
            ((-math.inf, True), (math.inf, True)),
            ((-math.inf, True), (self._upper_bound, self._is_closed_upper_bound)),
            ((self._lower_bound, self._is_closed_lower_bound), (math.inf, True))
        ]
        if not self._is_closed_upper_bound:
            atoms.append(tuple([(-math.inf, True), (self._upper_bound, True)]))
        if not self._is_closed_lower_bound:
            atoms.append(tuple([(self._lower_bound, True), (math.inf, True)]))

        return {self.__class__(v) for v in atoms}

    @property
    def min_pattern(self) -> Optional[Self]:
        """Minimal possible pattern, the sole one per Pattern class. `None` if undefined"""
        return self.__class__("[-inf, +inf]")

    @property
    def max_pattern(self) -> Optional[Self]:
        """Minimal possible pattern, the sole one per Pattern class. `None` if undefined"""
        return self.__class__("ø")


class ClosedIntervalPattern(IntervalPattern):
    PatternValueType = tuple[float, float]

    def __init__(self, value: Union[PatternValueType, str]):
        if isinstance(value, str):
            if value != 'ø':
                assert value[0] == '[' and value[-1] == ']', \
                    'Only closed intervals are supported within ClosedIntervalPattern. ' \
                    'Change the bounds of interval {value} to square brackets to make it close'
        else:
            # Use this to accomodate the functions of the parent class
            value = [(v, True) if isinstance(v, Number) else v for v in value]

        super().__init__(value)

    @property
    def value(self) -> PatternValueType:
        return self._lower_bound, self._upper_bound

    def __repr__(self) -> str:
        return super().__repr__().replace('Interval', 'ClosedInterval')

    @property
    def atomic_patterns(self) -> set[Self]:
        """Return the set of all less precise patterns that cannot be obtained by intersection of other patterns"""
        atoms = [
            (-math.inf, math.inf),
            (-math.inf, self._upper_bound),
            (self._lower_bound, math.inf)
        ]
        return {self.__class__(v) for v in atoms}


class NgramSetPattern(Pattern):
    PatternValueType = frozenset[tuple[str, ...]]

    def __init__(self, value: Union[PatternValueType, Collection[str]]):
        value = [re.sub(r" +", " ", v).strip().split(' ') if isinstance(v, str) else v for v in value]
        super().__init__(frozenset(map(tuple, value)))

    def __repr__(self) -> str:
        ngrams = sorted(self.value, key=lambda ngram: (-len(ngram), ngram))
        ngrams_verb = [' '.join(ngram) for ngram in ngrams]
        pattern_verb = "{'" + "', '".join(ngrams_verb) + "'}"
        return f"NgramSetPattern({pattern_verb})"

    def __and__(self, other: Self) -> Self:
        """Return self & other, i.e. the most precise pattern that is less precise than both self and other"""
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
        """Return self | other, i.e. the least precise pattern that is more precise than both self and other"""
        return self.__class__(self.filter_max_ngrams(self.value | other.value))

    @staticmethod
    def _issubngram(ngram_a: tuple[str], ngram_b: tuple[str]):
        if len(ngram_a) > len(ngram_b):
            return False
        return any(ngram_b[i:i + len(ngram_a)] == ngram_a for i in range(len(ngram_b) - len(ngram_a) + 1))

    @classmethod
    def filter_max_ngrams(self, ngrams: PatternValueType) -> PatternValueType:
        ngrams = sorted(ngrams, key=lambda ngram: len(ngram), reverse=True)
        i = 0
        while i < len(ngrams):
            if any(self._issubngram(ngrams[i], other) for other in ngrams[:i]):
                ngrams.pop(i)
                continue
            i += 1
        return frozenset(ngrams)

    @property
    def atomic_patterns(self) -> set[Self]:
        """Return the set of all less precise patterns that cannot be obtained by intersection of other patterns"""
        atoms = set()

        for ngram in self.value:
            for atom_size in range(1, len(ngram)+1):
                atoms |= {ngram[i:i+atom_size] for i in range(len(ngram)-atom_size+1)}

        return {self.__class__([v]) for v in atoms}

    @property
    def min_pattern(self) -> Optional[Self]:
        """Minimal possible pattern, the sole one per Pattern class. `None` if undefined"""
        return self.__class__([])
