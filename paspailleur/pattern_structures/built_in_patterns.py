import math
from typing import Self, Collection, Optional, Sequence, Type
from numbers import Number
from frozendict import frozendict
import re


from .pattern import Pattern


class ItemSetPattern(Pattern):
    PatternValueType = frozenset

    @property
    def value(self) -> PatternValueType:
        return self._value

    def __repr__(self) -> str:
        return repr(set(self.value))

    def __len__(self) -> int:
        """Minimal number of atomic patterns required to generate the pattern

        For ItemSetPattern, the length of the pattern is the length of its pattern.value
        """
        return len(self.value)

    def __sub__(self, other: Self) -> Self:
        """Return self - other, i.e. the least precise pattern s.t. (self-other)|other == self

         (if it's not possible, return self)"""
        return self.__class__(self.value - other.value)

    @classmethod
    def parse_string_description(cls, value: str) -> PatternValueType:
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
        return frozenset(value)

    @property
    def atomic_patterns(self) -> set[Self]:
        """Return the set of all less precise patterns that cannot be obtained by intersection of other patterns"""
        return {self.__class__({v}) for v in self.value}

    @property
    def min_pattern(self) -> Self:
        """Minimal possible pattern, the sole one per Pattern class. `None` if undefined"""
        return self.__class__(frozenset())


class CategorySetPattern(ItemSetPattern):
    PatternValueType = frozenset
    Universe: Optional[frozenset] = None  # The set of all possible categories

    def __and__(self, other):
        return self.__class__(self.value | other.value)

    def __or__(self, other):
        return self.__class__(self.value & other.value)

    def __repr__(self) -> str:
        repr_negative = self.Universe is not None and len(self.value) > len(self.Universe) / 2
        s = set(self.Universe) - self.value if repr_negative else self.value
        s = repr(set(s))
        if repr_negative:
            s = f"NOT({s})"
        return s

    def __sub__(self, other):
        if self.min_pattern is not None and self == other:
            return self.min_pattern
        return self.__class__(self.value)

    @property
    def atomic_patterns(self) -> set[Self]:
        assert self.min_pattern is not None,\
            f"Atomic patterns of {self.__class__} class cannot be computed without predefined min_pattern value. " \
            f"The proposed solution is to inherit a new class from the current class and " \
            f"explicitly specify the value of min_pattern."

        leftout_vals = self.min_pattern.value - self.value
        return {self.__class__(self.min_pattern.value-{v}) for v in leftout_vals}

    @property
    def min_pattern(self) -> Optional[Self]:
        """Minimal possible pattern, the sole one per Pattern class. `None` if undefined"""
        if self.Universe is None:
            return None
        return self.__class__(self.Universe)

    @property
    def max_pattern(self) -> Self:
        """Maximal possible pattern, the sole one per Pattern class. `None` if undefined"""
        return self.__class__(frozenset())

    def __len__(self) -> int:
        """Minimal number of atomic patterns required to generate the pattern

        For CategorySetPattern, the length of a pattern is the categories the pattern.value does _not_ include
        """
        assert self.min_pattern is not None, f"Length of pattern of {self.__class__} " \
                                             f"class cannot be computed without the predefined min_pattern value."
        return len(self.min_pattern.value) - len(self.value)


class IntervalPattern(Pattern):
    # PatternValue semantics: ((lower_bound, is_closed), (upper_bound, is_closed))
    PatternValueType = tuple[tuple[float, bool], tuple[float, bool]]
    BoundsUniverse: list[float] = None

    @property
    def lower_bound(self) -> float:
        return self.value[0][0]

    @property
    def is_lower_bound_closed(self) -> bool:
        return self.value[0][1]

    @property
    def upper_bound(self) -> float:
        return self.value[1][0]

    @property
    def is_upper_bound_closed(self) -> bool:
        return self.value[1][1]

    def __repr__(self) -> str:
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
        """Minimal number of atomic patterns required to generate the pattern

        For IntervalPattern, the length of a pattern is the number of non-infinite bounds
        """
        return int(self.lower_bound != -math.inf) + int(self.upper_bound != math.inf)

    @classmethod
    def parse_string_description(cls, value: str) -> PatternValueType:
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
        if isinstance(value, Number):
            value = (value, value)

        if len(value) == 2 and all(isinstance(v, Number) for v in value):
            value = (float(value[0]), True), (float(value[1]), True)

        lb, closed_lb = value[0]
        rb, closed_rb = value[1]

        is_contradictive = (rb < lb) or (lb == rb and not (closed_lb and closed_rb))
        if is_contradictive:
            lb, rb = 0, 0

        if cls.BoundsUniverse is not None:
            lb = max(b for b in cls.BoundsUniverse if b <= lb) if lb > -math.inf else lb
            rb = min(b for b in cls.BoundsUniverse if rb <= b) if rb < math.inf else rb

        return (float(lb), bool(closed_lb)), (float(rb), bool(closed_rb))

    def __and__(self, other: Self) -> Self:
        """Return self & other, i.e. the most precise pattern that is less precise than both self and other"""
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
        """Return self | other, i.e. the least precise pattern that is more precise than both self and other"""
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
        """Return self - other, i.e. the least precise pattern s.t. (self-other)|other == self"""
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

    @property
    def atomic_patterns(self) -> set[Self]:
        """Return the set of all less precise patterns that cannot be obtained by intersection of other patterns"""
        if self.value == self.max_pattern.value:
            return {self.max_pattern}

        atoms = [
            ((-math.inf, True), (math.inf, True)),
            ((-math.inf, True), (self.upper_bound, self.is_upper_bound_closed)),
            ((self.lower_bound, self.is_lower_bound_closed), (math.inf, True))
        ]
        if not self.is_upper_bound_closed:
            atoms.append(tuple([(-math.inf, True), (self.upper_bound, True)]))
        if not self.is_lower_bound_closed:
            atoms.append(tuple([(self.lower_bound, True), (math.inf, True)]))

        return {self.__class__(v) for v in atoms}

    @property
    def min_pattern(self) -> Self:
        """Minimal possible pattern, the sole one per Pattern class. `None` if undefined"""
        return self.__class__(((-math.inf, True), (math.inf, True)))

    @property
    def max_pattern(self) -> Self:
        """Minimal possible pattern, the sole one per Pattern class. `None` if undefined"""
        return self.__class__("ø")

    @property
    def maximal_atoms(self) -> Optional[set[Self]]:
        return {self.max_pattern}


class ClosedIntervalPattern(IntervalPattern):
    PatternValueType = tuple[float, float]

    @property
    def lower_bound(self) -> float:
        return self._value[0]

    @property
    def upper_bound(self) -> float:
        return self._value[1]

    @property
    def is_lower_bound_closed(self) -> bool:
        return True

    @property
    def is_upper_bound_closed(self) -> bool:
        return True

    @classmethod
    def parse_string_description(cls, value: str) -> PatternValueType:
        if value != 'ø' and ',' in value:
            assert value[0] == '[' and value[-1] == ']', \
                'Only closed intervals are supported within ClosedIntervalPattern. ' \
                f'Change the bounds of interval "{value}" to square brackets to make it close'

        parsed_value = super(ClosedIntervalPattern, cls).parse_string_description(value)
        return parsed_value[0][0], parsed_value[1][0]

    @classmethod
    def preprocess_value(cls, value) -> PatternValueType:
        if isinstance(value, Sequence) and len(value) == 2 and all(isinstance(v, Number) for v in value):
            return float(value[0]), float(value[1])

        try:
            processed_value = super(ClosedIntervalPattern, cls).preprocess_value(value)
            return processed_value[0][0], processed_value[1][0]
        except Exception as e:
            pass

        raise ValueError(f'Value {value} cannot be preprocessed into {cls.__name__}')


class NgramSetPattern(Pattern):
    PatternValueType = frozenset[tuple[str, ...]]
    StopWords: set[str] = frozenset()

    def __repr__(self) -> str:
        ngrams = sorted(self.value, key=lambda ngram: (-len(ngram), ngram))
        ngrams_verb = [' '.join(ngram) for ngram in ngrams]
        pattern_verb = "{'" + "', '".join(ngrams_verb) + "'}"
        return pattern_verb

    @classmethod
    def parse_string_description(cls, value: str) -> PatternValueType:
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
        value = [re.sub(r" +", " ", v).strip().split(' ') if isinstance(v, str) else v for v in value]
        value = [ngram for ngram in value if not set(ngram) <= cls.StopWords]
        return frozenset(map(tuple, value))

    def __len__(self) -> int:
        """Minimal number of atomic patterns required to generate the pattern

        For NgramSetPattern, the length of a pattern is the number of ngram it contains
        """
        return len(self.value)

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

    def __sub__(self, other: Self) -> Self:
        """Return self - other, i.e. the least precise pattern s.t. (self-other)|other == self"""
        if self == other:
            return self.min_pattern

        return self.__class__(self.filter_max_ngrams(self.value - other.value))

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


class CartesianPattern(Pattern):
    PatternValueType = frozendict[str, Pattern]
    DimensionTypes: dict[str, Type[Pattern]] = None  # required for parsing stings of dimensional patterns

    def __repr__(self) -> str:
        return repr(dict(self.value))

    @classmethod
    def preprocess_value(cls, value) -> PatternValueType:
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
        return self.__class__({k: self.value[k] & other.value[k] for k in set(self.value) & set(other.value)})

    def __or__(self, other: Self) -> Self:
        keys_a, keys_b = set(self.value), set(other.value)
        left_keys, common_keys, right_keys = keys_a-keys_b, keys_a & keys_b, keys_b - keys_a
        join = {k: self.value[k] | other.value[k] for k in common_keys}
        join |= {k: self.value[k] for k in left_keys} | {k: other.value[k] for k in right_keys}
        return self.__class__(join)

    def __sub__(self, other: Self) -> Self:
        """Return self - other, i.e. the least precise pattern s.t. (self-other)|other == self"""
        if self == other:
            return self.min_pattern

        return self.__class__({k: (v - other.value[k]) if k in other.value else v for k, v in self.value.items()})

    @property
    def atomic_patterns(self) -> set[Self]:
        return {self.__class__({k: atom}) for k, pattern in self.value.items() for atom in pattern.atomic_patterns}

    def __len__(self) -> int:
        """Minimal number of atomic patterns required to generate the pattern

        For CartesianPattern, the length of a pattern is the sum of lengths of all 'dimensional' subpatterns
        """
        return sum(len(subpattern) for subpattern in self.value.values())

    @property
    def min_pattern(self) -> Optional[Self]:
        if any(p.min_pattern is None for p in self.value.values()):
            return None

        return self.__class__({k: pattern.min_pattern for k, pattern in self.value.items()})

    @property
    def max_pattern(self) -> Optional[Self]:
        if any(p.max_pattern is None for p in self.value.values()):
            return None

        return self.__class__({k: pattern.max_pattern for k, pattern in self.value.items()})

    @property
    def maximal_atoms(self) -> Optional[set[Self]]:
        max_atoms = set()
        for k, pattern in self.value.items():
            if pattern.maximal_atoms is None:
                continue
            max_atoms |= {self.__class__({k: atom}) for atom in pattern.maximal_atoms}
        return max_atoms

    def __getitem__(self, item: str) -> Pattern:
        return self.value[item]
