import math
from typing import Self, Union, Collection, Optional, Sequence
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
        return f"ItemSetPattern({set(self.value)})"

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
    def min_pattern(self) -> Optional[Self]:
        """Minimal possible pattern, the sole one per Pattern class. `None` if undefined"""
        return self.__class__(frozenset())


class IntervalPattern(Pattern):
    # PatternValue semantics: ((lower_bound, is_closed), (upper_bound, is_closed))
    PatternValueType = tuple[tuple[float, bool], tuple[float, bool]]

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
            str_descr = 'ø'
        else:
            lbound_sign = '[' if self.is_lower_bound_closed else '('
            ubound_sign = ']' if self.is_upper_bound_closed else ')'
            str_descr = f"{lbound_sign}{self.lower_bound}, {self.upper_bound}{ubound_sign}"
        return f"IntervalPattern({str_descr})"

    @classmethod
    def parse_string_description(cls, value: str) -> PatternValueType:
        if value == 'ø':
            return (0, False), (0, False)

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
            return (float(value), True), (float(value), True)

        lb, closed_lb = value[0]
        rb, closed_rb = value[1]

        is_contradictive = (rb < lb) or (lb == rb and not (closed_lb and closed_rb))
        if is_contradictive:
            lb, rb = 0, 0

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
        # TODO: Find out how to implement this. And should it be implemented
        raise NotImplementedError

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
    def min_pattern(self) -> Optional[Self]:
        """Minimal possible pattern, the sole one per Pattern class. `None` if undefined"""
        return self.__class__("[-inf, +inf]")

    @property
    def max_pattern(self) -> Optional[Self]:
        """Minimal possible pattern, the sole one per Pattern class. `None` if undefined"""
        return self.__class__("ø")


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

    def __repr__(self) -> str:
        return super().__repr__().replace('Interval', 'ClosedInterval')

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

    def __repr__(self) -> str:
        ngrams = sorted(self.value, key=lambda ngram: (-len(ngram), ngram))
        ngrams_verb = [' '.join(ngram) for ngram in ngrams]
        pattern_verb = "{'" + "', '".join(ngrams_verb) + "'}"
        return f"NgramSetPattern({pattern_verb})"

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
        return frozenset(map(tuple, value))

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


class CartesianPattern(Pattern):
    PatternValueType = frozendict[str, Pattern]

    def __repr__(self) -> str:
        return "CartesianPattern("+repr(dict(self.value))+")"

    @classmethod
    def preprocess_value(cls, value) -> PatternValueType:
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

    def __sub__(self, other):
        raise NotImplementedError

    @property
    def atomic_patterns(self) -> set[Self]:
        return {self.__class__({k: atom}) for k, pattern in self.value.items() for atom in pattern.atomic_patterns}

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
