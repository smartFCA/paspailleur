from .pattern import Pattern


class ItemSetPattern(Pattern):
    PatternValueType = frozenset

    def __init__(self, value):
        super().__init__(frozenset(value))

    @property
    def value(self):
        return set(self._value)

    def __repr__(self):
        return f"ItemSetPattern({self.value})"
