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
