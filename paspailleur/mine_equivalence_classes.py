from .pattern_structures import AbstractPS
from bitarray import frozenbitarray as fbarray


def list_concepts_via_Lindig(data: list['PatternDescription'], pattern_structure: AbstractPS)\
        -> tuple[list[fbarray], list['PatternDescription']]:
    """List extents and intents of pattern concepts from `data` described by `pattern_structure`"""
    raise NotImplementedError
