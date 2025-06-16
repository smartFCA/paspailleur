from typing import Union, TypeVar

from paspailleur.pattern_structures.pattern import Pattern
from paspailleur.pattern_structures import built_in_patterns as bip

TPattern = TypeVar('TPattern', bound=Pattern)


def pattern_factory(pattern_class: Union[type[TPattern], str], pattern_name: str = 'FactoredPattern', **kwargs) -> type[TPattern]:
    """
    Inherit new pattern class from built-in `pattern_class` with specified attributes from `**kwargs`

    Parameters
    ----------
    pattern_class: type[Pattern] or str
        A pattern class to inherit from.
        If a string value is provided, then instantiate from the built-in class with the said class name.
    pattern_name: str, dafault='FactoredPattern'
        The name of the class returned by the factored.
        Specifying the name of the pattern class is not necessary (thus it is an optional parameters),
        but it can improve the overall user experience.
    kwargs:
        Class attributes to specify in the produced pattern class.
        The set of possible class attributes depends on each specific pattern class.

    Returns
    -------
    inherited_class: type[Pattern]
        Pattern class inherited from the built-in `pattern_class`.

    Raises
    ------
    ValueError:
        If the provided `pattern_class` does have attributes provided via `**kwargs`.
        Also, if the provided `pattern_class` is a string that cannot be mapped to
        any built-in pattern class from `built_in_patterns` module.

    Examples
    --------
    >>> AgePattern = pattern_factory(bip.IntervalPattern, 'AgePattern', BoundsUniverse=tuple(range(0, 100, 10)))
    >>> print(isinstance(AgePattern(25)), bip.IntervalPattern)
    True

    >>> NamePattern = pattern_factory("NgramSetPattern")
    >>> print(NamePattern("Evariste Galois"))
    "{'Evariste Galois'}"

    """
    if isinstance(pattern_class, str):
        pattern_mapping = {
            'ItemSetPattern': bip.ItemSetPattern,
            'CategorySetPattern': bip.CategorySetPattern,
            'IntervalPattern': bip.IntervalPattern,
            'ClosedIntervalPattern': bip.ClosedIntervalPattern,
            'NgramSetPattern': bip.NgramSetPattern,
            'CartesianPattern': bip.CartesianPattern,
        }
        if isinstance(pattern_class, str) and pattern_class not in pattern_mapping:
            raise ValueError(
                f'Pattern class `{pattern_class}` cannot be mapped to any existing built-in patterns. '
                f'Supported string values for naming built-in patterns are: {set(pattern_mapping.keys())}.')
        pattern_class = pattern_mapping[pattern_class]

    FactoredPattern = type(pattern_name, (pattern_class,), {})

    for k, v in kwargs.items():
        if not hasattr(FactoredPattern, k):
            raise ValueError(f"Pattern class {pattern_class} does not have predefined attribute {k}. "
                             f"Consult the documentation for the pattern class to find the list of existing attributes.")
        setattr(FactoredPattern, k, v)
    return FactoredPattern
