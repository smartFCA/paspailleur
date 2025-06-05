Pattern Class
=============

.. autoclass:: paspailleur.pattern_structures.pattern.Pattern


Values' Encapsulation
.....................

.. autoproperty:: paspailleur.pattern_structures.pattern.Pattern.value

.. automethod:: paspailleur.pattern_structures.pattern.Pattern.parse_string_description

.. automethod:: paspailleur.pattern_structures.pattern.Pattern.preprocess_value


Patterns' Order
...............

.. automethod:: paspailleur.pattern_structures.pattern.Pattern.issubpattern

.. automethod:: paspailleur.pattern_structures.pattern.Pattern.issuperpattern

.. autoproperty:: paspailleur.pattern_structures.pattern.Pattern.min_pattern

.. autoproperty:: paspailleur.pattern_structures.pattern.Pattern.max_pattern


Patterns' Merge
...............

.. automethod:: paspailleur.pattern_structures.pattern.Pattern.meet

.. automethod:: paspailleur.pattern_structures.pattern.Pattern.join

.. automethod:: paspailleur.pattern_structures.pattern.Pattern.intersection

.. automethod:: paspailleur.pattern_structures.pattern.Pattern.union

.. automethod:: paspailleur.pattern_structures.pattern.Pattern.difference

.. autoproperty:: paspailleur.pattern_structures.pattern.Pattern.meetable

.. autoproperty:: paspailleur.pattern_structures.pattern.Pattern.joinable

.. autoproperty:: paspailleur.pattern_structures.pattern.Pattern.substractable


Atomic Representations
......................

.. autoproperty:: paspailleur.pattern_structures.pattern.Pattern.atomic_patterns

.. autoproperty:: paspailleur.pattern_structures.pattern.Pattern.maximal_atoms

.. autoproperty:: paspailleur.pattern_structures.pattern.Pattern.atomisable
