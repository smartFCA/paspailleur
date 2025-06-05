Pattern Class
=============

.. autoclass:: paspailleur.pattern_structures.pattern.Pattern


Values' Encapsulation
.....................

.. automethod:: paspailleur.pattern_structures.pattern.Pattern.value

.. automethod:: paspailleur.pattern_structures.pattern.Pattern.parse_string_description

.. automethod:: paspailleur.pattern_structures.pattern.Pattern.preprocess_value


Patterns' Order
...............

.. automethod:: paspailleur.pattern_structures.pattern.Pattern.issubpattern

.. automethod:: paspailleur.pattern_structures.pattern.Pattern.issuperpattern

.. automethod:: paspailleur.pattern_structures.pattern.Pattern.min_pattern

.. automethod:: paspailleur.pattern_structures.pattern.Pattern.max_pattern


Patterns' Merge
...............

.. automethod:: paspailleur.pattern_structures.pattern.Pattern.meet

.. automethod:: paspailleur.pattern_structures.pattern.Pattern.join

.. automethod:: paspailleur.pattern_structures.pattern.Pattern.intersection

.. automethod:: paspailleur.pattern_structures.pattern.Pattern.union

.. automethod:: paspailleur.pattern_structures.pattern.Pattern.difference

.. automethod:: paspailleur.pattern_structures.pattern.Pattern.meetable

.. automethod:: paspailleur.pattern_structures.pattern.Pattern.joinable

.. automethod:: paspailleur.pattern_structures.pattern.Pattern.substractable


Atomic Representations
......................

.. automethod:: paspailleur.pattern_structures.pattern.Pattern.atomic_patterns

.. automethod:: paspailleur.pattern_structures.pattern.Pattern.maximal_atoms

.. automethod:: paspailleur.pattern_structures.pattern.Pattern.atomisable
