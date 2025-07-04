Pattern Class
=============

.. autoclass:: paspailleur.Pattern


Values' Encapsulation
.....................

.. autoproperty:: paspailleur.Pattern.value

.. automethod:: paspailleur.Pattern.parse_string_description

.. automethod:: paspailleur.Pattern.preprocess_value


Patterns' Order
...............

.. automethod:: paspailleur.Pattern.issubpattern

.. automethod:: paspailleur.Pattern.issuperpattern

.. autoproperty:: paspailleur.Pattern.min_pattern

.. autoproperty:: paspailleur.Pattern.max_pattern


Patterns' Merge
...............

.. automethod:: paspailleur.Pattern.meet

.. automethod:: paspailleur.Pattern.join

.. automethod:: paspailleur.Pattern.intersection

.. automethod:: paspailleur.Pattern.union

.. automethod:: paspailleur.Pattern.difference

.. autoproperty:: paspailleur.Pattern.meetable

.. autoproperty:: paspailleur.Pattern.joinable

.. autoproperty:: paspailleur.Pattern.substractable


Atomic Representations
......................

.. autoproperty:: paspailleur.Pattern.atomic_patterns

.. autoproperty:: paspailleur.Pattern.maximal_atoms

.. autoproperty:: paspailleur.Pattern.atomisable
