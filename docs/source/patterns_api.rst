Patterns API
============

Pattern Class
-------------

.. autoclass:: paspailleur.pattern_structures.pattern.Pattern


Values' Encapsulation
.....................

.. automethod:: paspailleur.pattern_structures.pattern.Pattern.value

.. automethod:: paspailleur.pattern_structures.pattern.Pattern.parse_string_description

.. automethod:: paspailleur.pattern_structures.pattern.Pattern.preprocess_value


Patterns' Order
...............

.. automethod:: paspailleur.pattern_structures.pattern.Pattern.issubpattern

.. automethod:: paspailleur.pattern_structures.pattern.Pattern.issupersubpattern

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


Built-in Patterns
-----------------

.. automodule:: paspailleur.pattern_structures.built_in_patterns
   :members:


Pattern Structure
-----------------

.. autoclass:: paspailleur.pattern_structures.pattern_structure.PatternStructure


Basic operations on a context
.............................

.. automethod:: paspailleur.pattern_structures.pattern_structure.PatternStructure.extent

.. automethod:: paspailleur.pattern_structures.pattern_structure.PatternStructure.intent


Initialisation of the Pattern Structure
.......................................

.. automethod:: paspailleur.pattern_structures.pattern_structure.PatternStructure.fit

.. automethod:: paspailleur.pattern_structures.pattern_structure.PatternStructure.intent


Properties that are easy to compute
...................................

.. automethod:: paspailleur.pattern_structures.pattern_structure.PatternStructure.min_pattern

.. automethod:: paspailleur.pattern_structures.pattern_structure.PatternStructure.max_pattern

.. automethod:: paspailleur.pattern_structures.pattern_structure.PatternStructure.max_atoms

.. automethod:: paspailleur.pattern_structures.pattern_structure.PatternStructure.atomic_patterns

.. automethod:: paspailleur.pattern_structures.pattern_structure.PatternStructure.n_atomic_patterns

.. automethod:: paspailleur.pattern_structures.pattern_structure.PatternStructure.atomic_patterns_order

.. automethod:: paspailleur.pattern_structures.pattern_structure.PatternStructure.premaximal_patterns


Pattern Iterators
.................

.. automethod:: paspailleur.pattern_structures.pattern_structure.PatternStructure.iter_atomic_patterns

.. automethod:: paspailleur.pattern_structures.pattern_structure.PatternStructure.iter_patterns

.. automethod:: paspailleur.pattern_structures.pattern_structure.PatternStructure.iter_keys

.. automethod:: paspailleur.pattern_structures.pattern_structure.PatternStructure.iter_subgroups


High-level FCA API
..................

.. automethod:: paspailleur.pattern_structures.pattern_structure.PatternStructure.mine_concepts

.. automethod:: paspailleur.pattern_structures.pattern_structure.PatternStructure.mine_implications

.. automethod:: paspailleur.pattern_structures.pattern_structure.PatternStructure.mine_implications_from_premises

.. automethod:: paspailleur.pattern_structures.pattern_structure.PatternStructure.next_pattern


Measures of Patterns
....................

.. automethod:: paspailleur.pattern_structures.pattern_structure.PatternStructure.measure_support

.. automethod:: paspailleur.pattern_structures.pattern_structure.PatternStructure.measure_frequency

.. automethod:: paspailleur.pattern_structures.pattern_structure.PatternStructure.measure_delta_stability


Helping Functions
.................

.. automethod:: paspailleur.pattern_structures.pattern_structure.PatternStructure.verbalise_extent
