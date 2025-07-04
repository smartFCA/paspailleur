Pattern Structure
=================

.. autoclass:: paspailleur.pattern_structures.pattern_structure.PatternStructure


Basic operations on a context
.............................

.. automethod:: paspailleur.pattern_structures.pattern_structure.PatternStructure.extent

.. automethod:: paspailleur.pattern_structures.pattern_structure.PatternStructure.intent


Initialisation of the Pattern Structure
.......................................

.. automethod:: paspailleur.pattern_structures.pattern_structure.PatternStructure.fit

.. automethod:: paspailleur.pattern_structures.pattern_structure.PatternStructure.init_atomic_patterns


Properties that are easy to compute
...................................

.. autoproperty:: paspailleur.pattern_structures.pattern_structure.PatternStructure.min_pattern

.. autoproperty:: paspailleur.pattern_structures.pattern_structure.PatternStructure.max_pattern

.. autoproperty:: paspailleur.pattern_structures.pattern_structure.PatternStructure.max_atoms

.. autoproperty:: paspailleur.pattern_structures.pattern_structure.PatternStructure.atomic_patterns

.. autoproperty:: paspailleur.pattern_structures.pattern_structure.PatternStructure.n_atomic_patterns

.. autoproperty:: paspailleur.pattern_structures.pattern_structure.PatternStructure.atomic_patterns_order

.. autoproperty:: paspailleur.pattern_structures.pattern_structure.PatternStructure.premaximal_patterns


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

.. automethod:: paspailleur.pattern_structures.pattern_structure.PatternStructure.next_patterns


Measures of Patterns
....................

.. automethod:: paspailleur.pattern_structures.pattern_structure.PatternStructure.measure_support

.. automethod:: paspailleur.pattern_structures.pattern_structure.PatternStructure.measure_frequency

.. automethod:: paspailleur.pattern_structures.pattern_structure.PatternStructure.measure_delta_stability


Helping Functions
.................

.. automethod:: paspailleur.pattern_structures.pattern_structure.PatternStructure.verbalise_extent
