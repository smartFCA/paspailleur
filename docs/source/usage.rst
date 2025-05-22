Paspailleur in one page
=======================

.. _installation:

Install
-------

To use Paspailleur, first install it using pip:

.. code-block:: console

   $ pip install paspailleur


Define complex patterns
-----------------------

Paspailleur lets you describe the data using complex "patterns" such as itemsets (``ItemSetPattern``), intervals (``IntervalPattern``), categories (``CategorySetPattern``), and ngrams (``NgramSetPattern``).

Built-in patterns
.................

<Here goes some example of various "complex" built-in patterns>

Pattern Structure to represent the dataset
..........................................

<Here goes some example of representing >


Discover patterns in the data
-----------------------------

After defining the intents and extents of the search now we create the pattern structures

.. code-block:: python
   :linenos:

   from paspailleur.pattern_structures.pattern_structure import PatternStructure

   ps = PatternStructure()
   ps.fit(context, min_atom_support=80)

What this code does, is create pattern structures by joining atomic patterns together, the created pattern structure has a minimum of 80 support atoms.
A pattern describes the objects covered by all atomic patterns it consists of.

Mine concepts
.............

Here is the code for mining pattern concepts

.. code-block:: python
   :linenos:

   concepts = ps.mine_concepts(min_delta_stability=20, min_support=80, algorithm='gSofia', use_tqdm=True)
   print(  type(concepts), len(concepts))

The parameters used to mine concepts are:
``min support`` which is the minimum number of objects covered by the concept.
``min_delta_stability`` which means that all more precise concepts will cover less objects.

Mine implications
.................

<Here comes some easy-to-understand example>

Mine subgroups
..............

<Here comes some easy-to-understand example>

Iterate all patterns
....................

<Here comes some easy-to-understand example>