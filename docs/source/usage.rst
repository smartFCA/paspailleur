.. _getting_started:

Paspailleur in one page
=======================

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

The study will use the Titanic example and focus on only 20 Passengers, not all of them.
We'll be using the Titanic example since it has more than one pattern type, which uses more than one pattern class:

.. code-block:: python
   :linenos:
   
   from paspailleur.pattern_structures import built_in_patterns as bip

   class SurvivedPattern(bip.CategorySetPattern):
      Universe = ("No", "Yes")

   class KnownAgePattern(bip.CategorySetPattern):
      Universe = ("No", "Yes")

   class KnownCabinPattern(bip.CategorySetPattern):
      Universe = ("No", "Yes")

   class SexPattern(bip.CategorySetPattern):
      Universe = ("female", "male")

   class EmbarkedPattern(bip.CategorySetPattern):
      Universe = ("Southampton", "Cherbourg", "Queenstown")

   class PassengerClassPattern(bip.IntervalPattern):
      ...

   class AgePattern(bip.IntervalPattern):
      BoundsUniverse = (0, 20, 40, 60, 80)

   class NSiblingsPattern(bip.IntervalPattern):
      BoundsUniverse = (0, 1, 2, 3, 5)

   class NParentsPattern(bip.IntervalPattern):
      BoundsUniverse = (0, 1, 2, 3, 5)

   class FarePattern(bip.IntervalPattern):
      BoundsUniverse = (0, 50, 100, 200, 300, 600)

With this we've initialized the pattern classes to be used in the analysis.

Pattern Structure to represent the dataset
..........................................

With the pattern classes defined, you can now construct a dataset pattern structure:

.. code-block:: python
   :linenos:

   from paspailleur.pattern_structures.built_in_patterns import CartesianPattern

   pattern_structure = CartesianPattern(
      df,
      {
        'Survived': SurvivedPattern, 
        'Known Age': KnownAgePattern, 
        'Known Cabin': KnownCabinPattern, 
        'Sex': SexPattern, 
        'Embarked': EmbarkedPattern, 
        'Passenger Class': PassengerClassPattern, 
        'Age': AgePattern,
        '# Siblings and Spouses': NSiblingsPattern, 
        '# Parents and Children': NParentsPattern,
        'Fare': FarePattern, 
        'Name': NamePattern
    }
   )

Which are the intents behind the pattern classes, used to make the actual analysis.

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