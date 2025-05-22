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

For this we'll be using the Titanic example since it has more than one pattern type:

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

   from paspailleur.pattern_structures import MixedPatternStructure

   pattern_structure = MixedPatternStructure(
      df,
      {
         "Survived": SurvivedPattern,
         "Known Age": KnownAgePattern,
         "Known Cabin": KnownCabinPattern,
         "Sex": SexPattern,
         "Embarked": EmbarkedPattern,
         "Passenger Class": PassengerClassPattern,
         "Age": AgePattern,
         "# Siblings and Spouses": NSiblingsPattern,
         "# Parents and Children": NParentsPattern,
         "Fare": FarePattern
      }
   )

Which is the intents behind the pattern classes, used to make the actual analysis.

Discover patterns in the data
-----------------------------

Mine concepts
.............

<Here comes some easy-to-understand example>

Mine implications
.................

<Here comes some easy-to-understand example>

Mine subgroups
..............

<Here comes some easy-to-understand example>

Iterate all patterns
....................

<Here comes some easy-to-understand example>