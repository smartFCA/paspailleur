:html_theme.sidebar_secondary.remove:

.. Paspailleur documentation master file, created by
   sphinx-quickstart on Tue May 13 15:15:08 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==========================
Paspailleur Python Package
==========================

Pattern Mining with Pattern Structures.


.. grid::

   .. grid-item-card:: 🚢Getting started

      Learn the basics of Paspailleur in a single page.

      +++
      .. button-ref:: example_from_titanic
         :ref-type: doc
         :click-parent:
         :color: secondary
         :expand:
         :tooltip: To the Getting started guide
         :shadow:

   .. grid-item-card:: 🌀Patterns API

      Read the documentation for Patterns and PatternStructure classes: the frontend of the package.

      +++
      .. button-ref:: patterns_api
         :ref-type: ref
         :click-parent:
         :color: secondary
         :expand:
         :tooltip: To Patterns API documentation


   .. grid-item-card:: ⚙️Algorithms API

      Read the documentation for functions that work behind Patterns API: the backend of the package.

      +++
      .. button-ref:: algorithms_api
         :ref-type: ref
         :click-parent:
         :color: secondary
         :expand:
         :tooltip: To Algorithms API documentation
         :shadow:


.. rubric:: Remember the name

Package name `Paspailleur` is pronounced as [pas.pa.jœʁ] that can be simplified to [paspajor] or [paspayor].
The name comes from the french word "Orpailleur" meaning "gold miner".
Orpailleur was also the name of the team where the package's development was initiated.
So, in a sense, `Paspailleur` can be deciphered as "Pattern Structures miner".


.. rubric:: Useful links

GALACTIC software (https://galactic.univ-lr.fr): a Python package for working with Pattern Structures developed in La Rochelle Université.
It is also a part of SmartFCA project.

More software developed within SmartFCA project: https://www.smartfca.org/software/.

Uta Priss webpage on FCA (https://upriss.github.io/fca/fca.html): old-by-gold landing page for FCA community.


.. rubric:: Funding

The package development is supported by ANR project SmartFCA (ANR-21-CE23-0023).

SmartFCA (https://www.smartfca.org/) is a big platform that will contain many extensions of Formal Concept Analysis including pattern structures, Relational Concept Analysis, Graph-FCA and others. While caspailleur is a small python package that covers only the basic notions of FCA.


.. toctree::
    :hidden:
    :maxdepth: 3
    :caption: Contents
    
    example_from_titanic  
    patterns_api/index
    algorithms_api
