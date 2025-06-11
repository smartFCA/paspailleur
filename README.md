[![PyPi](https://img.shields.io/pypi/v/paspailleur)](https://pypi.org/project/paspailleur)
[![Documentation](https://img.shields.io/github/actions/workflow/status/smartFCA/paspailleur/documentation.yml?logo=github&label=Documentation)](https://smartfca.github.io/paspailleur)
[![Licence](https://img.shields.io/github/license/EgorDudyrev/FCApy)](https://github.com/EgorDudyrev/paspailleur/blob/main/LICENSE)
[![LORIA](https://img.shields.io/badge/Made_in-LORIA-61acdf)](https://www.loria.fr/)
[![SmartFCA](https://img.shields.io/badge/Funded_by-SmartFCA-537cbb)](https://www.smartfca.org)
[![DOI](https://zenodo.org/badge/636730644.svg)](https://doi.org/10.5281/zenodo.15639145)

# paspailleur
An add-on for caspailleur to work with Pattern Structures

A Pattern Structure `(D, ⊑)` represents a description space `D`
where every two descriptions can be compared by a "less precise" operator `⊑`.
For example, if `D` is a set of ngrams then ngram `(hello,)` is less precise then `(hello, world)`: `(hello, ) ⊑ (hello, world)`,
that is every ngram that contains `(hello, world)` contains `(hello,)`.


> [!IMPORTANT]
> The latest version v0.1 is fully functioning but published without proper README.md in order to meet the submission deadlines of CONCEPTS'25 conference.
> README.md and some Quality-of-Life improvements will come soon.
