[![PyPi](https://img.shields.io/pypi/v/paspailleur)](https://pypi.org/project/paspailleur)
[![Docs](https://img.shields.io/badge/Docs-61acdf)](https://smartfca.github.io/paspailleur)
![GitHub deployments](https://img.shields.io/github/deployments/smartFCA/paspailleur/documentation.yml)
[![Licence](https://img.shields.io/github/license/EgorDudyrev/FCApy)](https://github.com/EgorDudyrev/paspailleur/blob/main/LICENSE)
[![LORIA](https://img.shields.io/badge/Made_in-LORIA-61acdf)](https://www.loria.fr/)
[![SmartFCA](https://img.shields.io/badge/Funded_by-SmartFCA-537cbb)](https://www.smartfca.org)


# paspailleur
An add-on for caspailleur to work with Pattern Structures

A Pattern Structure `(D, ⊑)` represents a description space `D`
where every two descriptions can be compared by a "less precise" operator `⊑`.
For example, if `D` is a set of ngrams then ngram `(hello,)` is less precise then `(hello, world)`: `(hello, ) ⊑ (hello, world)`,
that is every ngram that contains `(hello, world)` contains `(hello,)`.


> [!WARNING]
> The new version of the package will be published soon. And the README file will be greatly updated
