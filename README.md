[![PyPi](https://img.shields.io/pypi/v/paspailleur)](https://pypi.org/project/paspailleur)
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
> The package is in active development stage. Things can change often.


# Implemented Pattern Structures

## General use

**IntervalPS**

Every description is a closed interval of real numbers `[a,b]`.
Description `[a,b]` is less precise than description `[c,d]` if `a<=c, d<=b`.
For example, description `[1.5, 3.14]` is less precise than `[2, 3]`, i.e. `[1.5, 3.14] ⊑ [2, 3]`
_(yes the notation is counterintuitive here)_.

**SubSetPS**

Every description is a set of values.
Description `A` is less precise than description `B` if `A` is a subset of `B`: `A ⊆ B`.
For example description `{green, cubic}` is less precise than `{green, cubic, heavy}`.

**SuperSetPS**

Every description is a set of values.
Description `A` is less precise than description `B` if `A` is a superset of `B`: `A ⊇ B`.
For example description `{green, yellow, red}` is less precise than `{green, yellow}`.

**CartesianPS**

A pattern structure to combine various independent basic pattern structures in one.


## NLP

**NgramPS**

Every description is an ngram, i.e. tuple of words.
Description `A = (a_1, a_2, ..., a_n)` is less precise than description `B = (b_1, b_2, ..., b_m)` if `A` can be embedded into `B`:
i.e. exists `i = 1, ..., m-n` s.t. `A = B[i:i+n]`.
For example `(hello, world)` is less precise than `(hello, world, !)`. 

**SynonymPS**

Every description is a set of words, representing the synonyms of words contained in some text.
Description `A` is less precise than description `B` if `A` is a subset of `B`: `A ⊆ B`.

**AntonymPS**

Every description is a set of words, representing the antonyms of words contained in some text.
Description `A` is less precise than description `B` if `A` is a subset of `B`: `A ⊆ B`.

## Tabular data

> [!INFO]
> Coming soon

**NumberPS**

**CategoryPS**

## Graphs

> [!INFO]
> Coming soon

_Coming soon_

**GraphPS**

**OrderedGraphPS**

# How to create a custom Pattern Structure

_To be described_


