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

```python
from paspailleur import pattern_structures as PS
```


## General use

**IntervalPS**

Every description is a closed interval of real numbers `[a,b]`.
Description `[a,b]` is less precise than description `[c,d]` if `a<=c, d<=b`.
For example, description `[1.5, 3.14]` is less precise than `[2, 3]`, i.e. `[1.5, 3.14] ⊑ [2, 3]`
_(yes the notation is counterintuitive here)_.

```python
d1, d2 = (1.5, 3.14), (2, 3)
ps = PS.IntervalPS()
assert ps.is_less_precise(d1, d2)
```


**SubSetPS**

Every description is a set of values.
Description `A` is less precise than description `B` if `A` is a subset of `B`: `A ⊆ B`.
For example description `{green, cubic}` is less precise than `{green, cubic, heavy}`.

```python
d1, d2 = {'green', 'cubic'}, {'green', 'cubic', 'heavy'}
ps = PS.SubSetPS()
assert ps.is_less_precise(d1, d2)
```

**SuperSetPS**

Every description is a set of values.
Description `A` is less precise than description `B` if `A` is a superset of `B`: `A ⊇ B`.
For example description `{green, yellow, red}` is less precise than `{green, yellow}`.

```python
d1, d2 = {'green', 'yellow', 'red'}, {'green', 'yellow'}
ps = PS.SuperSetPS()
assert ps.is_less_precise(d1, d2)
```

**CartesianPS**

A pattern structure to combine various independent basic pattern structures in one.

```python
# Combining three previous examples together
d1 = [(1.5, 3.14), {'green', 'cubic'}, {'green', 'yellow', 'red'}]
d2 = [(2, 3), {'green', 'cubic', 'heavy'}, {'green', 'yellow'}]
basic_structures = [PS.IntervalPS(), PS.SubSetPS(), PS.SuperSetPS()]
ps = PS.CartesianPS(basic_structures)
assert ps.is_less_precise(d1, d2)
```


## NLP

**NgramPS**

Every description is a set of incomparable ngram, i.e. set of incomparable tuple of words.

Ngram `A = (a_1, a_2, ..., a_n)` is less precise than ngram `B = (b_1, b_2, ..., b_m)` if `A` can be embedded into `B`:
i.e. exists `i = 1, ..., m-n` s.t. `A = B[i:i+n]`.
For example `(hello, world)` is less precise than `(hello, world, !)`.

Description `D_1 = {A_1, A_2, ...}` is less precise than description `D_2 = {B_1, B_2, ...}`
if every ngram in `D1` is less precise than some ngram in `D2`.  

```python
d1 = {('hello', 'world'), ('!',)}
d2 = {('hello', 'world', '!')}
ps = PS.NgramPS()
assert ps.is_less_precise(d1, d2)
```

**SynonymPS**

Every description is a set of words, representing the synonyms of words contained in some text.
Description `A` is less precise than description `B` if `A` is a subset of `B`: `A ⊆ B`.

```python
d1, d2 = 'hello', 'hello world'
ps = PS.SynonymPS()
pattern1, pattern2 = ps.preprocess_data([d1, d2])
assert ps.is_less_precise(pattern1, pattern2)

print('pattern1:', pattern1)
print('pattern2:', pattern2)
```
> pattern1: {'hello'} \
> pattern2: {'hello', 'universe'}

**AntonymPS**

Every description is a set of words, representing the antonyms of words contained in some text.
Description `A` is less precise than description `B` if `A` is a subset of `B`: `A ⊆ B`.

```python
d1, d2 = 'good', 'good dog'
ps = PS.AntonymPS()
pattern1, pattern2 = ps.preprocess_data([d1, d2])
assert ps.is_less_precise(pattern1, pattern2)

print('pattern1:', pattern1)
print('pattern2:', pattern2)
```
> pattern1: {'evil'} \
> pattern2: {'evil'}

_So, the system does not know any antonym to "dog"._

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


