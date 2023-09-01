from collections import deque
from functools import reduce
from itertools import combinations
from typing import Iterator, Iterable

from bitarray import frozenbitarray as fbarray, bitarray
from bitarray.util import zeros as bazeros

from .abstract_ps import AbstractPS


import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')


class SynonymsPS(AbstractPS):
    PatternType = set[str]  # Every text is described by a set of synonyms to words in the text
    n_synonyms: int | None = 1  # number of synonyms for a word
    
    def __init__(self, n_synonyms: int | None = 1):
        self.n_synonyms = n_synonyms

    def get_synonyms(self, word: str) -> set[str]:
        synonyms = set()
        for synset in wordnet.synsets(word):
            for lemma in synset.lemmas():
                synonyms.add(lemma.name())
                if self.n_synonyms is not None and len(synonyms) >= self.n_synonyms:
                    return synonyms
        return synonyms

    def preprocess_data(self, data: Iterable[str], separator=' ') -> Iterator[PatternType]:
        for text in data:
            words = text.split(separator)
            synonyms = reduce(lambda a, b: a | b, (self.get_synonyms(word) for word in words), set())
            yield synonyms

    def join_patterns(self, a: PatternType, b: PatternType) -> PatternType:
        """Return the common synonyms between both patterns `a` and `b`"""
        return a.intersection(b)

    def is_less_precise(self, a: PatternType, b: PatternType) -> bool:
        return a.issubset(b)

    def iter_bin_attributes(self, data: list[PatternType], min_support: int = 0) -> Iterator[tuple[PatternType, fbarray]]:
        empty_extent = bazeros(len(data))

        synonyms_extents: dict[str, bitarray] = {}
        for i, pattern in enumerate(data):
            for ngram in pattern:
                for word in ngram:
                    if word not in synonyms_extents:
                        synonyms_extents[word] = empty_extent.copy()
                    synonyms_extents[word][i] = True

        total_pattern = set(synonym for synonym, extent in synonyms_extents.items() if extent.all())
        yield total_pattern, fbarray(~empty_extent)

        words_to_pop = [word for word, extent in synonyms_extents.items() if extent.count() < min_support or extent.all()]
        for word in words_to_pop:
            del synonyms_extents[word]

        synonyms, extents = [list(vals) for vals in zip(*synonyms_extents.items())]
        n_synonyms = len(synonyms)

        queue = deque([({word}, extent, i) for i, (word, extent) in enumerate(zip(synonyms, extents))])
        while queue:
            words, extent, max_word_idx = queue.popleft()
            yield words, fbarray(extent)

            for i in range(max_word_idx+1, n_synonyms):
                next_extent = extent & extents[i]
                if not next_extent.any() or next_extent.count() < min_support:
                    continue

                if extent & extents[i] == extent or extents[i] & extent == extents[i]:
                    continue

                next_words = words | {synonyms[i]}
                queue.append((next_words, next_extent, i))

        if min_support == 0:
            yield None, fbarray(empty_extent)

    def n_bin_attributes(self, data: list[PatternType]) -> int:
        """Count the number of attributes in the binary representation of `data`"""
        # TODO: Test if this equation works
        n = sum(len(pattern) for pattern in data)
        return (n * (n + 1)) // 2


class AntonymsPS(NgramPS):
    PatternType = set[tuple[str]]
    num_antonyms : int
    
    def __init__(self, num_antonyms = 1):
        self.num_antonyms = num_antonyms

    def get_antonyms(self, word):
        antonyms = set()
        for synset in wordnet.synsets(word):
            for lemma in synset.lemmas():
                if lemma.antonyms():
                    antonyms.add(lemma.antonyms()[0].name())
                    if self.num_antonyms is not None and len(antonyms) >= self.num_antonyms:
                        return antonyms
        return antonyms


    def get_antonyms_set_for_text(self, a: PatternType) -> PatternType:
      antonyms_list = []
      words_list = list(a)[0]
      for i in range(len(words_list)):
          for antonym in list(self.get_antonyms(words_list[i], self.num_antonyms)):
            antonyms_list.append(antonym)
      antonyms_tuple = [tuple(antonyms_list)]
      return(set(antonyms_tuple))


    def join_antonyms_patterns(self, a: PatternType, b: PatternType) -> PatternType:
        """Return the common antonyms between both patterns `a` and `b`"""
        antonyms_text1 = self.get_antonyms_set_for_text(a, self.num_antonyms)
        antonyms_text2 = self.get_antonyms_set_for_text(b, self.num_antonyms)
        return(self.join_patterns(antonyms_text1, antonyms_text2))


    def n_bin_attributes(self, data: list[PatternType]) -> int:
        """Count the number of attributes in the binary representation of `data`"""
        n = 0
        for i in range(len(data)):
            n += len(list(list(data[i])[0]))
        return (n * (n + 1)) // 2

