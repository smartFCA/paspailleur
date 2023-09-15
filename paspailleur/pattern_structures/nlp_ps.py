from functools import reduce
from typing import Iterator, Iterable

from .set_ps import SubSetPS


import nltk
from nltk.corpus import wordnet


class SynonymPS(SubSetPS):
    PatternType = frozenset[str]  # Every text is described by a set of synonyms to words in the text
    n_synonyms: int | None = 1  # number of synonyms for a word
    max_pattern = frozenset({'<MAX_SYNONYM'})  # Maximal pattern that should be more precise than any other pattern

    def __init__(self, n_synonyms: int | None = 1):
        nltk.download('wordnet')
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


class AntonymPS(SubSetPS):
    PatternType = frozenset[str]  # Every text is described by a set of antonyms to words in the text
    n_antonyms: int | None = 1  # number of antonyms for a word
    max_pattern = frozenset({'<MAX_ANTONYM'})  # Maximal pattern that should be more precise than any other pattern
    
    def __init__(self, n_antonyms: int = 1):
        nltk.download('wordnet')
        self.n_antonyms = n_antonyms

    def get_antonyms(self, word: str) -> PatternType:
        antonyms = set()
        for synset in wordnet.synsets(word):
            for lemma in synset.lemmas():
                if lemma.antonyms():
                    antonyms.add(lemma.antonyms()[0].name())
                    if self.n_antonyms is not None and len(antonyms) >= self.n_antonyms:
                        return frozenset(antonyms)
        return frozenset(antonyms)

    def preprocess_data(self, data: Iterable[str], separator=' ') -> Iterator[PatternType]:
        for text in data:
            words = text.split(separator)
            antonyms = reduce(lambda a, b: a | b, (self.get_antonyms(word) for word in words), set())
            yield antonyms
