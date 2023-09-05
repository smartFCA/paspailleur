from functools import reduce
from typing import Iterator, Iterable

from .set_ps import SubSetPS


import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')


class SynonymsPS(SubSetPS):
    PatternType = frozenset[str]  # Every text is described by a set of synonyms to words in the text
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


class AntonymsPS(SubSetPS):
    PatternType = frozenset[str]  # Every text is described by a set of antonyms to words in the text
    num_antonyms : int | None = 1  # number of antonyms for a word
    
    def __init__(self, num_antonyms: int = 1):
        self.num_antonyms = num_antonyms

    def get_antonyms(self, word: str) -> PatternType:
        antonyms = set()
        for synset in wordnet.synsets(word):
            for lemma in synset.lemmas():
                if lemma.antonyms():
                    antonyms.add(lemma.antonyms()[0].name())
                    if self.num_antonyms is not None and len(antonyms) >= self.num_antonyms:
                        return frozenset(antonyms)
        return frozenset(antonyms)

    def preprocess_data(self, data: Iterable[str], separator=' ') -> Iterator[PatternType]:
        for text in data:
            words = text.split(separator)
            antonyms = reduce(lambda a, b: a | b, (self.get_antonyms(word) for word in words), set())
            yield antonyms


if __name__ == '__main__':
    syn_ps = SynonymsPS(n_synonyms=5)
    assert syn_ps.get_synonyms('great') == {'outstanding', 'bang-up', 'great', 'bully', 'corking'}

