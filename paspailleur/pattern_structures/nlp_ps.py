from typing import Iterator, Optional
from bitarray import frozenbitarray as fbarray
from paspailleur.pattern_structures import AbstractPS
from nltk.util import ngrams
from nltk.corpus import wordnet
nltk.download('wordnet')

class NgramsPS(AbstractPS):
    PatternType =  Set[Tuple[str]]
    bottom = None
    
    def __init__(self, min_n = 2: int):
        self.min_n = min_n

    def join_patterns(self, a: PatternType, b: PatternType) -> PatternType:
        """Return the common ngrams between both patterns `a` and `b`"""

        # Transform the set of words into text
        tuple_text1 = list(a)[0]
        tuple_text2 = list(b)[0]
        text1 = ' '.join(tuple_text1)
        text2 = ' '.join(tuple_text2)

        # Find identical n-grams for all possible sizes
        identical_ngrams = set()
        max_n = min(len(text1.split()), len(text2.split()))
        for n in range(self.min_n, max_n + 1):
            ngrams1 = set(ngrams(text1.split(), n))
            ngrams2 = set(ngrams(text2.split(), n))
            identical_ngrams.update(ngrams1.intersection(ngrams2))

        # Delete identical n-grams contained in other identical n-grams
        for ngram in list(identical_ngrams):
            for other_ngram in identical_ngrams:
                if ngram != other_ngram and set(ngram).issubset(set(other_ngram)):
                    identical_ngrams.remove(ngram)
                    break

        return identical_ngrams

    def iter_bin_attributes(self, data: list[PatternType]) -> Iterator[tuple[PatternType, fbarray]]:
        """Iterate binary attributes obtained from `data` (from the most general to the most precise ones)

        :parameter
            data: list[PatternType]
             list of object descriptions
        :return
            iterator of (description: PatternType, extent of the description: frozenbitarray)
        """
        raise NotImplementedError


    def is_less_precise(self, a: PatternType, b: PatternType) -> bool:
        """Return True if pattern `a` is less precise than pattern `b`"""
        for smaller_tuple in b:
            inclusion_found = False
            for larger_tuple in a:
                if any(smaller_tuple == larger_tuple[i:i+len(smaller_tuple)] for i in range(len(larger_tuple) - len(smaller_tuple) + 1)):
                    inclusion_found = True
                    break
            if not inclusion_found:
                return False
        return True

    def n_bin_attributes(self, data: list[PatternType]) -> int:
        """Count the number of attributes in the binary representation of `data`"""
        raise NotImplementedError



class SynonymsPS(NgramsPS(1)):
    PatternType = Set[Tuple[str]]

    def __init__(self, num_synonyms: int):
        self.num_synonyms = num_synonyms # number of synonyms for a word

    def get_synonyms(self, word):
      synonyms = set()
      for synset in wordnet.synsets(word):
          for lemma in synset.lemmas():
              synonyms.add(lemma.name())
              if self.num_synonyms is not None and len(synonyms) >= self.num_synonyms:
                  return synonyms
      return synonyms

    def get_synonyms_set_for_text(self, a: PatternType) -> PatternType:
      synonym_liste = []
      words_liste = list(a)[0]
      for i in range(len(words_liste)):
          for synonym in list(self.get_synonyms(words_liste[i], self.num_synonyms)):
            synonym_liste.append(synonym)
      synonym_tuple = tuple(synonym_liste)
      return(set(synonym_tuple))


    def join_synonyms_patterns(self, a: PatternType, b: PatternType) -> PatternType:
        """Return the common synonyms between both patterns `a` and `b`"""
        synonyms_text1 = self.get_synonyms_set_for_text(a, self.num_synonyms)
        synonyms_text2 = self.get_synonyms_set_for_text(b, self.num_synonyms)
        return(self.join_patterns(synonyms_text1, synonyms_text2))

    def is_less_precise(self, a: PatternType, b: PatternType) -> bool:
        return(self.is_less_precise)


    def iter_bin_attributes(self, data: list[PatternType]) -> Iterator[tuple[PatternType, fbarray]]:
        """Iterate binary attributes obtained from `data` (from the most general to the most precise ones)

        :parameter
            data: list[PatternType]
             list of object descriptions
        :return
            iterator of (description: PatternType, extent of the description: frozenbitarray)
        """
        raise NotImplementedError

    def n_bin_attributes(self, data: list[PatternType]) -> int:
        """Count the number of attributes in the binary representation of `data`"""

        raise NotImplementedError



class AntonymsPS(NgramsPS(1)):
    PatternType = Set[Tuple[str]]

    def __init__(self, num_antonyms: int):
        self.num_antonyms = num_antonyms

    def get_antonyms(word, num_antonyms=4):
        antonyms = set()
        for synset in wordnet.synsets(word):
            for lemma in synset.lemmas():
                if lemma.antonyms():
                    antonyms.add(lemma.antonyms()[0].name())
                    if num_antonyms is not None and len(antonyms) >= num_antonyms:
                        return antonyms
        return antonyms


    def get_antonyms_set_for_text(self, a: PatternType) -> PatternType:
      antonyms_liste = []
      words_liste = list(a)[0]
      for i in range(len(words_liste)):
          for antonym in list(self.get_antonyms(words_liste[i], self.num_antonyms)):
            antonyms_liste.append(antonym)
      antonyms_tuple = tuple(antonyms_liste)
      return(set(antonyms_tuple))


    def join_antonyms_patterns(self, a: PatternType, b: PatternType) -> PatternType:
        """Return the common antonyms between both patterns `a` and `b`"""
        antonyms_text1 = self.get_antonyms_set_for_text(a, self.num_antonyms)
        antonyms_text2 = self.get_antonyms_set_for_text(b, self.num_antonyms)
        return(self.join_patterns(antonyms_text1, antonyms_text2))

    def is_less_precise(self, a: PatternType, b: PatternType) -> bool:
        return(self.is_less_precise)


    def iter_bin_attributes(self, data: list[PatternType]) -> Iterator[tuple[PatternType, fbarray]]:
        """Iterate binary attributes obtained from `data` (from the most general to the most precise ones)

        :parameter
            data: list[PatternType]
             list of object descriptions
        :return
            iterator of (description: PatternType, extent of the description: frozenbitarray)
        """
        raise NotImplementedError

    def n_bin_attributes(self, data: list[PatternType]) -> int:
        """Count the number of attributes in the binary representation of `data`"""
        raise NotImplementedError










