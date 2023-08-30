from typing import Iterator, Optional
from bitarray import frozenbitarray as fbarray
from bitarray import bitarray
from paspailleur.pattern_structures import AbstractPS
from nltk.util import ngrams
from nltk.corpus import wordnet
nltk.download('wordnet')

class NgramsPS(AbstractPS):
    PatternType =  set[tuple[str]]
    bottom = None
    min_n : int
    
    def __init__(self, min_n = 1):
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
        List_ngrams = []
        for i in range(len(data)):
            for ngram in data[i]:
                List_ngrams.append(ngram)
        List_ngrams = list(set(List_ngrams))
        List_ngrams = sorted(List_ngrams)
        L_extent = []
        for i in range(len(data)):
            ngram_i = [data[i]]
            bit_i = bazeros(len(List_ngrams))
            for j in range(len(List_ngrams)):
                if List_ngrams[j] in data[i]:
                    bit_i[j] = 1
            ngram_i.append(bit_i)
            L_extent.append(tuple(ngram_i))
        L_extent = sorted(L_extent, key=lambda x: (x[1].count(1), x[1]), reverse=True)
        yield List_ngrams, bitarray([1]*len(List_ngrams))
        for n_gram, extent in L_extent:
            yield n_gram, extent


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
        count = 0
        nbr_of_words = 0
        for i in range(len(data)):
            nbr_of_words += len(list(list(data[i])[0]))
        for i in range(nbr_of_words):
            for j in range(i + self.min_n, nbr_of_words + 1):
                count += 1
        return(count)



class SynonymsPS(NgramsPS):
    PatternType = set[tuple[str]]
    num_synonyms : int
    
    def __init__(self, num_synonyms = 1):
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
      synonym_list = []
      words_list = list(a)[0]
      for i in range(len(words_list)):
          for synonym in list(self.get_synonyms(words_list[i], self.num_synonyms)):
            synonym_list.append(synonym)
      synonym_tuple = [tuple(synonym_list)]
      return(set(synonym_tuple))


    def join_synonyms_patterns(self, a: PatternType, b: PatternType) -> PatternType:
        """Return the common synonyms between both patterns `a` and `b`"""
        synonyms_text1 = self.get_synonyms_set_for_text(a, self.num_synonyms)
        synonyms_text2 = self.get_synonyms_set_for_text(b, self.num_synonyms)
        return(self.join_patterns(synonyms_text1, synonyms_text2))


    def n_bin_attributes(self, data: list[PatternType]) -> int:
        """Count the number of attributes in the binary representation of `data`"""
        n = 0
        for i in range(len(data)):
            n += len(list(list(data[i])[0]))
        return (n * (n + 1)) // 2




class AntonymsPS(NgramsPS):
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










