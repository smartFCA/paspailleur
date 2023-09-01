import difflib
from collections import deque
from functools import reduce
from typing import Iterator, Iterable

from bitarray import frozenbitarray as fbarray, bitarray
from bitarray.util import zeros as bazeros

from .abstract_ps import AbstractPS


class NgramPS(AbstractPS):
    PatternType = set[
        tuple[str, ...]]  # Every tuple represents an ngram of words. A pattern is a set of incomparable ngrams
    bottom: PatternType = None  # the most specific ngram for the data. Equals to None for simplicity
    min_n: int = 1  # Minimal size of an ngram to consider

    def __init__(self, min_n: int = 1):
        self.min_n = min_n

    def preprocess_data(self, data: Iterable[str], separator=' ') -> Iterator[PatternType]:
        for text in data:
            if not text:
                yield set()
                continue

            ngram = tuple(text.split(separator))
            pattern = {ngram} if len(ngram) >= self.min_n else set()
            yield pattern

    def join_patterns(self, a: PatternType, b: PatternType) -> PatternType:
        """Return the maximal sub-ngrams contained both in `a` and in `b`

        Example:
        >> join_patterns({('hello', 'world'), ('who', 'is', 'there')}, {('hello', 'there')}) == {('hello',), ('there',)}
        as 'hello' is contained both in 'hello world' and 'hello there'
        and 'there' is contained both in 'who is there' and 'hello there'
        """
        # Transform the set of words into text
        seq_matcher = difflib.SequenceMatcher(isjunk=lambda ngram: len(ngram) < self.min_n)

        common_ngrams = []
        for ngram_a in a:
            seq_matcher.set_seq1(ngram_a)
            for ngram_b in b:
                seq_matcher.set_seq2(ngram_b)

                blocks = seq_matcher.get_matching_blocks()[:-1]  # the last block is always empty, so skip it
                common_ngrams.extend((ngram_a[block.a: block.a + block.size] for block in blocks
                                      if block.size >= self.min_n))

        # Delete common n-grams contained in other common n-grams
        common_ngrams = sorted(common_ngrams, key=lambda ngram: len(ngram), reverse=True)
        for i in range(len(common_ngrams)):
            n_ngrams = len(common_ngrams)
            if i == n_ngrams:
                break

            ngram = common_ngrams[i]
            ngrams_to_pop = (j for j in reversed(range(i + 1, n_ngrams)) if
                             self.is_less_precise(common_ngrams[j], ngram))
            for j in ngrams_to_pop:
                common_ngrams.pop(j)

        return set(common_ngrams)

    def iter_bin_attributes(self, data: list[PatternType], min_support: int = 0) -> Iterator[
        tuple[PatternType, fbarray]]:
        """Iterate binary attributes obtained from `data` (from the most general to the most precise ones)

        :parameter
            data: list[PatternType]
             list of object descriptions
        :return
            iterator of (description: PatternType, extent of the description: frozenbitarray)
        """
        empty_extent = bazeros(len(data))

        words_extents: dict[str, bitarray] = {}
        for i, pattern in enumerate(data):
            for ngram in pattern:
                for word in ngram:
                    if word not in words_extents:
                        words_extents[word] = empty_extent.copy()
                    words_extents[word][i] = True

        total_pattern = set()
        if any(extent.all() for extent in words_extents.values()):
            total_pattern = reduce(self.join_patterns, data[1:], data[0])

        yield total_pattern, fbarray(~empty_extent)

        words_to_pop = [word for word, extent in words_extents.items() if extent.count() < min_support]
        for word in words_to_pop:
            del words_extents[word]

        queue = deque([((word,), extent) for word, extent in words_extents.items() if not extent.all()])
        while queue:
            ngram, extent = queue.popleft()
            yield ngram, fbarray(extent)

            for word, word_extent in words_extents.items():
                next_extent = word_extent & extent
                if not next_extent.any() or next_extent.count() < min_support:
                    continue

                support_delta = next_extent.count() - max(min_support, 1)

                next_ngram = ngram + (word,)
                for i in next_extent.itersearch(True):
                    if self.is_less_precise({next_ngram}, data[i]):
                        continue

                    next_extent[i] = False
                    support_delta -= 1
                    if support_delta < 0:
                        break
                else:  # no break, i.e. enough support
                    queue.append((next_ngram, next_extent))

        if min_support == 0:
            yield None, fbarray(empty_extent)

    def is_less_precise(self, a: PatternType, b: PatternType) -> bool:
        """Return True if pattern `a` is less precise than pattern `b`"""
        b_texts_sizes = [(len(ngram), ' '.join(ngram)) for ngram in
                         sorted(b, key=lambda ngram: len(ngram), reverse=True)]

        for smaller_tuple in a:
            smaller_text = ' '.join(smaller_tuple)
            smaller_size = len(smaller_tuple)

            inclusion_found = False
            for larger_size, larger_text in b_texts_sizes:
                if smaller_size > larger_size:
                    break

                if smaller_text in larger_text:
                    inclusion_found = True
                    break

            if not inclusion_found:
                return False

        return True

    def n_bin_attributes(self, data: list[PatternType], min_support: int = 0) -> int:
        """Count the number of attributes in the binary representation of `data`"""
        return sum(1 for _ in self.iter_bin_attributes(data, min_support))
