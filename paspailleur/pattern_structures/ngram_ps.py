import difflib
from collections import deque
from functools import reduce
from itertools import product
from typing import Iterator, Iterable

from bitarray import frozenbitarray as fbarray, bitarray
from bitarray.util import zeros as bazeros

#from .abstract_ps import AbstractPS
from paspailleur.pattern_structures.abstract_ps import AbstractPS


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

    def iter_bin_attributes(self, data: list[PatternType], min_support: int = 0) -> Iterator[tuple[PatternType, fbarray]]:
        """Iterate binary attributes obtained from `data` (from the most general to the most precise ones)

        :parameter
            data: list[PatternType]
             list of object descriptions
        :return
            iterator of (description: PatternType, extent of the description: frozenbitarray)
        """
        def compute_words_extents(ptrns):
            n_patterns = len(ptrns)
            words_extents: dict[str, bitarray] = {}
            for i, pattern in enumerate(ptrns):
                words = {word for ngram in pattern for word in ngram}
                for word in words:
                    if word not in words_extents:
                        words_extents[word] = bazeros(n_patterns)
                    words_extents[word][i] = True
            return words_extents

        def compute_total_pattern(words_exts, ptrns):
            total_pattern = set()
            if any(ext.all() for ext in words_exts.values()):
                total_pattern = reduce(self.join_patterns, ptrns[1:], ptrns[0])
            return total_pattern

        def drop_rare_words(words_exts, min_supp):
            rare_words = (w for w, ext in words_exts.items() if ext.count() < min_supp)
            for rare_word in rare_words:
                del words_exts[rare_word]

        def setup_search_space(prev_lvl_ngrams, n_words_):
            if n_words_ <= len(prev_lvl_ngrams):
                for prev_ngrm, word_i in product(prev_lvl_ngrams, range(n_words_)):
                    if prev_ngrm[1:] + (word_i,) in prev_lvl_ngrams:
                        yield prev_ngrm, word_i

            # len(prev_level_ngrams) < n_words:
            prefixes_dict: dict[tuple[int, ...], list[int]] = {}
            for prev_ngrm in prev_lvl_ngrams:
                prefix, word_i = prev_ngrm[:-1], prev_ngrm[-1]
                if prefix not in prefixes_dict:
                    prefixes_dict[prefix] = []
                prefixes_dict[prefix].append(word_i)

            for prev_ngrm in prev_lvl_ngrams:
                if prev_ngrm[1:] not in prefixes_dict:
                    continue

                for word_i in prefixes_dict[prev_ngrm[1:]]:
                    yield prev_ngrm, word_i

        def refine_extent(ext, ngram_verb, min_supp, ptrns):
            supp_delta = ext.count() - max(min_supp, 1)
            # the use of str `words` is important because of the way is_less_precise function works
            for ptrn_i in ext.itersearch(True):
                if self.is_less_precise({ngram_verb}, ptrns[ptrn_i]):
                    continue
                # new_ngram is not contained in i-th pattern
                ext[ptrn_i] = False
                supp_delta -= 1
                if supp_delta < 0:
                    break
            else:  # no break, i.e. new_ext.count() >= min_supp
                return ext
            return None

        words_extents = compute_words_extents(data)
        yield compute_total_pattern(words_extents, data), fbarray(~bazeros(len(data)))

        drop_rare_words(words_extents, min_support)
        words, extents = zip(*words_extents.items())
        n_words = len(words)

        ngrams = {(word_i,): ext for word_i, ext in enumerate(extents)}
        for (word_i,), ext in ngrams.items():
            yield (words[word_i],), fbarray(ext)

        while ngrams:
            prev_ngrams, ngrams = ngrams, {}

            search_space = setup_search_space(prev_ngrams, n_words)
            for prev_ngram, word_i in search_space:
                new_ngram: tuple[int, ...] = prev_ngram + (word_i,)
                proto_extent = prev_ngrams[prev_ngram] & prev_ngrams[new_ngram[1:]]
                if not proto_extent.any() or proto_extent.count() < min_support:
                    continue

                new_ngram_verb = tuple([words[word_j] for word_j in new_ngram])
                new_extent = refine_extent(proto_extent, new_ngram_verb, min_support, data)
                if new_extent is None:
                    continue

                ngrams[new_ngram] = new_extent
                if new_extent != extents[word_i] and new_extent != prev_ngrams[prev_ngram]\
                        and len(new_ngram) >= self.min_n:
                    yield new_ngram_verb, fbarray(new_extent)

        if min_support == 0:
            yield None, fbarray(bazeros(len(data)))

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


if __name__ == '__main__':
    patterns = [{('hello', 'world')}, {('hello', 'there')}, {('hi',)}, set()]

    ps = NgramPS()
    for subpattern, extent in ps.iter_bin_attributes(patterns):
        print(subpattern, extent)