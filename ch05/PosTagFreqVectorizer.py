# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import re
from operator import itemgetter
from collections import Mapping

import scipy.sparse as sp

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import strip_accents_ascii, strip_accents_unicode

import nltk

from collections import Counter

try:
    import ujson as json  # UltraJSON if available
except:
    import json

poscache_filename = "poscache.json"


class PosCounter(Counter):

    def __init__(self, iterable=(), normalize=True, poscache=None, **kwargs):
        self.n_sents = 0
        self.normalize = normalize

        self.poscache = poscache

        super(PosCounter, self).__init__(iterable, **kwargs)

    def update(self, other):
        """Adds counts for elements in other"""
        if isinstance(other, self.__class__):
            self.n_sents += other.n_sents
            for x, n in other.items():
                self[x] += n
        else:
            for sent in other:
                self.n_sents += 1

                if self.poscache is not None:
                    if sent in self.poscache:
                        tags = self.poscache[sent]
                    else:
                        self.poscache[sent] = tags = nltk.pos_tag(
                            nltk.word_tokenize(sent))
                else:
                    tags = nltk.pos_tag(nltk.word_tokenize(sent))

                for x in tags:
                    tok, tag = x
                    self[tag] += 1

            if self.normalize:
                for x, n in self.items():
                    self[x] /= float(self.n_sents)


class PosTagFreqVectorizer(BaseEstimator):

    """
    Convert a collection of raw documents to a matrix Pos tag frequencies
    """

    def __init__(self, input='content', charset='utf-8',
                 charset_error='strict', strip_accents=None,
                 vocabulary=None,
                 normalize=True,
                 dtype=float):

        self.input = input
        self.charset = charset
        self.charset_error = charset_error
        self.strip_accents = strip_accents
        if vocabulary is not None:
            self.fixed_vocabulary = True
            if not isinstance(vocabulary, Mapping):
                vocabulary = dict((t, i) for i, t in enumerate(vocabulary))
            self.vocabulary_ = vocabulary
        else:
            self.fixed_vocabulary = False

        try:
            self.poscache = json.load(open(poscache_filename, "r"))
        except IOError:
            self.poscache = {}

        self.normalize = normalize
        self.dtype = dtype

    def write_poscache(self):
        json.dump(self.poscache, open(poscache_filename, "w"))

    def decode(self, doc):
        """Decode the input into a string of unicode symbols

        The decoding strategy depends on the vectorizer parameters.
        """
        if self.input == 'filename':
            doc = open(doc, 'rb').read()

        elif self.input == 'file':
            doc = doc.read()

        if isinstance(doc, bytes):
            doc = doc.decode(self.charset, self.charset_error)
        return doc

    def build_preprocessor(self):
        """Return a function to preprocess the text before tokenization"""

        # unfortunately python functools package does not have an efficient
        # `compose` function that would have allowed us to chain a dynamic
        # number of functions. However the however of a lambda call is a few
        # hundreds of nanoseconds which is negligible when compared to the
        # cost of tokenizing a string of 1000 chars for instance.
        noop = lambda x: x

        # accent stripping
        if not self.strip_accents:
            strip_accents = noop
        elif hasattr(self.strip_accents, '__call__'):
            strip_accents = self.strip_accents
        elif self.strip_accents == 'ascii':
            strip_accents = strip_accents_ascii
        elif self.strip_accents == 'unicode':
            strip_accents = strip_accents_unicode
        else:
            raise ValueError('Invalid value for "strip_accents": %s' %
                             self.strip_accents)

        only_prose = lambda s: re.sub('<[^>]*>', '', s).replace("\n", " ")

        return lambda x: strip_accents(only_prose(x))

    def build_tokenizer(self):
        """Return a function that split a string in sequence of tokens"""
        return nltk.sent_tokenize

    def build_analyzer(self):
        """Return a callable that handles preprocessing and tokenization"""

        preprocess = self.build_preprocessor()

        tokenize = self.build_tokenizer()

        return lambda doc: tokenize(preprocess(self.decode(doc)))

    def _term_count_dicts_to_matrix(self, term_count_dicts):
        i_indices = []
        j_indices = []
        values = []
        vocabulary = self.vocabulary_

        for i, term_count_dict in enumerate(term_count_dicts):
            for term, count in term_count_dict.items():
                j = vocabulary.get(term)
                if j is not None:
                    i_indices.append(i)
                    j_indices.append(j)
                    values.append(count)
            # free memory as we go
            term_count_dict.clear()

        shape = (len(term_count_dicts), max(vocabulary.values()) + 1)
        spmatrix = sp.csr_matrix((values, (i_indices, j_indices)),
                                 shape=shape, dtype=self.dtype)
        return spmatrix

    def fit(self, raw_documents, y=None):
        """Learn a vocabulary dictionary of all tokens in the raw documents

        Parameters
        ----------
        raw_documents: iterable
            an iterable which yields either str, unicode or file objects

        Returns
        -------
        self
        """
        self.fit_transform(raw_documents)
        return self

    def fit_transform(self, raw_documents, y=None):
        """Learn the vocabulary dictionary and return the count vectors

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        raw_documents: iterable
            an iterable which yields either str, unicode or file objects

        Returns
        -------
        vectors: array, [n_samples, n_features]
        """
        if self.fixed_vocabulary:
            # No need to fit anything, directly perform the transformation.
            # We intentionally don't call the transform method to make it
            # fit_transform overridable without unwanted side effects in
            # TfidfVectorizer
            analyze = self.build_analyzer()
            term_counts_per_doc = [PosCounter(analyze(doc), normalize=self.normalize, poscache=self.poscache)
                                   for doc in raw_documents]
            return self._term_count_dicts_to_matrix(term_counts_per_doc)

        self.vocabulary_ = {}
        # result of document conversion to term count dicts
        term_counts_per_doc = []
        term_counts = Counter()

        analyze = self.build_analyzer()

        for doc in raw_documents:
            term_count_current = PosCounter(
                analyze(doc), normalize=self.normalize, poscache=self.poscache)
            term_counts.update(term_count_current)

            term_counts_per_doc.append(term_count_current)

        self.write_poscache()

        terms = set(term_counts)

        # store map from term name to feature integer index: we sort the term
        # to have reproducible outcome for the vocabulary structure: otherwise
        # the mapping from feature name to indices might depend on the memory
        # layout of the machine. Furthermore sorted terms might make it
        # possible to perform binary search in the feature names array.
        self.vocabulary_ = dict(((t, i) for i, t in enumerate(sorted(terms))))

        return self._term_count_dicts_to_matrix(term_counts_per_doc)

    def transform(self, raw_documents):
        """Extract token counts out of raw text documents using the vocabulary
        fitted with fit or the one provided in the constructor.

        Parameters
        ----------
        raw_documents: iterable
            an iterable which yields either str, unicode or file objects

        Returns
        -------
        vectors: sparse matrix, [n_samples, n_features]
        """
        if not hasattr(self, 'vocabulary_') or len(self.vocabulary_) == 0:
            raise ValueError("Vocabulary wasn't fitted or is empty!")

        # raw_documents can be an iterable so we don't know its size in
        # advance

        # XXX @larsmans tried to parallelize the following loop with joblib.
        # The result was some 20% slower than the serial version.
        analyze = self.build_analyzer()
        term_counts_per_doc = [Counter(analyze(doc)) for doc in raw_documents]
        return self._term_count_dicts_to_matrix(term_counts_per_doc)

    def get_feature_names(self):
        """Array mapping from feature integer indices to feature name"""
        if not hasattr(self, 'vocabulary_') or len(self.vocabulary_) == 0:
            raise ValueError("Vocabulary wasn't fitted or is empty!")

        return [t for t, i in sorted(iter(self.vocabulary_.items()),
                                     key=itemgetter(1))]
