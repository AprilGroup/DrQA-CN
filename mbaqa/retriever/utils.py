#!/usr/bin/env python3
"""Various retriever utilities."""

import regex
import unicodedata
import numpy as np
import scipy.sparse as sp
from sklearn.utils import murmurhash3_32
from . import DEFAULTS

# ------------------------------------------------------------------------------
# Sparse matrix saving/loading helpers.
# ------------------------------------------------------------------------------


def save_sparse_csr(filename, matrix, metadata=None):
    data = {
        'data': matrix.data,
        'indices': matrix.indices,
        'indptr': matrix.indptr,
        'shape': matrix.shape,
        'metadata': metadata,
    }
    np.savez(filename, **data)


def load_sparse_csr(filename):
    loader = np.load(filename)
    matrix = sp.csr_matrix((loader['data'], loader['indices'],
                            loader['indptr']), shape=loader['shape'])
    return matrix, loader['metadata'].item(0) if 'metadata' in loader else None

# ------------------------------------------------------------------------------
# bm25 corpus loading helpers.
# ------------------------------------------------------------------------------


def load_corpus(filename):
    corpus = []
    with open(filename, encoding='utf8') as file:
        for line in file:
            words = line.replace('\n', '').split(' ')
            corpus.append([word.strip() for word in words])
    return corpus


# ------------------------------------------------------------------------------
# Token hashing.
# ------------------------------------------------------------------------------


def hash(token, num_buckets):
    """Unsigned 32 bit murmurhash for feature hashing."""
    return murmurhash3_32(token, positive=True) % num_buckets


# ------------------------------------------------------------------------------
# Text cleaning.
# ------------------------------------------------------------------------------

def read_stopwords(filename=None):
    """Get stopwords from file, each word per line."""
    path = DEFAULTS['stopwords_path'] or filename
    stopwords = set()
    for line in open(path, 'r', encoding='utf8'):
        stopwords.add(line.replace('\n', '').strip())
    return stopwords


STOPWORDS = read_stopwords()


def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


def filter_word(text):
    """Take out stopwords, punctuation, and compound endings."""
    if text.lower() in STOPWORDS:
        return True
    return False


def filter_ngram(gram, mode='any'):
    """Decide whether to keep or discard an n-gram.

    Args:
        gram: list of tokens (length N)
        mode: Option to throw out ngram if
          'any': any single token passes filter_word
          'all': all tokens pass filter_word
          'ends': book-ended by filterable tokens
    """
    filtered = [filter_word(w) for w in gram]
    if mode == 'any':
        return any(filtered)
    elif mode == 'all':
        return all(filtered)
    elif mode == 'ends':
        return filtered[0] or filtered[-1]
    else:
        raise ValueError('Invalid mode: %s' % mode)
