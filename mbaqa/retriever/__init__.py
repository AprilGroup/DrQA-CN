#!/usr/bin/env python3

import os

from .. import DATA_DIR


DEFAULTS = {
    'db_path': os.path.join(DATA_DIR, 'db/mba.db'),
    'tfidf_path': os.path.join(
        DATA_DIR,
        'retriever/model/mba-tfidf-ngram=2-hash=16777216-tokenizer=ltp-numdocs=78259.npz'
    ),
    'stopwords_path': os.path.join(DATA_DIR, 'stopwords/stopwords.txt'),
    'bm25_corpus_path': os.path.join(DATA_DIR, 'output/unigrams-num=76437.txt')
}


def set_default(key, value):
    global DEFAULTS
    DEFAULTS[key] = value


def get_class(name):
    if name == 'tfidf':
        return TfidfDocRanker
    if name == 'sqlite':
        return DocDB
    if name == 'bm25':
        return BM25DocRanker
    raise RuntimeError('Invalid retriever class: %s' % name)


from .doc_db import DocDB
from .tfidf_doc_ranker import TfidfDocRanker
from .bm25_doc_ranker import BM25DocRanker
