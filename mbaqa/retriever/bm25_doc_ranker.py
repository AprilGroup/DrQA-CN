"""BM25 based doc ranker."""

import numpy as np

from gensim import corpora
from gensim.summarization import bm25

from .. import tokenizers, retriever

from multiprocessing.pool import ThreadPool
from functools import partial


class BM25DocRanker(object):
    def __init__(self):
        self.tokenizer = tokenizers.get_class('ltp')()
        self.docdb = retriever.get_class('sqlite')()
        self.corpus = retriever.utils.load_corpus(retriever.DEFAULTS['bm25_corpus_path'])
        self.bm25model = bm25.BM25(self.corpus)
        self.avg_idf = sum(map(lambda k: float(self.bm25model.idf[k]), self.bm25model.idf.keys())) \
            / len(self.bm25model.idf.keys())
        self.doc_titles = self.docdb.get_doc_ids()
        self.idx2title = {idx: self.doc_titles[idx] for idx in range(len(self.doc_titles))}

    def closest_docs(self, query, k=1):
        query_unigrams = self.get_query_ngrams(query, 1)
        scores = self.bm25model.get_scores(query_unigrams, self.avg_idf)
        scores = np.array(scores)
        if len(scores) <= k:
            o_sort = np.argsort(-scores)
        else:
            o = np.argpartition(-scores, k)[0:k]
            o_sort = o[np.argsort(-scores[o])]

        doc_scores = scores[o_sort]
        doc_ids = [self.idx2title[idx] for idx in o_sort]

        return doc_ids, doc_scores

    def batch_closest_docs(self, queries, k=1, num_workers=None):
        """Process a batch of closest_docs requests multithreaded.
        Note: we can use plain threads here as scipy is outside of the GIL.
        """
        with ThreadPool(num_workers) as threads:
            closest_docs = partial(self.closest_docs, k=k)
            results = threads.map(closest_docs, queries)
        return results

    def get_query_ngrams(self, query, k=1):
        tokens = self.tokenizer.tokenize(query)
        return tokens.ngrams(
            n=k, uncased=True, filter_fn=retriever.utils.filter_ngram
        )