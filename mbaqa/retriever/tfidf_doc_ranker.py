#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Rank documents with TF-IDF scores"""

import logging
import numpy as np
import scipy.sparse as sp
import scipy.spatial as ss

import sys
sys.path.append('/home/zrx/projects/MbaQA/')

from multiprocessing.pool import ThreadPool
from functools import partial

from mbaqa.retriever import utils
from mbaqa.retriever import DEFAULTS
from mbaqa import tokenizers

from scipy.ndimage.interpolation import shift

logger = logging.getLogger(__name__)


class TfidfDocRanker(object):
    """Loads a pre-weighted inverted index of token/document terms.
    Scores new queries by taking sparse dot products.
    """

    def __init__(self, tfidf_path=None, strict=True):
        """
        Args:
            tfidf_path: path to saved model file
            strict: fail on empty queries or continue (and return empty result)
        """
        # Load from disk
        tfidf_path = tfidf_path or DEFAULTS['tfidf_path']
        logger.info('Loading %s' % tfidf_path)
        matrix, metadata = utils.load_sparse_csr(tfidf_path)
        self.doc_mat = matrix
        self.ngrams = metadata['ngram']
        self.hash_size = metadata['hash_size']
        self.tokenizer = tokenizers.get_class(metadata['tokenizer'])()
        self.doc_freqs = metadata['doc_freqs'].squeeze()
        self.doc_dict = metadata['doc_dict']
        self.num_docs = len(self.doc_dict[0])
        self.strict = strict

        self.csc_matrix = None
        self.unigrams = metadata['unigrams']
        self.bigrams = metadata['bigrams']
        self.hash2gram = metadata['hash2gram']
        self.title_tfidf = metadata['title_tfidf']

        self.titles_tokens = []
        self.title_csc_matrix = None
        self.titles_lens = None

    @staticmethod
    def scale(res):
        """scale all values in csr_matrix between 0 and 1"""
        if len(res.data) < 1:
            return res
        return res / max(res.data)

    def tokenize_titles(self):
        """Tokenize all doc titles."""
        for title in self.doc_dict[1]:
            self.titles_tokens.append(self.tokenizer.tokenize(title).words(uncased=True))

    def get_doc_index(self, doc_id):
        """Convert doc_id --> doc_index"""
        return self.doc_dict[0][doc_id]

    def get_doc_id(self, doc_index):
        """Convert doc_index --> doc_id"""
        return self.doc_dict[1][doc_index]

    def get_doc_scores(self, query):
        """Compute match scores for all articles' content for one query."""
        spvec = self.text2spvec(query)
        res = spvec * self.doc_mat
        return res

    def get_titles_lens(self):
        """Get the number of all titles' tokens(stopwords filtered)."""
        indptr = self.title_csc_matrix.indptr
        indptr_shifted = shift(indptr, -1, cval=np.NaN)
        self.titles_lens = (indptr_shifted - indptr)[:-1]
        self.titles_lens = sp.csr_matrix(
            (self.titles_lens, list(range(len(self.titles_lens))), [0, len(self.titles_lens)])
        )

    def common_words_ratio(self, word_list_1, word_list_2):
        """Compute common ratio based on number of common words of two word lists."""
        common = set(word_list_1) & set(word_list_2)
        return 2 * len(common) / (len(word_list_1) + len(word_list_2))

    def get_title_scores_by_sim(self, query):
        """Compute all title scores based on similarity between title tokens and query tokens,
        stop words filtered.
           sim = 2 * len(common words) / (len(title words) + len(query_words))
        """
        # get unique word hash ids
        words = self.parse(utils.normalize(query))
        wids = [utils.hash(w, self.hash_size) for w in words]

        if len(wids) == 0:
            if self.strict:
                raise RuntimeError('No valid word in: %s' % query)
            else:
                logger.warning('No valid word in: %s' % query)
                return sp.csr_matrix((0, self.num_docs))

        wids_unique, wids_counts = np.unique(wids, return_counts=True)

        # get query sparse vector
        query_spvec = sp.csr_matrix(
            ([1] * len(wids_unique), wids_unique, [0, len(wids_unique)]), shape=(1, self.hash_size)
        )

        # get all titles' length, and get title csc_matrix for similarity computing
        if self.title_csc_matrix is None:
            self.get_title_csc_matrix()
        if self.titles_lens is None:
            self.get_titles_lens()

        self.title_tfidf.data = np.array([1] * len(self.title_tfidf.data))

        titles_scores = query_spvec * self.title_tfidf

        query_len_spvec = sp.csr_matrix(
            ([len(wids_unique)] * self.num_docs,
             list(range(self.num_docs)),
             [0, self.num_docs]), shape=(1, self.num_docs)
        )

        denominator = self.titles_lens + query_len_spvec
        titles_scores = 2 * titles_scores / denominator

        titles_scores = sp.csr_matrix(titles_scores)

        return titles_scores

    def get_title_scores_by_doc_product(self, query, k=1):
        """Closest docs by dot product between query and titles
        in tfidf weighted word vector space."""
        spvec = self.text2spvec(query)
        res = spvec * self.title_tfidf

        if len(res.data) <= k:
            o_sort = np.argsort(-res.data)
        else:
            o = np.argpartition(-res.data, k)[0:k]
            o_sort = o[np.argsort(-res.data[o])]

        doc_scores = res.data[o_sort]
        doc_ids = [self.get_doc_id(i) for i in res.indices[o_sort]]
        return doc_ids, doc_scores

    def closest_docs_by_title(self, query, k=1):
        """Rank docs for query, based on similarity between
        query <=> doc title
        """
        title_scores = self.get_title_scores_by_sim(query)

        if len(title_scores.data) <= k:
            o_sort = np.argsort(-title_scores.data)
        else:
            o = np.argpartition(-title_scores.data, k)[0:k]
            o_sort = o[np.argsort(-title_scores.data[o])]

        scores = title_scores.data[o_sort]
        titles = [self.doc_dict[1][i] for i in title_scores.indices[o_sort]]
        return titles, scores

    def closest_docs_by_content_and_title(self, query, title_weight=0.6, k=1):
        """Rank docs for query, based on similarity between
        query <=> doc content
        query <=> doc title
        """
        # get sparse vector for query
        spvec = self.text2spvec(query)

        # get scores based on similarity between query and contents
        content_scores = spvec * self.doc_mat
        # scale doc scores
        content_scores = self.scale(content_scores)

        # get scores based on similarity between query and titles
        title_scores = self.get_title_scores_by_sim(query)
        # scale title scores
        title_scores = self.scale(title_scores)

        # combine content scores and title scores to get final scores
        try:
            res = (1 - title_weight) * content_scores + title_weight * title_scores
        except:
            res = content_scores

        # select k top-scored docs
        if len(res.data) <= k:
            o_sort = np.argsort(-res.data)
        else:
            o = np.argpartition(-res.data, k)[0:k]
            o_sort = o[np.argsort(-res.data[o])]

        doc_scores = res.data[o_sort]
        doc_ids = [self.get_doc_id(i) for i in res.indices[o_sort]]
        return doc_ids, doc_scores

    def closest_docs_by_content(self, query, k=1):
        """Closest docs by dot product between query and documents
        in tfidf weighted word vector space.
        """
        spvec = self.text2spvec(query)
        res = spvec * self.doc_mat

        if len(res.data) <= k:
            o_sort = np.argsort(-res.data)
        else:
            o = np.argpartition(-res.data, k)[0:k]
            o_sort = o[np.argsort(-res.data[o])]

        doc_scores = res.data[o_sort]
        doc_ids = [self.get_doc_id(i) for i in res.indices[o_sort]]
        return doc_ids, doc_scores

    def batch_closest_docs(self, queries, k=1, title_weight=0.6, num_workers=None):
        """Process a batch of closest_docs requests multithreaded.
        Note: we can use plain threads here as scipy is outside of the GIL.
        """
        with ThreadPool(num_workers) as threads:
            closest_docs = partial(self.closest_docs_by_content_and_title, title_weight=title_weight, k=k)
            results = threads.map(closest_docs, queries)
        return results

    def parse(self, query):
        """Parse the query into tokens (either ngrams or tokens)."""
        tokens = self.tokenizer.tokenize(query)
        return tokens.ngrams(n=self.ngrams, uncased=True,
                             filter_fn=utils.filter_ngram)

    def text2spvec(self, query):
        """Create a sparse tfidf-weighted word vector from query.

        tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
        """
        # Get hashed ngrams
        words = self.parse(utils.normalize(query))
        wids = [utils.hash(w, self.hash_size) for w in words]

        if len(wids) == 0:
            if self.strict:
                raise RuntimeError('No valid word in: %s' % query)
            else:
                logger.warning('No valid word in: %s' % query)
                return sp.csr_matrix((1, self.hash_size))

        # Count TF
        wids_unique, wids_counts = np.unique(wids, return_counts=True)
        tfs = np.log1p(wids_counts)

        # Count IDF
        Ns = self.doc_freqs[wids_unique]
        idfs = np.log((self.num_docs - Ns + 0.5) / (Ns + 0.5))
        idfs[idfs < 0] = 0

        # TF-IDF
        data = np.multiply(tfs, idfs)

        # One row, sparse csr matrix
        indptr = np.array([0, len(wids_unique)])
        spvec = sp.csr_matrix(
            (data, wids_unique, indptr), shape=(1, self.hash_size)
        )

        return spvec

    # ------------------------------------------------------------------------------
    # csc(Compressed Sparse Column Matrix) basic operations
    # ------------------------------------------------------------------------------

    def get_csc_matrix(self):
        """Get csc matrix for doc similarity computation."""
        self.csc_matrix = self.doc_mat.tocsc()

    def get_title_csc_matrix(self):
        """Get title csc matrix."""
        self.title_csc_matrix = self.title_tfidf.tocsc()

    @staticmethod
    def get_spvec_for_ith_col(i, matrix, dim):
        """Select the sparse vector for i-th col of scipy sparse csc_matrix, 0 based."""
        start_idx = matrix.indptr[i]
        end_idx = matrix.indptr[i + 1]
        # row indices for non-zero values in column i, 0 based
        indices = matrix.indices[start_idx: end_idx]
        # non-zero values in column i, 0 based
        data = matrix.data[start_idx:end_idx]

        # One row, sparse csr matrix
        indptr = np.array([0, len(data)])
        spvec = sp.csr_matrix(
            (data, indices, indptr), shape=(1, dim)
        )
        return spvec

    def get_weights_for_title(self, title):
        """Get ngram weights for a title. Sorted by weight in desending order"""
        if self.title_csc_matrix is None:
            self.get_title_csc_matrix()
        num_col = self.doc_dict[0][title]
        spvec = self.get_spvec_for_ith_col(num_col, self.title_csc_matrix, self.hash_size)

        return self.get_weights_for_spvec(spvec)

    def get_weights_for_doc(self, doc_id):
        """Get ngram weights for a doc. Sorted by weight in desending order."""
        if self.csc_matrix is None:
            self.get_csc_matrix()
        if doc_id not in self.doc_dict[0]:
            raise ValueError('invalid doc {}.'.format(doc_id))
        num_col = self.doc_dict[0][doc_id]
        spvec = self.get_spvec_for_ith_col(num_col, self.csc_matrix, self.hash_size)

        return self.get_weights_for_spvec(spvec)

    def get_weights_for_spvec(self, spvec):
        """Get ngram weights for a sparse vector. Sorted by weight in descending order."""
        gram2weight = {}
        for i, indice in enumerate(spvec.indices):
            if indice in self.hash2gram:
                gram2weight[list(self.hash2gram[indice])[0]] = spvec.data[i]
        return dict(sorted(gram2weight.items(), key=lambda x: x[1], reverse=True))


if __name__ == '__main__':
    tfidf_path = '../../data/retriever/model/mba-tfidf-ngram=2-hash=16777216-tokenizer=ltp-numdocs=78259.npz'
    # tfidf_path = '../../data/retriever/mba_def-tfidf-ngram=2-hash=16777216-tokenizer=ltp-numdocs=300.npz'
    ranker = TfidfDocRanker(tfidf_path=tfidf_path)
    # query = '按期纳税是指根据纳税义务发生的时间，通过确定纳税间隔期，实行按日纳税。按期纳税的纳税间隔期分为l天、3天、5天、10天、15天和1个月，共6种期限。'
    # query = '什么是国内纯销售'
    # doc_ids, doc_scores = ranker.closest_docs_by_content_and_title(query, k=5)
    # print(doc_ids, doc_scores)
    # title_scores = ranker.get_title_scores(query)
    # print(ranker.closest_docs_by_title(title_scores, 5))

    # print(ranker.get_title_scores_by_doc_product(query, k=5))
    from tqdm import tqdm
    hash2gram = {}
    with tqdm(total=(len(ranker.unigrams)+len(ranker.bigrams))) as pbar:
        for gram in ranker.unigrams + ranker.bigrams:
            hash = utils.hash(gram, ranker.hash_size)
            if hash not in hash2gram:
                hash2gram[hash] = set()
            hash2gram[hash].add(gram)
            pbar.update()
            # if len(hash2gram) > 10000:
            #     break
    with open('../../data/retriever/model/hash2gram.txt', 'w', encoding='utf8') as file:
        for hash, gram in hash2gram.items():
            file.write('{}-{}\n'.format(hash, '||'.join(list(gram))))

    # test if all title unigrams and bigrams are covered by doc contents
    # counter = 0
    # ranker.tokenize_titles()
    # for title_token in ranker.titles_tokens:
    #     for token in title_token:
    #         if token not in ranker.unigrams:
    #             counter += 1
    #             print(token)
    # print(counter)

    # spvec = ranker.text2spvec(query)
    # weights = ranker.get_weights_for_spvec(spvec)
    # print(spvec)
    # print(weights)

    # print(ranker.get_weights_for_doc('基金受益人'))

    # number of non-empty rows of word->doc tfidf matrix
    # non_empty_rows_num = len(np.unique(ranker.doc_mat.indptr))
    # print(non_empty_rows_num)
    # number of grams(unigram and bigram)
    # unigram_num = len(ranker.unigrams)
    # bigram_num = len(ranker.bigrams)
    # grams_num = unigram_num + bigram_num
    # print(unigram_num, bigram_num, grams_num)
    # # number of hash table collisions
    # collisions = grams_num - non_empty_rows_num
    # print(collisions)
'"四就"直拨'
'"T+0"交易'