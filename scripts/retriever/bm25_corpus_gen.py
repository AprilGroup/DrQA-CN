"""Document retriever based on bm25 for comparision with default weight-tfidf model."""

import sys
sys.path.append('/home/zrx/projects/MbaQA/')

from tqdm import tqdm
from gensim import corpora
from gensim.summarization import bm25

from mbaqa import retriever, tokenizers

docdb = retriever.get_class('sqlite')()
tokenizer = tokenizers.get_class('ltp')()

titles = docdb.get_doc_ids()[:]
IDX2TITLE = {idx: titles[idx] for idx in range(len(titles))}

stop_words_path = '../../data/stopwords/stopwords.txt'
stopwords = []
with open(stop_words_path, encoding='utf8') as file:
    for line in file:
        stopwords.append(line.replace('\n', '').strip())


corpus = []
with tqdm(total=len(titles)) as pbar:
    for title in titles:
        # Tokenize
        tokens = tokenizer.tokenize(retriever.utils.normalize(docdb.get_doc_text(title)))

        # Get ngrams from tokens, with stopword/punctuation filtering.
        unigrams = tokens.ngrams(
            n=1, uncased=True, filter_fn=retriever.utils.filter_ngram
        )

        corpus.append(unigrams)
        pbar.update()

unigrams_save_path = '../../data/output/unigrams-num=76437.txt'
with open(unigrams_save_path, 'w', encoding='utf8') as file:
    for words in corpus:
        file.write(' '.join(words))
        file.write('\n')


