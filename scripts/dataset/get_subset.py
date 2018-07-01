"""A script to read in SQuAD format dataset and keep valid titles only."""

import json
import sys
sys.path.append('/home/zrx/projects/MbaQA/')
import scripts.dataset.utils as utils
from mbaqa.retriever.tfidf_doc_ranker import TfidfDocRanker

# read in dataset
with open('../../data/datasets/mba_def_complete_train.json', encoding='utf8') as file:
    data = json.load(file)

# read in valid titles
ranker = TfidfDocRanker(tfidf_path='../../data/retriever/mba_76437-tfidf-ngram=2-hash=16777216-tokenizer=ltp-numdocs=76437.npz')
# valid_titles = utils.read_lines('../../data/mba/mba_76437_terms.txt')
valid_titles = ranker.doc_dict[1]

subset = {'data': [], 'version': 0.1}
for doc in data['data']:
    if doc['title'] in valid_titles:
        subset['data'].append(doc)

# save
with open('../../data/datasets/mba_def_complete_train_76437_subset.json', 'w', encoding='utf8') as file:
    json.dump(subset, file)


