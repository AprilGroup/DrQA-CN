"""cal """

import json
import logging
import numpy as np

import sys
sys.path.append('/home/zrx/projects/MbaQA/')

from mbaqa import retriever
from tqdm import tqdm


logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


DATASET_PATH = '../../data/retriever/mba_def_complete_train_76437_subset.txt'
JSON_PATH = '../../data/datasets/mba_def_complete_train_76437_subset.json'

# read in q-a pairs and store them
logger.info('Reading data ...')
questions = []
answers = []
counter = 0
for line in open(DATASET_PATH):
    data = json.loads(line)
    question = data['question']
    answer = data['answer']
    questions.append(question)
    answers.append(answer)
    counter += 1

# build question -> doc_id(title) mapping
question2title = {}
with open(JSON_PATH, encoding='utf8') as f:
    dataset_json = json.load(f)
docs = dataset_json['data']
for doc in docs:
    title = doc['title']
    for para in doc['paragraphs']:
        for qa in para['qas']:
            question2title[qa['question']] = title


# get doc-scores, title-scores, label-doc-index for each query
logger.info('Computing scores ...')
ranker = retriever.get_class('tfidf')()
ranker.strict = False

query_doc_scores = {}
all_title_scores = []
all_doc_scores = []
labels = []
with tqdm(total=len(questions)) as pbar:
    for q in questions:
        doc_scores = ranker.get_doc_scores(q)  # .toarray().reshape(76437,)
        all_doc_scores.append(doc_scores)
        # print('top: {} - {}'.format(np.max(doc_scores), ranker.doc_dict[1][np.where(doc_scores == np.max(doc_scores))[0][0]]))
        title_scores = ranker.get_title_scores(q)  # .toarray().reshape(76437,)
        all_title_scores.append(title_scores)
        # print('top: {} - {}'.format(np.max(title_scores), ranker.doc_dict[1][np.where(title_scores == np.max(title_scores))[0][0]]))

        label_doc = question2title[q]
        label_doc_index = ranker.doc_dict[0][label_doc]
        labels.append(label_doc_index)
        # query_doc_scores[q] = (doc_scores, title_scores, label_doc_index)

        pbar.update()
        if len(labels) > 100:
            break

data = {'title_scores': all_title_scores,
        'doc_scores': all_doc_scores,
        'labels': labels}

# save as numpy compressed npz file
np.savez('../../data/retriever/query_scores_sparse', **data)


# save as TFRecords
def saveTfRecords():
    pass