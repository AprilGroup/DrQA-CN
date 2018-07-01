"""A Script to expand machine reading comprehension q-a dataset based on existing small
dataset using question templates."""


import pandas as pd
import sys
sys.path.append('/home/zrx/projects/MbaQA/')

from scripts.dataset.utils import *

# read in location type questions
dataset_path = '../../data/datasets/mba_location.json'
dataset = read_json(dataset_path)

# get all questions
# questions_path = '../../data/retriever/test_data/mba_location.txt'
# qas = []
# with open(questions_path) as file:
#     for line in file:
#         qas.append(json.loads(line))

#
qas = []
for doc in dataset['data']:
    for idx, para in enumerate(doc['paragraphs']):
        if len(para['qas']) > 0:
            for qa in para['qas']:
                for i in range(3):
                    qas.append({
                        'question': qa['question'],
                        'answer': qa['answers'][0]['text'],
                        'doc': doc['title'],
                        'para_idx': idx
                    })

output_path = '../../data/mba/location_questions.txt'
pd.DataFrame(qas).to_csv(output_path)