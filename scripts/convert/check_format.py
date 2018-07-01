"""检查数据集，该脚本最初目的为检查handwritten是否产生预期结果"""

import argparse
import json
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default=None)
args = parser.parse_args()


input = args.input or '../../data/datasets/mba_location.json'

with open(input, 'r', encoding='utf8') as file:
    dataset = json.load(file)

# for doc in dataset['data']:
#     print(doc['title'])
#     for para in doc['paragraphs']:
#         print(para['context'])
#         for qa in para['qas']:
#             print(qa['question'])
#             for answer in qa['answers']:
#                 print(answer['text'])
#                 print(answer['answer_start'])
    # break

# 检查文章，段落，问-答对的数量，段落及问题的长度
para_num = 0
qa_num = 0
answer_num = 0  # number of answers for one question, default 3
para_lens = []
answer_lens = []
for doc in dataset['data']:
    para_num += len(doc['paragraphs'])
    for para in doc['paragraphs']:
        qa_num += len(para['qas'])
        para_lens.append(len(para['context']))
        if len(para['qas']) > 0:
            answer_lens.append(len(para['qas'][0]['answers'][0]['text']))
print('-' * 50)
print('{} docs, {} paragraphs, {} q-a pairs.'.format(len(dataset['data']), para_num, qa_num))
print('-' * 50)
print('para lens distribution:')
print(pd.cut(pd.Series(sorted(para_lens)), 5).value_counts())
print('-' * 50)
print('answer lens distribution:')
print(pd.cut(pd.Series(sorted(answer_lens)), 5).value_counts())