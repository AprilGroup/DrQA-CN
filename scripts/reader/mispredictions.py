#!/usr/bin/env python3
"""A script to read in model predictions and ground truth, then output them with question and levenshtein
distance between ground truth(descending order) and prediction in the format:

question: q1
ground: g1
predic: p1
levdis: d1
----------------------------------------
question: q2
ground: g2
predic: p2
levdis: d2
----------------------------------------
...

"""

import json

prediction_file = '/home/zrx/projects/MbaQA/data/output/mba_def_dev-20180308-d004debe.preds'
ground_truth_file = '/home/zrx/projects/MbaQA/data/datasets/mba_def_dev.json'
output_file = '/home/zrx/projects/MbaQA/data/output/mispredictions_sorted.txt'


def levenshtein(first, second):
    """【编辑距离算法】 【levenshtein distance】 【字符串相似度算法】"""
    if len(first) > len(second):
        first, second = second, first
    if len(first) == 0:
        return len(second)
    if len(second) == 0:
        return len(first)
    first_length = len(first) + 1
    second_length = len(second) + 1
    distance_matrix = [list(range(second_length)) for _ in range(first_length)]
    # print distance_matrix
    for i in range(1, first_length):
        for j in range(1, second_length):
            deletion = distance_matrix[i-1][j] + 1
            insertion = distance_matrix[i][j-1] + 1
            substitution = distance_matrix[i-1][j-1]
            if first[i-1] != second[j-1]:
                substitution += 1
            distance_matrix[i][j] = min(insertion, deletion, substitution)
    return distance_matrix[first_length-1][second_length-1]


# read predictions
with open(prediction_file) as prediction_file:
    predictions = json.load(prediction_file)

# read groud truth
with open(ground_truth_file) as ground_truth_file:
    ground_truths = json.load(ground_truth_file)

# put question, ground_truth, prediction together, ordered by levenshtein distance
# between ground_truth and prediction, biggest distance first.
predict_result = []
for doc in ground_truths['data']:
    for para in doc['paragraphs']:
        for qa in para['qas']:
            ground_truth = qa['answers'][0]['text']
            prediction = predictions[qa['id']][0][0]
            predict_result.append((qa['question'],
                                   ground_truth,
                                   prediction,
                                   levenshtein(ground_truth, prediction)))
predict_result.sort(key=lambda x: x[3], reverse=True)


# output question, ground_truth[length], prediction[length]
with open(output_file, 'w', encoding='utf8') as file:
    for (question, ground_truth, prediction, leven_dis) in predict_result:
        file.write('question: ' + question + '\n')
        file.write('ground[{}]: {}\n'.format(len(ground_truth), ground_truth))
        file.write('predic[{}]: {}\n'.format(len(prediction), prediction))
        file.write('levdis: ' + str(leven_dis) + '\n')
        # for para in doc['paragraphs']:
        #     for qa in para['qas']:
        #         file.write('question: ' + qa['question'] + '\n'
        #                    + 'ground: ' + qa['answers'][0]['text'] + '\n'
        #                    + 'predic: ' + predictions[qa['id']][0][0] + '\n')
        file.write('-' * 100 + '\n')

