#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to make and save model predictions on an input dataset."""

import os
import regex
import time
import torch
import argparse
import logging
import json

import sys
sys.path.append('/home/zrx/projects/MbaQA/')

from tqdm import tqdm
from mbaqa.reader import Predictor

failing_afterprocess = open('fail.txt', 'w', encoding='utf8')


logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='../../data/datasets/mba_def_complete_test.json',
                    help='SQuAD-like dataset to evaluate on')
parser.add_argument('--model', type=str, default='../../data/reader/single.mdl',
                    help='Path to model to use')
parser.add_argument('--embedding-file', type=str, default=None,
                    help=('Expand dictionary to use all pretrained '
                          'embeddings in this file.'))
parser.add_argument('--out-dir', type=str, default='../../data/output/',
                    help=('Directory to write prediction file to '
                          '(<dataset>-<model>.preds)'))
parser.add_argument('--tokenizer', type=str, default=None,
                    help=("String option specifying tokenizer type to use "
                          "(e.g. 'corenlp')"))
parser.add_argument('--num-workers', type=int, default=None,
                    help='Number of CPU processes (for tokenizing, etc)')
parser.add_argument('--no-cuda', action='store_true',
                    help='Use CPU only')
parser.add_argument('--gpu', type=int, default=-1,
                    help='Specify GPU device id to use')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Example batching size')
parser.add_argument('--top-n', type=int, default=1,
                    help='Store top N predicted spans per example')
parser.add_argument('--official', action='store_true',
                    help='Only store single top span instead of top N list')
parser.add_argument('--after-process', type=bool, default=False,
                    help="Whether normalize answer to complete sentence.")
args = parser.parse_args()
t0 = time.time()

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.set_device(args.gpu)
    logger.info('CUDA enabled (GPU %d)' % args.gpu)
else:
    logger.info('Running on CPU only.')

predictor = Predictor(
    args.model,
    args.tokenizer,
    args.embedding_file,
    args.num_workers,
)
if args.cuda:
    predictor.cuda()


# ------------------------------------------------------------------------------
# Read in dataset and make predictions.
# ------------------------------------------------------------------------------

def after_process(answer_span, context):
    """Nomalize answer to get complete sentence(s)."""
    # --------------------------------------
    # maybe tuncate or extend answer head
    # --------------------------------------

    # maybe remove characters before the first zh character
    first_zh = regex.search('[\u4e00-\u9fa5]', answer_span)
    if first_zh and first_zh.start() > 0:
        answer_span = answer_span[first_zh.start():]

    # maybe add missing beginning of the first sentence
    answer_start = context.find(answer_span)
    if answer_start > 0:  # there are chars before answer
        # reverse search the last punc before answer span
        search_span = context[:answer_start]
        last_punc_before_answer = regex.search(r'(?r)\p{P}', search_span)
        # punc found, append chars to the beginning of the original answer span
        if last_punc_before_answer:
            answer_span = context[last_punc_before_answer.start() + 1: answer_start + len(answer_span)]
        # punc not found, answer from context start pos
        else:
            answer_span = context[:answer_start + len(answer_span)]

    # --------------------------------------
    # maybe tuncate or extend answer tail
    # --------------------------------------

    # update answer match object
    answer_start = context.find(answer_span)
    # reverse search the first period
    last_period = regex.search(r'(?r)。', answer_span)
    if last_period:  # period found
        # truncate the answer to the last peroid
        answer_span = answer_span[:last_period.end()]
    else:
        # or extend the answer span to the nearest period
        search_span = context[answer_start + len(answer_span):]
        first_period_after_answer = regex.search(r'。', search_span)
        if first_period_after_answer:
            answer_span = context[answer_start: first_period_after_answer.start()]

    return answer_span


examples = []
qids = []
with open(args.dataset) as f:
    data = json.load(f)['data']
    for article in data:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                qids.append(qa['id'])
                examples.append((context, qa['question']))

results = {}
for i in tqdm(range(0, len(examples), args.batch_size)):
    predictions = predictor.predict_batch(
        examples[i:i + args.batch_size], top_n=args.top_n
    )
    for j in range(len(predictions)):
        # Official eval expects just a qid --> span
        if args.official:
            results[qids[i + j]] = after_process(predictions[j][0][0], examples[i + j][0])

        # Otherwise we store top N and scores for debugging.
        else:
            if args.after_process:
                results[qids[i + j]] = [(after_process(p[0], examples[i + j][0]), float(p[1])) for p in predictions[j]]
            else:
                results[qids[i + j]] = [(p[0], float(p[1])) for p in predictions[j]]

model = os.path.splitext(os.path.basename(args.model or 'default'))[0]
basename = os.path.splitext(os.path.basename(args.dataset))[0]
outfile = os.path.join(args.out_dir, basename + '-' + model + '-afterprocess=' + str(args.after_process) + '.preds')

logger.info('Writing results to %s' % outfile)
with open(outfile, 'w') as f:
    json.dump(results, f)

logger.info('Total time: %.2f' % (time.time() - t0))
