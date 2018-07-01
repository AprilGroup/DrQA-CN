#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Evaluate the accuracy of the DrQA retriever module."""

import regex as re
import logging
import argparse
import json
import time
import os

import sys
sys.path.append('/home/zrx/projects/MbaQA/')

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from mbaqa import retriever, tokenizers
from mbaqa.retriever import utils
from tqdm import tqdm

# ------------------------------------------------------------------------------
# Multiprocessing target functions.
# ------------------------------------------------------------------------------

PROCESS_TOK = None
PROCESS_DB = None


def init(tokenizer_class, tokenizer_opts, db_class, db_opts):
    global PROCESS_TOK, PROCESS_DB
    PROCESS_TOK = tokenizer_class(**tokenizer_opts)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)


def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
        )
    except BaseException:
        return False
    return pattern.search(text) is not None


def has_answer(answer, doc_id, match):
    """Check if a document contains an answer string.

    If `match` is string, token matching is done between the text and answer.
    If `match` is regex, we search the whole text with the regex.
    If 'match' is find, string matching is done between the text and answer.
    """
    global PROCESS_DB, PROCESS_TOK
    text = PROCESS_DB.get_doc_text(doc_id)
    text = utils.normalize(text)
    if match == 'string':
        # Answer is a list of possible strings
        text = PROCESS_TOK.tokenize(text).words(uncased=True)
        for single_answer in answer:
            single_answer = utils.normalize(single_answer)
            single_answer = PROCESS_TOK.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)
            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i: i + len(single_answer)]:
                    return True
    elif match == 'regex':
        # Answer is a regex
        single_answer = utils.normalize(answer[0])
        if regex_match(text, single_answer):
            return True
    elif match == 'find':
        single_answer = utils.normalize(re.sub('\s+', '', answer[0]))
        if re.sub('\s+', '', text).find(single_answer) != -1:
            return True
    return False


def get_score(answer_doc, match):
    """Search through all the top docs to see if they have the answer."""
    answer, (doc_ids, doc_scores) = answer_doc
    for doc_id in doc_ids:
        if has_answer(answer, doc_id, match):
            return 1
    return 0

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--model-type', type=str, default='tfidf')
    parser.add_argument('--doc-db', type=str, default=None,
                        help='Path to Document DB')
    parser.add_argument('--tokenizer', type=str, default='regexp')
    parser.add_argument('--n-docs', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--num-questions', type=int, default=None)
    parser.add_argument('--match', type=str, default='string',
                        choices=['regex', 'string', 'find'])
    parser.add_argument('--report-on', type=bool, default=False,
                        help='Whether or not output retriever test_report')
    parser.add_argument('--report-name', type=str, default=None,
                        help='Path to failing retrieved report.')
    parser.add_argument('--json-path', type=str, default=None,
                        help='Path to json format dataset for question->doc mapping.')
    parser.add_argument('--title-weight', type=float, default=0.8,
                        help='weight for title.')
    args = parser.parse_args()

    # start time
    start = time.time()

    # read all the data and store it
    logger.info('Reading data ...')
    questions = []
    answers = []
    counter = 0
    for line in open(args.dataset):
        data = json.loads(line)
        question = data['question']
        answer = data['answer']
        questions.append(question)
        answers.append(answer)
        counter += 1
        if counter >= args.num_questions:
            break

    # get the closest docs for each question.
    logger.info('Initializing ranker...')
    ranker = retriever.get_class('tfidf')(tfidf_path=args.model)
    ranker.strict = False  # return empty result for empty queries

    logger.info('Ranking...')
    closest_docs = ranker.batch_closest_docs(
        questions, k=args.n_docs, title_weight=args.title_weight, num_workers=args.num_workers
    )

    # closest_docs = []
    # with tqdm(total=len(questions)) as pbar:
    #     for question in questions:
    #         closest_docs.append(ranker.closest_docs_by_content_and_title(question, title_weight=args.title_weight, k=5))
    #         pbar.update()

    answers_docs = zip(answers, closest_docs)

    # define processes
    tok_class = tokenizers.get_class(args.tokenizer)
    tok_opts = {}
    db_class = retriever.DocDB
    db_opts = {'db_path': args.doc_db}
    processes = ProcessPool(
        processes=args.num_workers,
        initializer=init,
        initargs=(tok_class, tok_opts, db_class, db_opts)
    )

    # compute the scores for each pair, and print the statistics
    logger.info('Retrieving and computing scores...')
    get_score_partial = partial(get_score, match=args.match)
    scores = processes.map(get_score_partial, answers_docs)

    # get failing questions
    failing_questions = [(questions[i], answers[i][0], closest_docs[i]) for i in range(len(scores)) if scores[i] == 0]

    filename = os.path.basename(args.dataset)
    stats = (
        "\n" + "-" * 50 + "\n" +
        "{filename}\n" +
        "Examples:\t\t\t{total}\n" +
        "Matches in top {k}:\t\t{m}\n" +
        "Match % in top {k}:\t\t{p:2.2f}\n" +
        "Total time:\t\t\t{t:2.4f} (s)\n"
    ).format(
        filename=filename,
        total=len(scores),
        k=args.n_docs,
        m=sum(scores),
        p=(sum(scores) / len(scores) * 100),
        t=time.time() - start,
    )

    print(stats)

    # output report if needed.
    if args.report_on == True:
        # build question -> doc mapping
        question2doc = {}
        with open(args.json_path, encoding='utf8') as f:
            dataset_json = json.load(f)
        docs = dataset_json['data']
        for doc in docs:
            title = doc['title']
            for para in doc['paragraphs']:
                for qa in para['qas']:
                    question2doc[qa['question']] = title

        if args.report_name:
            filename = args.report_name + '-' + str(len(questions)) + '-titleweight-' + str(args.title_weight) + '.txt'
            with open(filename, 'w', encoding='utf8') as file:
                file.write(stats)
                file.write('-' * 50 + '\n')
                file.write('Failing examples and retrieved docs:\n')
                for question, answer, answer_docs in failing_questions:
                    label_doc_id = 'aaa'
                    file.write('question: ' + question + '\n')
                    # found = [str(answer[0]) for answer in answer_docs]
                    # file.write(''.join(found) + '\n')
                    file.write('label doc: ' + question2doc[question] + '\n')
                    file.write('answer: ' + answer + '\n')
                    file.write('retrieved docs: ' + ' '.join([str(answer) for answer in answer_docs]) + '\n')
                    file.write('-' * 50 + '\n')