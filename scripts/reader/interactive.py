#!/usr/bin/env python3
"""A script to run the MbaQA reader model interactively."""

import torch
import code
import argparse
import logging
import prettytable
import time

import sys
sys.path.append('/home/zrx/projects/MbaQA/')

from mbaqa.reader import Predictor

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


# ------------------------------------------------------------------------------
# Commandline arguments & init
# ------------------------------------------------------------------------------


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None,
                    help='Path to model to use')
parser.add_argument('--tokenizer', type=str, default=None,
                    help=("String option specifying tokenizer type to use "
                          "(e.g. 'corenlp')"))
parser.add_argument('--no-cuda', action='store_true',
                    help='Use CPU only')
parser.add_argument('--gpu', type=int, default=-1,
                    help='Specify GPU device id to use')
parser.add_argument('--no-normalize', action='store_true',
                    help='Do not softmax normalize output scores.')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.set_device(args.gpu)
    logger.info('CUDA enabled (GPU %d)' % args.gpu)
else:
    logger.info('Running on CPU only.')

predictor = Predictor(args.model, args.tokenizer, num_workers=0,
                      normalize=not args.no_normalize)
if args.cuda:
    predictor.cuda()


# ------------------------------------------------------------------------------
# Drop in to interactive mode
# ------------------------------------------------------------------------------


def process(document, question, candidates=None, top_n=1):
    t0 = time.time()
    predictions = predictor.predict(document, question, candidates, top_n)
    table = prettytable.PrettyTable(['Rank', 'Span', 'Score'])
    for i, p in enumerate(predictions, 1):
        table.add_row([i, p[0], p[1]])
    print(table)
    print('Time: %.4f' % (time.time() - t0))


banner = """
MbaQA Interactive Document Reader Module
>> process(document, question, candidates=None, top_n=1)
>> usage()
"""


def usage():
    print(banner)


code.interact(banner=banner, local=locals())
