"""A script for detect near duplicate docs(or docs almost the same) in the full doc set,
group near docs' titles together."""

import os
import sys
import json
import argparse
import logging
sys.path.append('/home/zrx/projects/MbaQA/')

from mbaqa import retriever
from mbaqa.retriever import utils
from mbaqa import tokenizers


from tqdm import tqdm
from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from simhash import Simhash, SimhashIndex

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

# ------------------------------------------------------------------------------
# Multiprocessing functions
# ------------------------------------------------------------------------------

DOC2IDX = None
PROCESS_TOK = tokenizers.get_class('ltp')()
PROCESS_DB = None


def init(tokenizer_class, db_class, db_opts):
    global PROCESS_TOK, PROCESS_DB
    PROCESS_TOK = tokenizer_class()
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)


def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_id)


def tokenize(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)


# ------------------------------------------------------------------------------
# Build title --> Simhash.
# ------------------------------------------------------------------------------


def get_features(text):
    """Get Unigrams from text with stopword/punctuation filtering."""
    global PROCESS_TOK
    # Tokenize
    tokens = tokenize(retriever.utils.normalize(text))

    # Get ngrams from tokens, with stopword/punctuation filtering.
    ngrams = tokens.ngrams(
        n=1, uncased=True, filter_fn=retriever.utils.filter_ngram
    )
    return ngrams


def read_drqa_format_dataset_as_dict(filepath):
    dataset = {}
    with open(filepath, encoding='utf8') as file:
        for line in file:
            doc = json.loads(line)
            dataset[doc['id']] = doc['text']
    return dataset


def read_docs_from_db(args, db, db_opts):
    """retrieve docs from sqlite db"""
    logger.info('Retrieving docs from db...')

    data = {}
    db_class = retriever.get_class(db)
    with db_class(**db_opts) as doc_db:
        titles = doc_db.get_doc_ids()
        # control number for test
        if args.num_docs > 0:
            titles = titles[:args.num_docs]
        for title in titles:
            data[title] = doc_db.get_doc_text(title)
    return data


def title2text_dic_2_title2hash_dic(data, title):
    """Map doc text to simhash, keep title unchanged."""
    return title, Simhash(get_features(data[title]))


def build_simhash(args, source='db'):
    title2text = {}
    titles = {}
    # retrieve docs from db
    if source == 'db':
        title2text = read_docs_from_db(args, args.doc_db, args.db_opts)
    # retrieve docs from json
    elif source == 'json':
        title2text = read_drqa_format_dataset_as_dict(args.json_path)
        titles = list(title2text.keys())
        # control number when testing code
        if args.num_docs > 0:
            titles = titles[:args.num_docs]
            title2text = {title: title2text[title] for title in titles}

    logger.info('Mapping...')
    title2hash = []
    tok_class = tokenizers.get_class(args.tokenizer)
    # multiprocessing
    if args.work_type == 'multi':
        # Setup worker pool
        workers = ProcessPool(
            args.num_workers,
            initializer=init,
            initargs=(tok_class, retriever.get_class(args.doc_db), {'db_path': args.doc_db})
        )
        step = max(int(len(title2text) / 10), 1)
        batches = [titles[i:i + step] for i in range(0, len(titles), step)]
        _convert = partial(title2text_dic_2_title2hash_dic, title2text)

        # map doc text to simhash using multiprocess

        for i, batch in enumerate(batches):
            logger.info('-' * 25 + 'Batch %d/%d' % (i + 1, len(batches)) + '-' * 25)
            for title, simhash in workers.imap_unordered(_convert, batch):
                title2hash.append((title, simhash))
        workers.close()
        workers.join()

    # single processing
    elif args.work_type == 'single':
        with tqdm(total=len(title2text)) as pbar:
            for (k, v) in title2text.items():
                title2hash.append(title2text_dic_2_title2hash_dic(title2text, k))
                pbar.update()
    return title2hash


def save_duplicates(save_path, text2hash_dict, k=5):
    """Group similar docs' title"""
    # Construct SimhashIndex object for similar docs detection. k is tolerance.
    index = SimhashIndex(text2hash_dict, k=k)

    done = list()
    with tqdm(total=len(text2hash_dict)) as pbar:
        with open(save_path, 'w', encoding='utf8') as file:
            for i in range(len(text2hash_dict)-1):
                # get near duplicates
                near_dups = index.get_near_dups(text2hash_dict[i][1])
                # near dups includes origin title, len > 1 requested
                if len(near_dups) > 1 and text2hash_dict[i][0] not in done:
                    for title in near_dups:
                        file.write(title)
                        file.write('\n')
                    file.write('#' * 5 + '\n')
                    done.extend(near_dups)
                pbar.update()


def count_groups(file_path):
    """Count near dups groups from file. Groups seperated by '#####'"""
    count = 0
    with open(file_path, 'r', encoding='utf8') as file:
        for line in file:
            if line.startswith('#####'):
                count += 1
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--doc-db', type=str, default=None,
                        help='Path to sqlite db holding document texts.')
    parser.add_argument('--json-path', type=str, default=None,
                        help='Path to json file holding document texts.')
    parser.add_argument('--out-dir', type=str, default='../../data/output/',
                        help='Directory for saving grouped titles.')
    parser.add_argument('--k', type=int, default=3,
                        help='Tolerance factor of simhash index. Small number means strict.')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of CPU processes (for tokenizing, etc)')
    parser.add_argument('--num-docs', type=int, default=-1,
                        help='Number of docs to build simhash. -1 means all docs.')
    parser.add_argument('--tokenizer', type=str, default='ltp')
    parser.add_argument('--work-type', type=str, default='multi',
                        choices=['multi', 'single'])
    parser.add_argument('--source', type=str, default='db',
                        choices=['db', 'json'])
    args = parser.parse_args()

    text2hash = build_simhash(
        args, args.source
    )

    file_name = 'duplicates-k={}-num={}.txt'.format(args.k, len(text2hash))
    logger.info('Saving results at {}'.format(file_name))
    save_duplicates(os.path.join(args.out_dir, file_name), text2hash, args.k)