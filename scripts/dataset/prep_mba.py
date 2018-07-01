"""A script to preprocess mba docs (maybe remove trailing chars in title, split content into
 paragraphs, normalize encoding of titles and paras) and output json file in the following
 format, for the following db construction:
 {
    {'id': title1, 'text': text of doc1},
    {'id': title2, 'text': text of doc2},
    ...
 }
"""

import json
import sys
import os
sys.path.append('/home/zrx/projects/MbaQA/')
import scripts.dataset.utils as utils
from tqdm import tqdm

MBA_ORIGIN_RAW = '../../data/mba/raw/mba_docs_full_134116.json'
MBA_DEF_RAW = '../../data/mba/raw/mba_def.json'
MBA_ORIGIN_PREPROCESSED = '../../data/mba/drqa_format/mba_origin.json'
MBA_DEF_PREPROCESSED = '../../data/mba/drqa_format/mba_def.json'


def preprocess_origin(input_path, output_path):
    """preprocess original mba docs"""
    # read raw docs
    docs = json.load(open(input_path, encoding='utf8'))
    # preprocess
    with tqdm(total=len(docs)) as pbar:
        with open(output_path, 'w', encoding='utf8') as file:
            for id, doc in docs.items():
                # remove trailing chars in title
                cleaned_title = utils.clean_title(doc['title'])
                # split content into paragraphs
                paras = '\n\n'.join([para for para in utils.split_doc(doc['content'])])
                # normalize encoding
                normalized_paras = utils.normalize(paras)
                # save
                file.write(json.dumps({'id': utils.normalize(cleaned_title),
                                       'text': normalized_paras}))
                file.write('\n')
                pbar.update()


def preprocess_def(input_path, output_path):
    """preprocess mba definition docs"""
    docs = json.load(open(input_path, encoding='utf8'))
    with tqdm(total=len(docs)) as pbar:
        with open(output_path, 'w', encoding='utf8') as file:
            for title, (content, definition, synonms) in docs.items():
                # split content into paragraphs
                paras = '\n\n'.join([para for para in utils.split_doc(content)])
                # normalize encoding
                normalized_paras = utils.normalize(paras)
                # save
                file.write(json.dumps({'id': utils.normalize(title),
                                       'text': normalized_paras}))
                file.write('\n')
                pbar.update()


def merge(mba_origin_json_path, mba_def_json_path, output_path):
    """
    merge preprocessed mba origin docs and mba def docs
    :param mba_origin_json_path: path to json file in the following format:
            {
                {'id': title1, 'text': text of doc1},
                {'id': title2, 'text': text of doc2},
                ...
            }
    :param mba_def_json_path: path to json file the same format as above.
    :return: None
    """
    # read in preprocessed origin docs
    if not os.path.exists(mba_origin_json_path):
        preprocess_origin(MBA_ORIGIN_RAW, mba_origin_json_path)
    origin_docs = utils.read_drqa_format_dataset_as_dict(mba_origin_json_path)

    # read in preprocessed def docs
    if not os.path.exists(mba_def_json_path):
        preprocess_def(MBA_DEF_RAW, mba_def_json_path)
    def_docs = utils.read_drqa_format_dataset_as_dict(mba_def_json_path)

    # merge docs, put mba def docs into mba origin docs, keep def doc if two docs are of
    # the same title
    for title, content in def_docs.items():
        origin_docs[title] = content

    # save merged docs in the format the same as the file to which mba_origin_json_path points
    with open(output_path, 'w', encoding='utf8') as file:
        for title, content in origin_docs.items():
            file.write(json.dumps({'id':title, 'text': content}))
            file.write('\n')


if __name__ == '__main__':
    # output_path = '../../data/mba/drqa_format/mba_merged.json'
    # merge(MBA_ORIGIN_PREPROCESSED, MBA_DEF_PREPROCESSED, output_path)
    # preprocess_origin(MBA_ORIGIN_RAW, MBA_ORIGIN_PREPROCESSED)
    # preprocess_def(MBA_DEF_RAW, MBA_DEF_PREPROCESSED)
    preprocess_origin(MBA_ORIGIN_RAW, MBA_ORIGIN_PREPROCESSED)