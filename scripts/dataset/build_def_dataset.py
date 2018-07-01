"""A script to build document reader train/dev/test dataset from json encoded docs and templates."""

import json
import logging
import sys
sys.path.append('/home/zrx/projects/MbaQA/')
from scripts.dataset.utils import *


logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


def def_type_questions(term):
    """Generate multiple definition type questions for one term."""
    templates = ['什么是{}', '{}是什么', '{}的定义', '{}是什么意思', '解释一下{}', '{}解释']
    return [temp.format(term) for temp in templates]


def build_qa(docs, valid_titles):
    """
    Build qa dataset restricted to valid titles.
    :param docs: {
                    'title1': (definition1, content1, synonyms seperated by '*'),
                    'title2': (definition2, content2, sysnonym1*sysnonym2*sysnonym3),
                    ...
                 }
    :param valid_titles: ['valid_title1', 'valid_title2', ...]
    :return: [
                {
                    'title': title1,
                    'paragraphs':
                    [
                        [{'context': text of para 1 without qa pairs,
                        'qas': []
                        }],
                        [{'context': text of para 2 with qa pairs,
                        'qas': [
                                    {'q': question1, 'a': answer1},
                                    {'q': question2, 'a': answer2},
                                    ......
                                ]
                        }],
                        ......
                    ]
                }
                ...
             ]
    """
    dataset = []
    for title, (definition, content, synonms) in docs.items():
        if title in valid_titles and len(definition) > 1:
            answer_found = False
            doc = {'title': title, 'paragraphs': []}
            questions = def_type_questions(title)
            paras = split_doc(content)
            for para in paras:
                paragraph = {'context': para, 'qas': []}
                if para.find(definition) != -1 and not answer_found:
                    paragraph['qas'] = [{'q': question, 'a': definition} for question in questions]
                    answer_found = True
                doc['paragraphs'].append(paragraph)
            dataset.append(doc)
    return dataset


def save_dataset(dataset, save_path):
    with open(save_path, 'w', encoding='utf8') as file:
        for doc in dataset:
            file.write('title:{}\n'.format(doc['title']))
            for para in doc['paragraphs']:
                file.write('paragraph:{}\n'.format(para['context']))
                if len(para['qas']) > 0:
                    for qa in para['qas']:
                        file.write('q:{}\n'.format(qa['q']))
                        file.write('a:{}\n'.format(qa['a']))
            file.write('#####\n')


if __name__ == '__main__':
    MBA_DEF_RAW_PATH = '../../data/mba/raw/mba_def.json'
    TITLES_PATH = '../../data/mba/terms/mba_terms.txt'
    SAVE_PATH = '../../data/corpus/handwritten/mba_def.txt'

    # load raw data
    logger.info('Retrieving docs...')
    mba_def_docs = json.load(open(MBA_DEF_RAW_PATH, encoding='utf8'))
    for title in mba_def_docs:
        with open('../../data/mba/terms/mba_def_terms.txt', 'w', encoding='utf8') as file:
            file.write(title)
            file.write('\n')
    logger.info('{} docs loaded'.format(len(mba_def_docs)))

    # load valid titles
    logger.info('Retrieving valid titles...')
    valid_titles = read_lines(TITLES_PATH)
    #
    # # build q-a pairs
    logger.info('Building qa dataset...')
    dataset = build_qa(mba_def_docs, valid_titles)

    # # save dataset
    logger.info('Saving at {}...'.format(SAVE_PATH))
    save_dataset(dataset, SAVE_PATH)
