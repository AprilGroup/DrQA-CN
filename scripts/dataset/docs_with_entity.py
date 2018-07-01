"""A script to find the docs containing specific type of named entity in dataset
for use of document reader train/dev/test dataset manually generation."""

import sys
sys.path.append('/home/zrx/projects/MbaQA/scripts/dataset')

from ner_detector import Detector
import json
import os
from tqdm import tqdm
import random
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


class DocFilter:
    """A Class to select docs by different standards in full doc set."""
    def __init__(self, docs_path):
        """
        :param docs_path: path to docs formatted(json):
                          [
                            {"id": title1, "text": text of doc1},
                            {"id": title2, "text": text of doc2}
                          ]
        """
        # read in all docs, for the following filter process
        self.docs = []
        logger.info('read docs...')
        with tqdm(total=109034) as pbar:
            for line in open(docs_path, encoding='utf8'):
                self.docs.append(json.loads(line))
                pbar.update()
        self.doc_num = len(self.docs)
        # init detector
        self.detector = Detector()

    def docs_with_entity(self, entity_type, num):
        """
        select docs randomly from all docs with specific entity type
        :param entity_type: {'person', 'date', 'location'}
        :param num: number of docs to select
        :return: {
                    'title1': (first_entity_1, doc_text_1),
                    'title2': (first_entity_2, doc_text_2),
                     ...
                 }
                 first entity for visual check
        """
        # entity type validation
        valid_entity_type = {'person', 'date', 'location', 'org'}
        if entity_type not in valid_entity_type:
            raise ValueError('illegal entity_type: {}, must be in {}'.
                             format(entity_type, valid_entity_type))

        # select docs with named entity of specific type
        logger.info('select docs with {}s...'.format(entity_type))
        selected_docs = {}
        with tqdm(total=num) as pbar:  # processing bar tool wrapper
            while len(selected_docs) < num:
                # select one doc randomly
                doc = self.docs[random.randint(0, self.doc_num-1)]

                if doc['id'] in selected_docs:
                    continue

                if entity_type == 'person':
                    if self.detector.contain_person(doc['text']):
                        persons = ' '.join(self.detector.entity_tokens_with_type('Nh', doc['text']))
                        selected_docs[doc['id']] = (persons, doc['text'])
                elif entity_type == 'location':
                    if self.detector.contain_location(doc['text']):
                        locations = ' '.join(self.detector.entity_tokens_with_type('Ns', doc['text']))
                        selected_docs[doc['id']] = (locations, doc['text'])
                elif entity_type == 'org':
                    if self.detector.contain_organization(doc['text']):
                        orgs = ' '.join(self.detector.entity_tokens_with_type('Ni', doc['text']))
                        selected_docs[doc['id']] = (orgs, doc['text'])
                elif entity_type == 'date':
                    if self.detector.contain_date(doc['text']):
                        dates = ' '.join(self.detector.dates(doc['text']))
                        selected_docs[doc['id']] = (dates, doc['text'])
                pbar.update()
        return selected_docs

    def save_docs(self, docs, save_dir, encoding='utf8'):
        """
        save docs in dir, one single txt file per doc named by doc title,
        first_entity is the first line of txt, for dataset generation convenience.
        :param docs: [
                        title_1: (first_entity_1, text1),
                        ...
                    ]
        :param save_dir: directory to save docs
        :param encoding:
        :return:
        """
        # new folder if not exist
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        # save docs in save_dir
        logger.info('save docs in {}'.format(save_dir))
        with tqdm(total=len(docs)) as pbar:
            for title in docs:
                content = docs[title][1]
                entities = docs[title][0]
                if title.find('/') != -1:
                    title = title.replace('/', '-')
                file_name = title + '.txt'
                with open(os.path.join(save_dir, file_name), 'w', encoding=encoding) as file:
                    file.write(entities + '\n')
                    file.write('-' * 100 + '\n')
                    for para in content.split('\n\n'):
                        file.write('paragraph:' + para + '\n')
                        file.write('q:\n')
                        file.write('a:\n')
                pbar.update()


if __name__ == '__main__':
    # init DocFilter
    mba_docs_path = '../../data/mba/raw/mba_def.json'
    doc_filter = DocFilter(mba_docs_path)

    # select docs with location
    # location_docs_dir = '../../data/mba/MBA_500_LOCATION'
    # docs_location = doc_filter.docs_with_entity('location', 500)
    # doc_filter.save_docs(docs_location, location_docs_dir)

    # select docs with person name
    # person_docs_dir = '../../data/mba/MBA_500_PERSON'
    # docs_person = doc_filter.docs_with_entity('person', 500)
    # doc_filter.save_docs(docs_person, person_docs_dir)

    # select docs with organization name
    # organization_docs_dir = '../../data/mba/MBA_500_ORG'
    # docs_org = doc_filter.docs_with_entity('org', 500)
    # doc_filter.save_docs(docs_org, organization_docs_dir)

    # select docs with date
    date_docs_dir = '../../data/mba/MBA_500_DATE'
    docs_date = doc_filter.docs_with_entity('date', 500)
    doc_filter.save_docs(docs_date, date_docs_dir)
