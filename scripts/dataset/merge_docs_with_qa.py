"""A script to walk through a dir containing multiple txt files(each for one doc) and
merge them to one file in the following format which can then be transformed to
meet the requirement of the training process of document reader by
MbaQA/scripts/convert/handwritten.py:

title: text of title 1
paragraph: text of paragraph 1 with q-a pairs
q:
a:
..
paragraph: text of paragraph 2 without q-a pairs
paragraph: text of paragraph 3 without q-a pairs
#####
title: text of title 2
paragraph: text of paragraph 1 with or without q-a pairs
...

"""


import os

from . import utils


def merge(input_dir, output_path, valid_file_set):
    """
    Merge multiple txt files into a single text file.
    :param input_dir: dir containing multiple text files which may contain q-a pairs
    :param output_path: path to the single txt file with the format described in the
           header of the script.
    :param valid_file_set: only keep files in valid_file_set, list[str]
    :return:
    """
    with open(output_path, 'w', encoding='utf8') as outf:
        for (root, dirs, files) in os.walk(input_dir):
            for file in files:
                if file.replace('.txt', '') in valid_file_set:
                    with open(os.path.join(root, file), 'r', encoding='utf8') as f:
                        # write title
                        outf.write('title:' + file.replace('.txt', '') + '\n')
                        # write paragraphs and q-a pairs
                        for idx, line in enumerate(f):
                            if line.startswith('paragraph'):
                                outf.write(line)
                            elif line.startswith('q:') and len(line) > 3:
                                outf.write(line)
                            elif line.startswith('a:') and len(line) > 3:
                                outf.write(line)
                        outf.write('#' * 5 + '\n')
    return 0


TITLES_PATH = '../../data/mba/mba_76437_terms.txt'
INPUT_DIR = '../../data/corpus/handwritten/MBA_DATE'
OURPUT_PATH = '../../data/corpus/handwritten/mba_date_76437_subset.txt'

valid_file_set = utils.read_lines(TITLES_PATH)
merge(INPUT_DIR, OURPUT_PATH, valid_file_set)