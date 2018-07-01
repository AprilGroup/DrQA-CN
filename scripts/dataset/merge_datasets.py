"""A script to merge multiple SQuAD-like datasets."""

import sys
sys.path.append('/home/zrx/projects/MbaQA/')
from scripts.dataset.utils import *


def merge_datasets(filenames):
    if len(filenames) < 2:
        raise RuntimeError('Two or more datasets are required.')
    merged = read_json(filenames[0])
    for filename in filenames[1:]:
        tmp_dataset = read_json(filename)
        merged['data'].extend(tmp_dataset['data'])
    return merged


if __name__ == '__main__':
    files = ['../../data/datasets/mba_def.json',
             '../../data/datasets/mba_org.json',
             '../../data/datasets/mba_location.json']
    output_path = '../../data/datasets/mba_def_org_location.json'
    merged = merge_datasets(files)
    save_json(merged, output_path)