"""A script to read in doc titles to be dropped and drop near duplicates
from full dataset."""

import json

from tqdm import tqdm


# read in titles to be dropped
drop_list_file = '../../data/output/droplist-k=3-num=134974.txt'
drop_list = []
with open(drop_list_file, encoding='utf8') as file:
    for idx, line in enumerate(file):
        if idx > 3:  # skip header
            drop_list.append(line.replace('\n', '').strip())


# drop duplicates on merged dataset
input_path = '../../data/mba/drqa_format/mba_merged.json'
output_path = '../../data/mba/drqa_format/mba.json'
num_docs = 0
with tqdm(total=134802) as pbar:
    with open(output_path, 'w', encoding='utf8') as outf:
        with open(input_path, 'r', encoding='utf8') as inf:
            for line in inf:
                doc = json.loads(line)
                if doc['id'] not in drop_list:
                    outf.write(json.dumps(doc))
                    outf.write('\n')
                    num_docs += 1
                pbar.update()
print(num_docs)