#!/usr/bin/env python3

"""A script to convert the default mba dataset to the format:

'{"title": "t1", "content": "c1"}'
...
'{"title": "tN", "content": "cN"}'

"""

import pandas as pd
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('output', type=str)
args = parser.parse_args()


# Read dataset
dataset = pd.read_json(args.input).T.set_index('title')

# Iterate and write title-content pairs
with open(args.output, 'w') as f:
    for row in dataset.iterrows():
        title = row[0]
        text = row[1]['content']
        f.write(json.dumps({'title': title, 'text': text}))
        f.write('\n')