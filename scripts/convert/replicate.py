"""Replicate dataset multiple times to get enough data
for the document reader preprocess script"""

import json

file = open('/home/zrx/projects/MbaQA/data/corpus/SQuADlike/mba_demo.json', encoding='utf-8')
dataset = json.load(file)
new_dataset = dataset
for i in range(5):
    new_dataset['data'].extend(dataset['data'])
print(len(new_dataset['data']))
json.dump(new_dataset,
          open('/home/zrx/projects/MbaQA/data/corpus/SQuADlike/mba_demo_200-dev.json', 'w'),
          ensure_ascii=False)