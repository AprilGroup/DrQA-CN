"""把人工构建的问答语料转为SQuAD格式(如下)，供Document Reader训练使用
file.json
├── "data"
│   └── [i]
│       ├── "paragraphs"
│       │   └── [j]
│       │       ├── "context": "paragraph text"
│       │       └── "qas"
│       │           └── [k]
│       │               ├── "answers"
│       │               │   └── [l]
│       │               │       ├── "answer_start": N
│       │               │       └── "text": "answer"
│       │               ├── "id": "<uuid>"
│       │               └── "question": "paragraph question?"
│       └── "title": "document id"
└── "version": 1.1
"""

import uuid
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default=None)
parser.add_argument('--output', type=str, default=None)
parser.add_argument('--encoding', type=str, default='utf8')
args = parser.parse_args()

# input = args.input or '../../data/corpus/handwritten/mba_dataset.txt'
# output = args.output or '../../data/corpus/SQuADlike/mba_demo.json'
input = '../../data/corpus/handwritten/mba_def.txt'
output = '../../data/datasets/mba_def.json'

# Read dataset
dataset = {'data': [], 'version': 0.1}
with open(input, encoding=args.encoding) as f:
    doc = {'title': '', 'paragraphs': []}
    context = ''
    qa = {}
    paragraph = {}
    qas = []
    num_q = 0
    for idx, line in enumerate(f):
        # 跳过说明和空行
        if len(line) < 3:
            continue
        # 文章结束标志
        if line.startswith('#####'):
            # 添加文章的最后一个段落
            if 'context' in paragraph:
                doc['paragraphs'].append(paragraph)
            # 添加文章
            # if doc['paragraphs'][0]['qas'][0]['answers'][0]['answer_start'] != -1:
            dataset['data'].append(doc)
            # 清零
            doc = {'title': '', 'paragraphs': []}
            context = ''
            qa = {'question': '', 'answers': []}
            paragraph = {}
        # 文章标题
        elif line.startswith('title'):
            doc['title'] = line.split(':', 1)[1].replace('\n', '')
        # 文章段落
        elif line.startswith('paragraph'):
            if 'context' in paragraph:
                doc['paragraphs'].append(paragraph)
            context = line.split(':', 1)[1].replace('\n', '')
            paragraph = {'context': context, 'qas': []}
        # 问题
        elif line.startswith('q:'):
            num_q += 1
            qa = {'id': str(uuid.uuid4()).replace('-', '')[:24],
                  'question': line.split(':', 1)[1].replace('\n', ''),
                  'answers': []}
        # 答案
        elif line.startswith('a:'):
            # 答案内容
            answer_text = line.split(':', 1)[1].replace('\n', '')
            # 答案起始位置
            answer_start = context.find(answer_text)
            # 添加答案 * 3(SQuAD format)
            for i in range(3):
                qa['answers'].append({'text': answer_text, 'answer_start': answer_start})
            # 添加问答对
            # 如果没有copy，由于dict变量默认存的是引用，再次读到新的问答对时，会导致qas中联动，最后结果就是，
            # qas中存的全都是最后一个问答对
            paragraph['qas'].append(qa.copy())

with open(output, 'w', encoding=args.encoding) as f:
    f.write(json.dumps(dataset))




