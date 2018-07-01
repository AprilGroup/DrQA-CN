"""dataset related utils"""

import unicodedata
import json


def read_lines(filename):
    """read lines from txt file, remove head and tail blank."""
    lines = []
    with open(filename, encoding='utf8') as file:
        for line in file:
            lines.append(line.replace('\n', '').strip())
    return lines


def split_doc(content, group_length=300):
    """Given a doc, split it into chunks (by paragraph)."""
    curr = []
    curr_len = 0
    for split in content.split('|'):
        split = split.strip()
        if len(split) == 0:
            continue
        # Maybe group paragraphs together until we hit a length limit
        if len(curr) > 0 and curr_len + len(split) > group_length:
            yield ' '.join(curr)
            curr = []
            curr_len = 0
        curr.append(split)
        curr_len += len(split)
    if len(curr) > 0:
        yield ' '.join(curr)


def clean_title(text):
    """clean title in original mba docs"""
    return text.replace('.mba.wiki', '').replace('.iwencai.wiki', '(wencai)')


def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


def read_drqa_format_dataset_as_dict(filepath):
    dataset = {}
    with open(filepath, encoding='utf8') as file:
        for line in file:
            doc = json.loads(line)
            dataset[doc['id']] = doc['text']
    return dataset


def read_json(filepath):
    with open(filepath, encoding='utf8') as file:
        return json.load(file)


def save_json(dataset, save_path):
    with open(save_path, 'w', encoding='utf8') as file:
        json.dump(dataset, file)