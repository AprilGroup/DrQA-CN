"""A Script to compute document retriever nagrams statistics."""

from tqdm import tqdm
import numpy as np


ngrams_path = '../../data/retriever/model/hash2gram.txt'

hash_counts = {}
# compute number of words mapped to a hashcode for each hashcode
with tqdm(total=5234649) as pbar:
    with open(ngrams_path, encoding='utf8') as file:
        for i, line in enumerate(file):
            # get hashcode
            hashcode = line.split('-')[0]
            # number of words mapped to the hash code
            num_words = len(line.split('-')[1].split('||'))
            # update info dict
            hash_counts[hashcode] = num_words

            pbar.update()


print('{} hashed in total\n'.format(len(hash_counts)))
print(np.bincount(list(hash_counts.values())))