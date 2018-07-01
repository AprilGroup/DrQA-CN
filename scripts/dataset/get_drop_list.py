"""A script to drop near duplicates in full doc set. Mba definition docs are of high
priority because document retriever test data is based on them."""

import json
import logging

from tqdm import tqdm


logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


# get doc titles in mba definition dataset
logger.info('Retrieving mba definition dataset titles.')
mba_def_titles = []
with open('../../data/mba/drqa_format/mba_def.json', 'r', encoding='utf8') as file:
    for line in file:
        doc = json.loads(line)
        mba_def_titles.append(doc['id'])


# get doc titles in original mba dataset
logger.info('Retrieving original mba dataset titles.')
mba_origin_titles = []
with tqdm(total=134116) as pbar:
    with open('../../data/mba/drqa_format/mba_origin.json', encoding='utf8') as file:
        for line in file:
            doc = json.loads(line)
            mba_origin_titles.append(doc['id'])
            pbar.update()


def get_drop_list(tmp_group, mba_def_titles, mba_origin_titles):
    # all docs in this group are from definition dataset, keep the first doc only
    if len(set(tmp_group) & set(mba_origin_titles)) == 0:
        # logger.info('all from def')
        return tmp_group[1:], 0
    # all docs in this group are from origin dataset, keep the first doc only
    elif len(set(tmp_group) & set(mba_def_titles)) == 0:
        # logger.info('all from origin')
        return tmp_group[1:], 1
    # if there are common docs between tmp_group and mba_def_titles, keep the first
    # common title only
    else:
        # logger.info('from both')
        first_common_title_in_def = list(set(tmp_group) & set(mba_def_titles))[0]
        # logger.info('first common title: {}'.format(first_common_title_in_def))
        return [title for title in tmp_group if title != first_common_title_in_def], 2


# get doc titles to be dropped
drop_list = []
done_num_groups = 0
type_count = [0, 0, 0]  # [all from def num, all from origin num, from both num]
with tqdm(total=114613) as pbar:  # total lines minus num of '#####'s
    with open('../../data/output/duplicates-k=3-num=134974.txt') as inf:
        tmp_group = []
        for line in inf:
            if line.startswith('#####'):
                tmp_dropped, type = get_drop_list(tmp_group, mba_def_titles, mba_origin_titles)
                type_count[type] += 1
                if (len(tmp_group) - 1) != len(tmp_dropped):
                    raise RuntimeError('get dropped list failed.')
                drop_list.append(tmp_dropped)
                tmp_group = []
                done_num_groups += 1
                # if done_num_groups > 10:
                #     break
                continue
            title = line.replace('\n', '').strip()
            tmp_group.append(title)
            pbar.update()

# output duplicates report, and titles to be dropped.
with open('../../data/output/droplist-k=3-num=134974.txt', 'w', encoding='utf8') as outf:
    outf.write('duplicates stat:\nall from def: {}\nall from origin: {}\nfrom both: {}\n'.
               format(type_count[0], type_count[1], type_count[2]))
    for title in sum(drop_list, []):
        outf.write(title)
        outf.write('\n')






