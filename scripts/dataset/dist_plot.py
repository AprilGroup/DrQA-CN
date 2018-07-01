"""A script to plot the distribution of the length of paragraph, question and answer
of mba vs squad dataset."""
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys
sys.path.append('/home/zrx/projects/MbaQA/')
from mbaqa.tokenizers.ltp_tokenizer import LtpTokenizer

# read in datasets
squad_train_path = '../../data/datasets/train-v1.1.json'
mba_def_path = '../../data/datasets/mba_def.json'
squad_train = json.load(open(squad_train_path))
mba_def = json.load(open(mba_def_path))

# tokenizer
tokenizer = LtpTokenizer(annotators=set())


def get_lens(dataset):
    question_lens, answer_lens, para_lens = [], [], []
    for idx, doc in enumerate(dataset):
        for para in doc['paragraphs']:
            para_len = len(tokenizer.tokenize(para['context']))
            para_lens.append(para_len)
            if 'qas' in para:
                for qa in para['qas']:
                    q_len = len(tokenizer.tokenize(qa['question']))
                    question_lens.append(q_len)
                    for answer in qa['answers']:
                        a_len = len(tokenizer.tokenize(answer['text']))
                        answer_lens.append(a_len)

        print(idx)
        if idx > 100:
            break
    return question_lens, answer_lens, para_lens

def plot_lens(q_lens, a_lens, para_lens):
    plt.figure(figsize=(10, 3))
    plt.subplot(131)
    sns.distplot(q_lens)
    plt.subplot(132)
    sns.distplot(a_lens)
    plt.subplot(133)
    sns.distplot(para_lens)
    plt.show()

#
mba_question_lens, mba_answer_lens, mba_para_lens = get_lens(mba_def['data'])
plot_lens(mba_question_lens, mba_answer_lens, mba_para_lens)
print(1)
# mba_para_lens = []
# mba_question_lens = []
# squad_answer_lens = []
# squad_para_lens = []
# squad_question_lens = []


