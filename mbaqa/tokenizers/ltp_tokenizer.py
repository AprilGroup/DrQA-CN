#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tokenizer that is backed by HIT ltp.

Requires pyltp package and the model data.
"""

import os
import copy
from .tokenizer import Tokens, Tokenizer
from . import DEFAULTS
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import NamedEntityRecognizer

LTP_DATA_DIR = DEFAULTS['ltp_datapath']  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`


class LtpTokenizer(Tokenizer):

    def __init__(self, **kwargs):
        """
        Args:
            annotators: set that can include pos and ner.
            model: ltp model to use (path).
        """
        self.segmentor = Segmentor()  # 初始化分词器实例
        self.recognizer = NamedEntityRecognizer() # 初始化命名实体识别器实例
        self.postagger = Postagger() # 初始化词性标注实例

        self.segmentor.load(cws_model_path)  # 加载分词模型

        self.annotators = copy.deepcopy(kwargs.get('annotators', set()))
        if {'pos'} & self.annotators:
            self.postagger.load(pos_model_path)
        if {'ner'} & self.annotators:
            self.postagger.load(pos_model_path)
            self.recognizer.load(ner_model_path)

    def tokenize(self, text):
        # We don't treat new lines as tokens.
        clean_text = text.replace('\n', ' ')

        words = list(self.segmentor.segment(clean_text))  # 分词
        postags = ['empty'] * len(words)
        netags = ['empty'] * len(words)

        if {'pos'} & self.annotators:  # 交集
            postags = list(self.postagger.postag(words))  # 词性标注
        if {'ner'} & self.annotators:
            postags = list(self.postagger.postag(words))
            netags = list(self.recognizer.recognize(words, postags))  # 命名实体识别

        data = []
        tmp_idx = 0
        for i in range(len(words)):
            # Get whitespace
            start_ws = tmp_idx
            end_ws = tmp_idx + len(words[i])
            tmp_idx += len(words[i])

            data.append((
                words[i],
                text[start_ws: end_ws],
                (start_ws, end_ws),
                postags[i],
                words[i],  # lemma
                netags[i],
            ))

        # Set special option for non-entity tag: '' vs 'O' in spaCy
        return Tokens(data, self.annotators, opts={'non_ent': ''})