#!/usr/bin/env python3

"""Tokenizer that is backed by jieba.

Requires jieba package.
"""

import jieba

from .tokenizer import Tokens, Tokenizer


class JiebaTokenizer(Tokenizer):

    def __init__(self, **kwargs):
        """
        Args:
            annotators: set that can include pos, lemma, and ner.
        """
        self.annotators = set()

    def tokenize(self, text):
        result = jieba.tokenize(text)

        data = []
        for token in result:
            data.append((
                token[0],
                text[token[1]: token[2]],
                (token[1], token[2]),
            ))

        return Tokens(data, self.annotators)
