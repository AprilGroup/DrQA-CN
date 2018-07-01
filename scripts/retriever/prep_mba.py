#!/usr/bin/env python3

"""preprocess mba docs"""


def preprocess(article):
    for k, v in article.items():
        if k == 'title':
            # remove '.mba.wiki' in mba title
            article[k] = article[k].replace('.mba.wiki', '')
        if k == 'text':
            # remove '|' in mba content
            article[k] = v.replace('|', '')

    # Return doc with `id` set to `title`
    return {'id': article['title'], 'text': article['text']}

