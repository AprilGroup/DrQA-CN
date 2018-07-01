"""A flask web wrapper for document retriever interactive mode."""

import json
import re
import traceback
import random
from flask import Flask, jsonify
from flask_cors import CORS

import sys
sys.path.append('/home/zrx/projects/MbaQA')
from mbaqa import retriever
from mbaqa.tokenizers import LtpTokenizer
import scripts.dataset.utils as utils

ranker = retriever.get_class('tfidf')(tfidf_path='../../data/retriever/model/mba-tfidf-ngram=2-hash=16777216-tokenizer=ltp-numdocs=78259.npz')
doc_db = retriever.doc_db.DocDB(db_path='../../data/db/mba.db')
tokenizer = LtpTokenizer()

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    """Show Some titles of doc set."""
    titles = doc_db.get_doc_ids()
    titles = random.sample(titles, 10)
    return json.dumps(titles)


@app.route('/drqa/retriever/<query>')
def task_drqa(query, k=5):
    """Find k top-scored docs for query."""
    try:
        results = {}
        doc_names, doc_scores = ranker.closest_docs_by_content_and_title(query, k=k)
        for i in range(len(doc_names)):
            results[i] = {"Rank": i + 1,
                          "Doc Id": doc_names[i],
                          "Doc Score": '%.5g' % doc_scores[i]}
        return json.dumps(results)
    except Exception as e:
        error_message = {'error': u"ERROR in System"}
        return json.dumps(error_message)


@app.route('/drqa/retriever/doc/<doc_id>')
def get_doc(doc_id):
    """Get doc text for doc title(doc_id)."""
    try:
        doc_content = doc_db.get_doc_text(doc_id)
        return json.dumps({'title': doc_id, 'content': doc_content})
    except Exception as e:
        error_message = {'error': u"ERROR in System"}
        return json.dumps(error_message)


@app.route('/drqa/retriever/docweight/<doc_id>')
def get_doc_ngram_weights(doc_id):
    """Get ngrams and weights for a doc given title(doc_id)."""
    try:
        doc_weights = ranker.get_weights_for_doc(doc_id)
        return json.dumps(doc_weights)
    except Exception as e:
        error_message = {'error': u"ERROR in System"}
        return json.dumps(error_message)


@app.route('/drqa/retriever/queryweight/<query>')
def get_query_ngram_weights(query):
    """Get ngrams and weights for a query."""
    try:
        spvec = ranker.text2spvec(query)
        query_weights = ranker.get_weights_for_spvec(spvec)
        return json.dumps(query_weights)
    except Exception as e:
        error_message = {'error': u"ERROR in System"}
        return json.dumps(error_message)


@app.route('/drqa/retriever/titleweight/<title>')
def get_title_ngram_weights(title):
    """Get ngrams and weights for a title."""
    try:
        title_weights = ranker.get_weights_for_title(title)
        return json.dumps(title_weights)
    except Exception as e:
        traceback.print_exc()
        error_message = {'error': u"ERROR in System"}
        return json.dumps(error_message)


@app.route('/drqa/retriever/containanswer/<answer>', methods=['GET', 'POST'])
def contain_answer(answer, doc_id='国债远期交易', match='string'):
    """Check whether a doc contains answer."""
    # get doc text
    text = doc_db.get_doc_text(doc_id)
    text = utils.normalize(text)
    if match == 'string':
        # Answer is a list of possible strings
        text = tokenizer.tokenize(text).words(uncased=True)
        for single_answer in answer:
            single_answer = utils.normalize(single_answer)
            single_answer = tokenizer.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)
            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i: i + len(single_answer)]:
                    return json.dumps({'contain': True,
                                       'answer_words': single_answer,
                                       'text_words': text})
            # all answers are the same, one loop enough.
            return json.dumps({'contain': True,
                               'answer_words': single_answer,
                               'text_words': text})
    elif match == 'regex':
        # Answer is a regex
        single_answer = utils.normalize(answer[0])
        if regex_match(text, single_answer):
            return json.dumps({'contain': True, 'answer': single_answer, 'text': text})
        else:
            return json.dumps({'contain': False, 'answer': single_answer, 'text': text})
    elif match == 'find':
        single_answer = utils.normalize(answer[0])
        return json.dumps(text.find(single_answer) != -1)


@app.route('/drqa/retriever/closesttitles/<query>')
def closest_titles(query):
    titles, scores = ranker.closest_docs_by_title(query, 5)
    result = {titles[i]: scores[i] for i in range(len(titles))}
    return json.dumps(result)


@app.route('/drqa/ngrams/<query>')
def get_ngrams(query, n=1):
    tokens = tokenizer.tokenize(query)
    ngrams = tokens.ngrams(
        n, uncased=True, filter_fn=retriever.utils.filter_ngram, as_strings=True
    )
    return json.dumps(ngrams)


def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
        )
    except BaseException:
        return False
    return pattern.search(text) is not None


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=16891, threaded=True)
    # print(closest_titles('Rho'))
    # print(get_title_ngram_weights('24国集团'))