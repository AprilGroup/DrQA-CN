#!/usr/bin/env python3

"""Documents, in a sqlite database."""

import sqlite3
from mbaqa.retriever import utils
from mbaqa.retriever import DEFAULTS


class DocDB(object):
    """Sqlite backed document storage.

    Implements get_doc_text(doc_id).
    """

    def __init__(self, db_path=None):
        self.path = db_path or DEFAULTS['db_path']
        self.connection = sqlite3.connect(self.path, check_same_thread=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def path(self):
        """Return the path to the file that backs this database."""
        return self.path

    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def get_doc_ids(self):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

    def get_doc_text(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT text FROM documents WHERE id = ?",
            (utils.normalize(doc_id),)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]


if __name__ == '__main__':
    doc_db = DocDB(db_path='../../data/db/mba.db')
    keys = doc_db.get_doc_ids()
    # output titles
    with open('../../data/mba/terms/mba_terms.txt', 'w', encoding='utf8') as f:
        for key in keys:
            f.write(key)
            f.write('\n')
    # print(doc_db.get_doc_text('马尔科夫链'))
    # print(len(doc_db.get_doc_text('"T+0"交易.mba.wiki')))