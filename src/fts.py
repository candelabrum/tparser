import os.path
import datetime
import os
import csv
import time
import pickle
import datetime
import torch
import os.path
import pandas as pd
from whoosh.qparser import QueryParser
from whoosh import scoring
from tqdm import tqdm
from whoosh.fields import Schema, STORED, TEXT, DATETIME, NUMERIC
from whoosh.index import create_in, open_dir
from collections import defaultdict
from dateutil import relativedelta
from whoosh.query import DateRange


pd.set_option("mode.copy_on_write", True)


class Counter:
    def __init__(self, counter_path=None):
        self.counter_path = counter_path
        self.true_id_str_to_int = dict()
        if counter_path is not None:
            if not os.path.exists(counter_path):
                self.save_dict()

            with open(counter_path, 'rb') as fd:
                self.true_id_str_to_int = pickle.load(fd)

    def save_dict(self):
        with open(self.counter_path, 'wb') as fd:
            pickle.dump(self.true_id_str_to_int, fd)

    def get_index(self, true_id):
        if index := self.true_id_str_to_int.get(true_id, None):
            return index

        idx = len(self.true_id_str_to_int)
        self.true_id_str_to_int[true_id] = idx
        self.save_dict()

        return idx


class FullTextSearch:
    def __init__(self, name_index, counter_path='counter_path.pickle'):
        self.name_index = name_index
        self.schema = Schema(
            channel=TEXT(stored=True),
            message=TEXT(stored=True),
            date=DATETIME(stored=True),
            id=NUMERIC(stored=True),
            true_id=NUMERIC(stored=True, unique=True)
        )
        self.channel2max = defaultdict(int)
        self.fields = ['message', 'date', 'id']
        self.counter = Counter(counter_path)
#        if counter_path is None:
#
#        else:
#            with open(counter_path, 'rb') as fd:
#                self.counter = pickle.load(fd)

#    @cached_property
    @property
    def ix(self):
        from whoosh import index
        if not os.path.exists(self.name_index):
            os.mkdir(self.name_index)

        if not index.exists_in(self.name_index):
            ix = create_in(self.name_index, self.schema)
        else:
            ix = open_dir(self.name_index)

        return ix

    def search(self, query_text):
        qp = QueryParser("message", schema=self.schema)
        q = qp.parse(query_text)
        docs = []
        with self.ix.searcher(weighting=scoring.TF_IDF()) as s:
            results = s.search(q, limit=100000)
            for result in results:
                idx_doc = result['id']  # Get the document ID
                # Retrieve the full document using its ID
                doc = s.document(id=idx_doc)  # Get the full document by ID
                # print(doc)

                # Now you can access all fields of the document
                channel = doc['channel']
                content = doc['message']
                dt = doc['date']
                docs.append((channel, content, dt))

        return docs

    def get_all_news_by_date(self, date):

        query = DateRange("date", date, date +
                          relativedelta.relativedelta(days=1))
        with self.ix.searcher() as s:
            results = s.search(query, limit=None)
#            print(results)
            docs = []
            for result in results:
                idx_doc = result['id']  # Get the document ID
                # Retrieve the full document using its ID
                doc = s.document(id=idx_doc)  # Get the full document by ID
    #            print(doc)

                # Now you can access all fields of the document
                channel = doc['channel']
                content = doc['message']
                dt = doc['date']
                docs.append((channel, content, dt))

        return docs

    def update_docs(self, docs):
        for channel, messages in docs.items():
            for doc in tqdm(messages):
                idx = doc['id']
                writer = self.ix.writer()
                self.channel2max[channel] = max(self.channel2max[channel], idx)
                my_doc = dict()
                for field in self.fields:
                    my_doc[field] = doc.get(field, None)
                my_doc['channel'] = channel
                my_doc['true_id'] = self.counter.get_index(
                    channel + '_^_' + str(doc['id']))
                writer.update_document(**my_doc)
                writer.commit()
