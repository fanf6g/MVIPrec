import logging
import random

import numpy as np
import pymongo
from sklearn.feature_extraction.text import CountVectorizer

from KNN.knn_imdb import KNN_imdb

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')

ID = '_id'

ACTORS = 'actors'
DIRECTOR = 'director'
TITLE = 'title'

GENRES = 'genres'
PRODUCER = 'producer'

LBL = 'lbl'

client = pymongo.MongoClient('localhost', 27017)
# db = client['movies']
queryfilter = {TITLE: {'$exists': 'true'},
               DIRECTOR: {'$exists': 'true'},
               ACTORS: {'$exists': 'true'},
               GENRES: {'$exists': 'true'},
               PRODUCER: {'$exists': 'true'},
               }
topk = {PRODUCER: 1050, GENRES: 27}


class KNN_imdb_dup(KNN_imdb):
    def __init__(self, db, cv, attr, queryfilter):
        super().__init__(db, cv, attr, queryfilter)

    def sampling(self, n_samples=50000, dup_ratio=0.5):
        moviedb = self.db.moviedb
        sampledb = self.db.get_collection('sampledb')
        # sampledb0 = self.db.get_collection('sampledb0')

        cur = moviedb.find(queryfilter)
        rec_list = [rec for rec in cur]
        univ_list = list(filter(lambda x: x[self.attr].count(' ') == 0, rec_list))

        attr_list = [x[self.attr] for x in univ_list]

        attrs, counts = np.unique(attr_list, return_counts=True)
        idx = np.argsort(counts)[::-1]

        K = topk.get(self.attr)

        top_K_attr = set(attrs[idx[0:K]])
        print(top_K_attr)
        print(len(top_K_attr))

        freq_recs = [rec for rec in univ_list if rec[self.attr] in top_K_attr][0:n_samples]

        n_sample = int(n_samples * dup_ratio)
        n_dup = [len(d) for d in np.array_split(range(n_sample), 5)]

        dups = []
        random.seed(17)
        for n in n_dup:
            dups.extend(random.sample(freq_recs, n))

        dups.extend(freq_recs)

        random.seed(17)
        random.shuffle(dups)

        for rec in dups:
            rec.pop(ID, '')
            sampledb.save(rec.copy())

        print(len(dups))


if __name__ == "__main__":
    client = pymongo.MongoClient('localhost', 27017)
    db = client['movies']
    cv12 = CountVectorizer(dtype='int16', stop_words='english')
    knn = KNN_imdb_dup(db, cv12, PRODUCER, queryfilter)
    knn.clean()
    knn.sampling()
    knn.shuffle_label()
    knn.split(train_count=60000, valid_count=7500, test_count=7500)

    knn.knn_pred(1)
    knn.knn_verify()

    pass
