import logging
import random

import numpy as np
import pymongo
from sklearn.feature_extraction.text import CountVectorizer

from KNN.knn import KNN

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
# queryfilter = {'title': {'$exists': 'true'},
#                'producer': {'$exists': 'true'},
#                'actors': {'$exists': 'true'},
#                'genres': {'$exists': 'true'},
#                'director': {'$exists': 'true'}
#                }

topk = {PRODUCER: 1050, GENRES: 27}


class KNN_imdb(KNN):
    def __init__(self, db, cv, attr, queryfilter):
        super().__init__(db, cv, attr, queryfilter)

    def clean(self):
        '''
        清理数据表
        :return:
        '''
        names = ['test', 'train', 'valid', 'sampledb', 'dblptmp', 'sampledb0']
        for name in names:
            logging.info('delete %s' % (name))
            self.db.get_collection(name).drop()

    def sampling(self, n_samples=50000):
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

        freq_recs = [rec for rec in univ_list if rec[self.attr] in top_K_attr]

        random.seed(17)
        random.shuffle(freq_recs)

        for rec in freq_recs[0:n_samples]:
            rec.pop(ID)
            sampledb.save(rec.copy())

    def knn_pred(self, K=200):
        train = self.db.train
        test = self.db.test
        valid = self.db.valid

        train_title = self._extract(train, TITLE)
        train_director = self._extract(train, DIRECTOR)
        train_author = self._extract(train, ACTORS)
        train_at = [a + ' ' + t + ' ' + d for (a, t, d) in zip(train_author, train_title, train_director)]
        train_label = self._extract(train, LBL)

        valid_title = self._extract(valid, TITLE)
        valid_director = self._extract(valid, DIRECTOR)
        valid_author = self._extract(valid, ACTORS)
        valid_at = [a + ' ' + t + ' ' + d for (a, t, d) in zip(valid_author, valid_title, valid_director)]
        valid_label = self._extract(valid, LBL)

        test_title = self._extract(test, TITLE)
        test_director = self._extract(test, DIRECTOR)
        test_author = self._extract(test, ACTORS)
        test_at = [a + ' ' + t + ' ' + d for (a, t, d) in zip(test_author, test_title, test_director)]
        test_label = self._extract(test, LBL)

        at = []
        at.extend(train_at)
        at.extend(valid_at)
        at.extend(test_at)
        self.cv.fit(at)
        train_data = self.cv.transform(train_at)
        test_data = self.cv.transform(test_at)

        m_test = test_data.sum(axis=1)
        m_train = train_data.sum(axis=1)
        m_intersect = (test_data.dot(train_data.T)).toarray()
        m_union = np.array(m_test + m_train.transpose() - m_intersect + 1.0e-4)

        logging.info("computing top_k")
        top_k_idx = []
        count = 0
        '''cnt_matrix ./ cnt2_matrix = sim(test_data, train_data)'''
        for num, denorm in zip(m_intersect, m_union):
            count += 1
            sim = num / denorm
            di = np.argsort(-sim)[0:K]
            top_k_idx.append(di)
            if (count % 100 == 0):
                logging.info(str(count))

        logging.info("computing top_k")

        self.top_k_idx = np.array(top_k_idx)
        self.train_label = np.array(train_label)
        self.test_label = np.array(test_label)


if __name__ == "__main__":
    client = pymongo.MongoClient('localhost', 27017)
    db1 = client['movies']
    cv12 = CountVectorizer(dtype='int16', stop_words='english')
    knn = KNN_imdb(db1, cv12, PRODUCER, queryfilter)
    knn.clean()
    knn.sampling()
    knn.shuffle_label()
    knn.split(train_count=40000, valid_count=5000, test_count=5000)

    knn.knn_pred(50)
    knn.knn_verify()

    pass
