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


class KNN_dup(KNN):
    def __init__(self, db, cv, attr, queryfilter):
        super().__init__(db, cv, attr, queryfilter)

    def sampling(self, n_samples=50000, dup_ratio=0.5):
        dblp = self.db.dblp
        sampledb = self.db.sampledb

        cur = dblp.find(self.queryfilter).limit(n_samples)

        tuples = []

        for rec in cur:
            authorList = rec.get(AUHTOR)
            author = ' '.join(['_'.join(author.split(' ')) for author in authorList])
            rec[AUHTOR] = author
            rec.pop('_id')
            tuples.append(rec)

        n_sample = int(n_samples * dup_ratio)
        n_dup = [len(d) for d in np.array_split(range(n_sample), 5)]

        dup = []
        random.seed(17)
        # random.shuffle()
        for n in n_dup:
            dup.extend(random.sample(tuples, n))

        dup.extend(tuples)

        for rec in dup:
            sampledb.save(rec.copy())

    def knn_pred(self, K=200):
        train = self.db.train
        valid = self.db.valid
        test = self.db.test

        train_title = self._extract(train, TITLE)
        train_author = self._extract(train, AUHTOR)
        train_at = [a + ' ' + t for (a, t) in zip(train_author, train_title)]
        train_label = self._extract(train, LBL)

        valid_title = self._extract(valid, TITLE)
        valid_author = self._extract(valid, AUHTOR)
        valid_at = [a + ' ' + t for (a, t) in zip(valid_author, valid_title)]
        valid_label = self._extract(valid, LBL)

        test_title = self._extract(test, TITLE)
        test_author = self._extract(test, AUHTOR)
        test_at = [a + ' ' + t for (a, t) in zip(test_author, test_title)]
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
        # top_k_idx = np.argsort(-M_sim, axis=1)[:, 0:K]

        self.top_k_idx = np.array(top_k_idx)
        self.train_label = np.array(train_label)
        self.test_label = np.array(test_label)


if __name__ == "__main__":
    ID = '_id'
    AUHTOR = 'author'
    TITLE = 'title'
    JOURNAL = 'journal'
    YEAR = 'year'
    LBL = 'lbl'
    client = pymongo.MongoClient('localhost', 27017)
    queryfilter = {TITLE: {'$exists': 'true'},
                   JOURNAL: {'$exists': 'true'},
                   # AUHTOR: {'$exists': 'true'},
                   YEAR: {'$exists': 'true'},
                   }
    db1 = client['accuracy']

    cv12 = CountVectorizer(dtype='int16', stop_words='english')
    knn = KNN_dup(db1, cv12, JOURNAL, queryfilter)
    knn.clean()
    knn.sampling()
    knn.shuffle_label()
    knn.split(train_count=60000, valid_count=7500, test_count=7500)

    knn.knn_pred(50)
    knn.knn_verify()

    pass
