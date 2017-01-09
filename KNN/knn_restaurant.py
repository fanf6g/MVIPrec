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

NAME = 'name'
CITY = 'city'
ADDR = 'addr'
TYPE = 'type'

LBL = 'lbl'

client = pymongo.MongoClient('localhost', 27017)

queryfilter = {TYPE: {'$exists': 'true'},
               CITY: {'$exists': 'true'},
               NAME: {'$exists': 'true'},
               LBL: {'$exists': 'true'}
               }


class KNN_restaurant(KNN):
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
        dblp = self.db.restaurant
        sampledb = self.db.sampledb

        cur = dblp.find(self.queryfilter).limit(n_samples)
        for rec in cur:
            sampledb.save(rec)

    def shuffle_label(self, seed=1):
        dblp = self.db.sampledb
        tmp = self.db.dblptmp

        records = []
        cities = []

        cur = dblp.find(self.queryfilter)

        for rec in cur:
            del rec['_id']
            cities.append(rec[CITY])
            records.append(rec)

        city2lbl = [(c, i) for (i, c) in enumerate(sorted(set(cities)))]
        city2dict = dict(city2lbl)

        random.seed(seed)
        random.shuffle(records)
        for rec in records:
            rec[LBL] = city2dict[rec[CITY]]
            tmp.insert(rec)

    def knn_pred(self, K=1):
        train = self.db.train
        test = self.db.test
        valid = self.db.valid

        train_type = self._extract(train, TYPE)
        train_addr = self._extract(train, ADDR)
        train_name = self._extract(train, NAME)
        train_at = [str(n) + ' ' + str(t) + ' ' + str(a) for (n, t, a) in
                    zip(train_name, train_type, train_addr)]
        train_label = self._extract(train, LBL)

        valid_type = self._extract(valid, TYPE)
        valid_addr = self._extract(valid, ADDR)
        valid_name = self._extract(valid, NAME)
        valid_at = [str(n) + ' ' + str(t) + ' ' + str(a) for (n, t, a) in
                    zip(valid_name, valid_type, valid_addr)]
        valid_label = self._extract(valid, LBL)

        test_type = self._extract(test, TYPE)
        test_addr = self._extract(test, ADDR)
        test_name = self._extract(test, NAME)
        test_at = [str(n) + ' ' + str(t) + ' ' + str(a) for (n, t, a) in
                   zip(test_name, test_type, test_addr)]
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
    LBL = 'lbl'
    client = pymongo.MongoClient('localhost', 27017)
    db1 = client['restaurant']
    cv12 = CountVectorizer(dtype='int16', stop_words='english')
    knn = KNN_restaurant(db1, cv12, LBL, queryfilter)
    knn.clean()
    knn.sampling(n_samples=800)
    knn.shuffle_label()
    knn.split(train_count=640, valid_count=80, test_count=80)

    knn.knn_pred(1)
    knn.knn_verify()

    pass
