import logging
import random

import numpy as np
import pymongo
from sklearn.feature_extraction.text import CountVectorizer

from KNN.knn_restaurant import KNN_restaurant

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')

ID = '_id'

NAME = 'name'
CITY = 'city'
TYPE = 'type'

LBL = 'lbl'

client = pymongo.MongoClient('localhost', 27017)

queryfilter = {TYPE: {'$exists': 'true'},
               CITY: {'$exists': 'true'},
               NAME: {'$exists': 'true'},
               LBL: {'$exists': 'true'}
               }


class KNN_restaurant_dup(KNN_restaurant):
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

    def sampling(self, n_samples=50000, dup_ratio=0.5):
        dblp = self.db.restaurant
        sampledb = self.db.sampledb

        records = [rec for rec in dblp.find(self.queryfilter).limit(n_samples)]

        n_sample = int(n_samples * dup_ratio)
        n_dup = [len(d) for d in np.array_split(range(n_sample), 5)]

        dups = []
        random.seed(17)
        for n in n_dup:
            dups.extend(random.sample(records, n))

        dups.extend(records)

        random.seed(17)
        random.shuffle(dups)

        for rec in dups:
            rec.pop(ID, '')
            sampledb.save(rec.copy())

        print(len(dups))

    def knn_pred(self, K=1):
        train = self.db.train
        test = self.db.test
        valid = self.db.valid

        train_type = self._extract(train, TYPE)
        train_city = self._extract(train, CITY)
        train_name = self._extract(train, NAME)
        train_at = [str(a) + ' ' + str(t) + ' ' + str(d) for (a, t, d) in
                    zip(train_name, train_type, train_city)]
        train_label = self._extract(train, LBL)

        valid_type = self._extract(valid, TYPE)
        valid_city = self._extract(valid, CITY)
        valid_name = self._extract(valid, NAME)
        valid_at = [str(a) + ' ' + str(t) + ' ' + str(d) for (a, t, d) in
                    zip(valid_name, valid_type, valid_city)]
        valid_label = self._extract(valid, LBL)

        test_type = self._extract(test, TYPE)
        test_city = self._extract(test, CITY)
        test_name = self._extract(test, NAME)
        test_at = [str(a) + ' ' + str(t) + ' ' + str(d) for (a, t, d) in zip(test_name, test_type, test_city)]
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
    db1 = client['restaurant']
    cv12 = CountVectorizer(dtype='int16', stop_words='english')
    knn = KNN_restaurant_dup(db1, cv12, LBL, queryfilter)
    knn.clean()
    knn.sampling(n_samples=800)
    knn.shuffle_label()
    knn.split(train_count=960, valid_count=120, test_count=120)

    knn.knn_pred(1)
    knn.knn_verify()

    pass
