import logging
import random

import numpy as np
import pymongo
from sklearn.feature_extraction.text import CountVectorizer

from model import Model

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')


# client = pymongo.MongoClient('localhost', 27017)
# queryfilter = {TITLE: {'$exists': 'true'},
#                JOURNAL: {'$exists': 'true'},
#                # AUHTOR: {'$exists': 'true'},
#                YEAR: {'$exists': 'true'},
#                }




class KNN(Model):
    def __init__(self, db, cv, attr, queryfilter):
        super().__init__(db, cv)
        self.attr = attr
        self.queryfilter = queryfilter

    def clean(self):
        '''
        清理数据表
        :return:
        '''
        names = ['test', 'train', 'valid', 'sampledb', 'dblptmp']
        for name in names:
            logging.info('delete %s' % (name))
            self.db.get_collection(name).drop()

    def sampling(self, n_samples=50000):
        dblp = self.db.dblp
        sampledb = self.db.sampledb

        cur = dblp.find(self.queryfilter).limit(n_samples)
        for rec in cur:
            authorList = rec.get(AUHTOR)
            author = ' '.join(['_'.join(author.split(' ')) for author in authorList])
            rec[AUHTOR] = author
            sampledb.save(rec)

    def shuffle_label(self, seed=17):
        dblp = self.db.sampledb
        tmp = self.db.dblptmp

        journals = dblp.distinct(self.attr)
        journals.sort()

        jkv = enumerate(journals)

        records = []

        m = {}
        # y = {}
        for (i, v) in jkv:
            m[v] = i

        cur = dblp.find(self.queryfilter)
        for rec in cur:
            del rec['_id']
            rec['lbl'] = m[rec[self.attr]]
            records.append(rec)

        random.seed(seed)
        random.shuffle(records)
        for rec in records:
            tmp.insert(rec)

    def split(self, train_count=40000, valid_count=5000, test_count=5000, dup=1):
        dblptmp = self.db.dblptmp
        train = self.db.train
        valid = self.db.valid
        test = self.db.test

        docs = []
        cur = dblptmp.find()
        for rec in cur:
            docs.append(rec)

        train_slice = slice(0, train_count)
        valid_slice = slice(train_count, train_count + valid_count)
        test_slice = slice(train_count + valid_count, train_count + valid_count + test_count)

        train_docs = docs[train_slice]
        valid_docs = docs[valid_slice]
        test_docs = docs[test_slice]

        for rec in train_docs:
            rec.pop('_id', '')
            for i in range(dup):
                train.save(rec.copy())

        for rec in valid_docs:
            rec.pop('_id', '')
            for i in range(dup):
                valid.save(rec.copy())

        for rec in test_docs:
            rec.pop('_id', '')
            for i in range(dup):
                test.save(rec.copy())

    def _extract(self, coll, attr):
        cur = coll.find()
        res = [rec[attr] for rec in cur]
        return res

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

    def knn_verify(self):
        '''
        验证 KNN 方案的准确率.
        :return:
        '''
        knn_labels = self.train_label[self.top_k_idx]
        mlh_labels = []
        for knn_label in knn_labels:
            label_counts = np.unique(knn_label, return_counts=True)
            di = np.argmax(label_counts[1])
            mlh_label = label_counts[0][di]
            mlh_labels.append(mlh_label)

        knn_pred = np.array(mlh_labels)
        res = np.sum(knn_pred == self.test_label)
        print(res)
        return res


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
    knn = KNN(db1, cv12, JOURNAL, queryfilter)
    knn.clean()
    knn.sampling()
    knn.shuffle_label()
    knn.split(train_count=40000, valid_count=5000, test_count=5000)

    # knn.knn_pred(1)
    # knn.knn_verify()

    pass
