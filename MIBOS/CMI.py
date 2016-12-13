import logging

import numpy as np
import pymongo
from pandas.core.frame import DataFrame
from sklearn.cluster import KMeans
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




class CMI(Model):
    def __init__(self, db, cv, attr, queryfilter):
        super().__init__(db, cv)
        self.attr = attr
        self.queryfilter = queryfilter

    def _extract(self, coll, attr):
        cur = coll.find()
        res = [rec[attr] for rec in cur]
        return res

    def k_CMI(self, K=8):
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
        # at.extend(test_at)
        self.cv.fit(at)
        train_data = self.cv.transform(at)
        test_data = self.cv.transform(test_at)

        kmeans = KMeans(n_clusters=K, random_state=0).fit(train_data)

        test_pred = kmeans.predict(test_data)
        print(test_pred.shape)

        lbl = []
        lbl.extend(train_label)
        lbl.extend(valid_label)
        # lbl.extend(test_label)

        print(kmeans.labels_.shape)

        d = {'cluster': kmeans.labels_, 'lbl': lbl}
        pd = DataFrame(d)

        grouped = pd.groupby('cluster')
        modes = []
        for key, group in grouped:
            v = group.mode()['lbl']
            print(key, v)
            modes.append(np.sum(v.values))

        modes = np.array(modes)
        print(modes)

        lbls = modes[test_pred]
        print(np.unique(lbls == np.array(test_label), return_counts=True))

        # print(pd.groupby(['cluster'])['lbl'])
        #
        #
        # at_train_valid = []
        # at_train_valid.extend(train_at)
        # at_train_valid.extend(valid_at)
        #
        # train_data = self.cv.transform(at_train_valid)
        # test_data = self.cv.transform(test_at)
        #
        # m_test = test_data.sum(axis=1)
        # m_train = train_data.sum(axis=1)
        # m_intersect = (test_data.dot(train_data.T)).toarray()
        # m_union = np.array(m_test + m_train.transpose() - m_intersect + 1.0e-4)
        #
        # logging.info("computing top_k")
        # top_k_idx = []
        # count = 0
        # '''cnt_matrix ./ cnt2_matrix = sim(test_data, train_data)'''
        # for num, denorm in zip(m_intersect, m_union):
        #     count += 1
        #     sim = num / denorm
        #     di = np.argsort(-sim)[0:K]
        #     top_k_idx.append(di)
        #     if (count % 100 == 0):
        #         logging.info(str(count))
        #
        # logging.info("computing top_k")
        # # top_k_idx = np.argsort(-M_sim, axis=1)[:, 0:K]
        #
        # self.top_k_idx = np.array(top_k_idx)
        # self.train_label = np.array(train_label)
        # self.test_label = np.array(test_label)


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
    knn = CMI(db1, cv12, JOURNAL, queryfilter)

    knn.k_CMI(100)

    pass
