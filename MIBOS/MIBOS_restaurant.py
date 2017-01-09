import logging

import numpy as np
import pymongo
from scipy.sparse.csr import csr_matrix
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




class MIBOS(Model):
    def __init__(self, db, cv, attr, queryfilter):
        super().__init__(db, cv)
        self.attr = attr
        self.queryfilter = queryfilter

    def _extract(self, coll, attr):
        cur = coll.find()
        res = [str(rec.get(attr, '')) for rec in cur]
        return res

    def mibos(self, K=8):

        def match(s1, s2):
            return 1 if s1 == s2 else 0

        v_match = np.vectorize(match)

        def _Pk(train_attrs, test_attrs):
            attrs = []
            attrs.extend(train_attrs)
            attrs.extend(test_attrs)

            self.cv.fit(attrs)

            train_features = self.cv.transform(train_attrs)
            test_features = self.cv.transform(test_attrs)

            row_ind = []
            col_ind = []

            train_sets = [set(row.indices) for row in train_features]
            test_sets = [set(row.indices) for row in test_features]

            for (i, s_i) in enumerate(test_sets):
                # si = set(fi.indices)
                j_ind = v_match(s_i, train_sets)
                j_ind[i] = 0
                ind = np.where(j_ind == 1)
                j_len = len(ind[0])
                if (j_len > 0):
                    row_ind.extend([i] * j_len)
                    col_ind.extend(ind[0])
                    # print(i, train_attrs[i], j_len)

            pk = csr_matrix((np.ones(len(row_ind), dtype='int16'), (row_ind, col_ind)),
                            shape=(len(test_attrs), len(train_attrs)))

            return pk

        train = self.db.train
        valid = self.db.valid
        test = self.db.test

        train_city = self._extract(train, ADDR)
        train_type = self._extract(train, TYPE)
        train_name = self._extract(train, NAME)
        train_lbl = self._extract(train, LBL)

        valid_city = self._extract(valid, ADDR)
        valid_type = self._extract(valid, TYPE)
        valid_name = self._extract(valid, NAME)
        valid_label = self._extract(valid, LBL)

        test_city = self._extract(test, ADDR)
        test_type = self._extract(test, TYPE)
        test_name = self._extract(test, NAME)
        test_label = self._extract(test, LBL)

        cities = []
        cities.extend(train_city)
        cities.extend(valid_city)
        # titles.extend(test_title)

        P_cities = _Pk(cities, test_city)

        names = []
        names.extend(train_name)
        names.extend(valid_name)

        # types = []
        # types.extend(train_type)
        # types.extend(valid_type)
        # authors.extend(test_author)

        lbl = []
        lbl.extend(train_lbl)
        lbl.extend(valid_label)
        # lbl.extend(test_label)
        a_lbl = np.array(lbl)
        a_test_lbl = np.array(test_label)

        P_names = _Pk(names, test_name)
        # P_types = _Pk(types, test_type)

        P_mul = P_names.multiply(P_cities)
        P_add = P_cities + P_names
        P_ = P_mul.multiply(P_add)
        P_.eliminate_zeros()

        print(P_.shape)
        print(type(P_))
        print(P_)

        n_match = 0
        sn = 0
        for (i, ns_i) in enumerate(P_):
            ns = ns_i.indices
            if (ns.size > 0):
                sn = sn + 1
                can = np.unique(a_lbl[ns])
                print(i, a_test_lbl[i], can)
                if (can.size == 1 and a_test_lbl[i] == can[0]):
                    n_match = n_match + 1
            pass

        print(n_match, sn)


if __name__ == "__main__":
    ID = '_id'

    # NAME = 'name'
    # CITY = 'city'
    # TYPE = 'type'

    NAME = 'name'
    CITY = 'city'
    ADDR = 'addr'
    TYPE = 'type'

    LBL = 'lbl'
    client = pymongo.MongoClient('localhost', 27017)
    queryfilter = {CITY: {'$exists': 'true'},
                   NAME: {'$exists': 'true'},
                   # AUHTOR: {'$exists': 'true'},
                   }
    db1 = client['restaurant']

    cv12 = CountVectorizer(dtype='int16', stop_words='english')
    knn = MIBOS(db1, cv12, LBL, queryfilter)

    knn.mibos(100)

    MAS = []

    pass
