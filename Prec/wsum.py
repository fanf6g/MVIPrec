import itertools
import logging

import numpy as np
import pymongo
from sklearn.feature_extraction.text import CountVectorizer

from model import Model

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')

client = pymongo.MongoClient('localhost', 27017)

TRAIN = 'train'
VALID = 'valid'
TEST = 'test'


class Collective2(Model):
    def __init__(self, db, cv=CountVectorizer(min_df=2, ngram_range=(1, 2), dtype='int16'), alpha=1.0e-8, tau=1,
                 theta=0.5, q=1.0, prec=0.8):
        super().__init__(db, cv)
        self.alpha = alpha
        self.tau = tau
        self.theta = theta
        self.q = q
        self.prec = prec

    def training(self, features, labels):
        '''
        计算 [P(a_j | f_k)]
        :param features:
        :param labels:
        :return:
        '''
        A_feature = features.toarray()

        n_records, n_features = A_feature.shape
        n_class = max(labels) + 1

        self.feature_cls_counts = np.zeros((n_class, n_features), dtype='int16')
        for (row, lbl) in zip(A_feature, labels):
            self.feature_cls_counts[lbl] += row

        self.feature_counts = np.sum(self.feature_cls_counts, axis=0)

        smoothed_fc = self.feature_cls_counts + self.alpha
        smoothed_cc = self.feature_counts + self.alpha * n_class

        self.proba = smoothed_fc / smoothed_cc

        # self.W = np.sum(np.power(self.proba, self.q), axis=0)

    def _update(self, prec, q, tau):
        self.prec = prec
        self.q = q
        self.tau = tau
        self.W = np.sum(np.power(self.proba, self.q), axis=0)

    def validating(self, features, labels):
        self._optimize(features, labels)

    def _softmax1d(self, x, tau=None):
        if tau is None:
            tau = self.tau
        x_max = np.max(x)
        return np.exp((x - x_max) * tau) / np.exp((x - x_max) * tau).sum()

    def _optimize(self, features, labels):

        # m_agg = np.power(m_agg, self.q)
        # with open('dblp.pkl', 'rb') as f:
        #     m_agg, lables = pickle.load(f)
        #     logging.info('dumping {0}'.format(f))

        A_feature = features.toarray()
        # logging.info('the shape of feature matrix is: {0}'.format(A_feature.shape))

        M_prob = np.transpose(np.power(self.proba * self.W, 1))
        m_agg = A_feature.dot(M_prob)

        # print(m_agg.shape)

        def _opt(tau):

            Z = np.array([np.max(self._softmax1d(agg, tau)) for agg in m_agg])
            Y = np.argsort(-m_agg, axis=1)[:, 0]
            Z_idx = np.argsort(-Z)

            theta_tmp = self.theta
            N_tp = 0.0
            N_h = 0.0

            for i in range(len(Z_idx)):
                N_h = N_h + 1
                if Y[Z_idx[i]] == labels[Z_idx[i]]:
                    N_tp = N_tp + 1
                    # print(N_tp / N_h)

                if (N_tp / N_h < self.prec and N_h > 100):
                    break
                else:
                    theta_tmp = Z[Z_idx[i]]
            return (tau, theta_tmp, N_tp, N_h)

        # tau_range = range(1, 16)
        # print('prec = {0}, q = {1}, tau = {2}'.format(self.prec, self.q, self.tau))
        df = _opt(self.tau)
        self.theta = df[1]
        self.opt_model = df
        # print(df)
        # m = df.as_matrix()
        # idx_max = m[:, 2].argmax()
        # print(idx_max)
        # print(df, self.q)
        # print(m[idx_max, :])
        # self.tau, self.theta = m[idx_max, 0:2]
        # print(self.tau, self.theta)

    def inference(self, features, labels):

        A_feature = features.toarray()
        row_index = np.array(np.where(A_feature.sum(axis=1) > 0)[0])

        predictions = []
        p_softs = []
        M_prob = np.transpose(np.power(self.proba * self.W, 1))
        # M_prob2 = M_prob * self.W
        for row in A_feature:
            pred = row.dot(M_prob)
            p_soft = self._softmax1d(pred)
            i_max = np.argmax(p_soft)
            p_softs.append(p_soft[i_max])
            predictions.append(i_max)

        res = np.unique(np.array(predictions)[row_index] == np.array(labels)[row_index], return_counts=True)

        row_index2 = sorted(set(row_index).intersection(np.where(np.array(p_softs) > self.theta)[0]))
        res2 = np.unique(np.array(predictions)[row_index2] == np.array(labels)[row_index2], return_counts=True)

        print('prec = {0}\t q = {1}\t tau = {2}\t opt_model={3}\t p1={4}\t p2={5}'
              .format(self.prec, self.q, self.tau, self.opt_model, res, res2))

        return res, row_index, res2


def dblp(cv, tau, prec, q=1.0):
    db = client['accuracy']
    model = Collective2(db, cv=cv, tau=tau, prec=prec, q=q)

    titles_train = model.extract(TRAIN, 'title')
    author_train = model.extract(TRAIN, 'author')
    at_train = [a + ' ' + t for (a, t) in zip(author_train, titles_train)]

    titles_valid = model.extract(VALID, 'title')
    author_valid = model.extract(VALID, 'author')
    at_valid = [a + ' ' + t for (a, t) in zip(titles_valid, author_valid)]

    titles_test = model.extract(TEST, 'title')
    author_test = model.extract(TEST, 'author')
    at_test = [a + ' ' + t for (a, t) in zip(author_test, titles_test)]

    # X_train, X_test = model.featuring(titles_train, titles_test),
    X_train, X_valid, X_test = model.featuring(at_train, at_valid, at_test)
    y_train = model.extract(TRAIN, 'lbl')
    y_valid = model.extract(VALID, 'lbl')
    y_test = model.extract(TEST, 'lbl')

    model.training(X_train, y_train)

    prec_range = np.arange(.85, .96, .05)
    # prec_range = np.arange(.2, .36, .05)

    q_range = np.arange(1.0, 2.1, 0.5)
    tau_range = np.arange(1, 6)

    for (prec, q, tau) in itertools.product(prec_range, q_range, tau_range):
        model._update(prec, q, tau)
        model.validating(X_valid, y_valid)
        res, row_index, res2 = model.inference(X_test, y_test)
        # print('testing: {0}/{1}'.format(res, res2))


def movie(cv, tau, prec, q=1.0):
    db = client['movies']
    model = Collective2(db, cv=cv, tau=tau, prec=prec, q=q)

    titles_train = model.extract(TRAIN, 'title')
    actors_train = model.extract(TRAIN, 'actors')
    director_train = model.extract(TRAIN, 'director')
    at_train = [a + ' ' + d + ' ' + t for (a, d, t) in zip(actors_train, director_train, titles_train)]

    titles_valid = model.extract(VALID, 'title')
    actors_valid = model.extract(VALID, 'actors')
    director_valid = model.extract(VALID, 'director')
    at_valid = [a + ' ' + d + ' ' + t for (a, d, t) in zip(actors_valid, director_valid, titles_valid)]

    titles_test = model.extract(TEST, 'title')
    actors_test = model.extract(TEST, 'actors')
    director_test = model.extract(TEST, 'director')
    at_test = [a + ' ' + d + ' ' + t for (a, d, t) in zip(actors_test, director_test, titles_test)]

    X_train, X_valid, X_test = model.featuring(at_train, at_valid, at_test)
    y_train = model.extract(TRAIN, 'lbl')
    y_valid = model.extract(VALID, 'lbl')
    y_test = model.extract(TEST, 'lbl')

    model.training(X_train, y_train)

    prec_range = np.arange(.8, .96, .05)
    # prec_range = np.arange(.6, .76, .05)

    q_range = np.arange(1.0, 2.1, 0.5)
    tau_range = np.arange(1, 6)

    for (prec, q, tau) in itertools.product(prec_range, q_range, tau_range):
        model._update(prec, q, tau)
        model.validating(X_valid, y_valid)
        res, row_index, res2 = model.inference(X_test, y_test)
        # print('testing: {0}/{1}'.format(res, res2))


def restaurant(cv, tau, prec, q=1.0):
    db = client['restaurant']
    model = Collective2(db, cv=cv, tau=tau, prec=prec, q=q)

    type_train = model.extract(TRAIN, 'type')
    city_train = model.extract(TRAIN, 'city')
    name_train = model.extract(TRAIN, 'name')
    at_train = [str(a) + ' ' + str(d) + ' ' + t for (a, d, t) in zip(city_train, name_train, type_train)]

    type_valid = model.extract(VALID, 'type')
    city_valid = model.extract(VALID, 'city')
    name_valid = model.extract(VALID, 'name')
    at_valid = [a + ' ' + d + ' ' + t for (a, d, t) in zip(city_valid, name_valid, type_valid)]

    type_test = model.extract(TEST, 'type')
    city_test = model.extract(TEST, 'city')
    name_test = model.extract(TEST, 'name')
    at_test = [a + ' ' + d + ' ' + t for (a, d, t) in zip(city_test, name_test, type_test)]

    X_train, X_valid, X_test = model.featuring(at_train, at_valid, at_test)
    y_train = model.extract(TRAIN, 'lbl')
    y_valid = model.extract(VALID, 'lbl')
    y_test = model.extract(TEST, 'lbl')

    model.training(X_train, y_train)

    prec_range = np.arange(.8, .96, .05)
    # prec_range = np.arange(.6, .76, .05)

    q_range = np.arange(1.0, 2.1, 0.5)
    tau_range = np.arange(1, 6)

    for (prec, q, tau) in itertools.product(prec_range, q_range, tau_range):
        model._update(prec, q, tau)
        model.validating(X_valid, y_valid)
        res, row_index, res2 = model.inference(X_test, y_test)
        # print('testing: {0}/{1}'.format(res, res2))


if __name__ == "__main__":
    # dblp()
    # cv11 = CountVectorizer(min_df=1, max_df=0.5, ngram_range=(1, 1), dtype='int16', stop_words='english')
    cv12 = CountVectorizer(min_df=1, max_df=0.5, ngram_range=(1, 2), dtype='int16', stop_words='english')
    # cv21 = CountVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 1), dtype='int16', stop_words='english')
    # cv22 = CountVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2), dtype='int16', stop_words='english')

    db = client['accuracy']
    # db = client['movies']

    train = db.get_collection(TRAIN)
    valid = db.get_collection(VALID)
    test = db.get_collection(TEST)

    test = db.get_collection(TEST)
    cnt = test.count()
    match = 0
    nomatch = 0
    n_iter = 1
    print(cnt)

    tau_range = range(1, 2)
    q_range = [1.0]
    prec_range = [0.5]

    try:
        for (prec, q, tau) in itertools.product(prec_range, q_range, tau_range):
            print(prec, q, tau)
            # res, res2 = movie(cv12, tau=tau, prec=prec, q=q)
            res, res2 = restaurant(cv12, tau=tau, prec=prec, q=q)
            print('tau = {0}'.format(tau))
            nomatch = nomatch + res2[1][0]
            match = match + res2[1][1]
            precision = (0.0 + match) / (nomatch + match)
            print(precision)
    except:
        pass

    # for cv in cvs:
    #     print(dblp(cv, alpha=1.0e-8))
    #     print(dblp(cv, alpha=1.0e-8))
    #     print(str(cv))

    pass
