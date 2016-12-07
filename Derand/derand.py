import itertools
import logging

import jellyfish
import numpy as np
import pymongo
from cvxpy import *
from scipy.sparse.csr import csr_matrix
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
JOURNAL = 'journal'
YEAR = 'year'
ALPHA = 3


class Derand(Model):
    def __init__(self, db, cv=CountVectorizer(min_df=2, ngram_range=(1, 2), dtype='int16')):
        super().__init__(db, cv)
        self.labels_train = []
        self.labels_test = []

    def training(self, features, labels):
        self.labels_train.extend(labels)
        pass

    def validating(self, features, labels):
        self.labels_train.extend(labels)
        pass

    def inference(self, features, labels):
        # self.labels.extend(labels)
        self.labels_test.extend(labels)

        self.N_I = len(labels)
        lbl = []
        lbl.extend(self.labels_train)
        lbl.extend(self.labels_test)

        self.N_lbl = len(set(lbl))

        I_A = np.array(self.derand())

        lbl_test = np.array(labels)
        truth = I_A == lbl_test
        print(np.array(truth, dtype='int16').sum())

    def X_nghb_test_train(self, alpha=ALPHA):
        '''产生近邻矩阵<M_nghb>, N_I * N_C '''
        logging.info('X_nghb')
        test = self.feature_test
        train = self.feature_train

        M_sim = test.dot(train.T).toarray()

        # V_test = np.sum(test.toarray(), axis=1)
        # V_train = np.sum(train.toarray(), axis=1)
        # denorm = np.outer(V_test, V_train) + 1.0e-8 - M_sim
        # SIM = M_sim / denorm
        i_sim = np.where(M_sim >= alpha)

        logging.info('test.shape={0}, train.shape={1}'.format(test.shape, train.shape))
        M_nghb = csr_matrix((np.ones(len(i_sim[0]), dtype='int16'), (i_sim[0], i_sim[1])),
                            shape=M_sim.shape,
                            dtype='int16').toarray()
        self.M_nghb_test_train = M_nghb
        # return M_nghb

    def X_nghb_test_test(self, alpha=ALPHA):
        '''产生近邻矩阵<M_nghb>, N_I * N_I '''
        logging.info('X_nghb')

        test = self.feature_test
        train = self.feature_test

        M_sim = test.dot(train.T).toarray()

        i_sim = np.where(M_sim >= alpha)
        # print(i_sim)
        logging.info('test.shape={0}, train.shape={1}'.format(test.shape, train.shape))
        M_nghb = csr_matrix((np.ones(len(i_sim[0]), dtype='int16'), (i_sim[0], i_sim[1])),
                            shape=M_sim.shape,
                            dtype='int16').toarray()

        for i in range(self.N_I):
            a_i = M_nghb[i, i]
            if a_i >= 1:
                M_nghb[i, i] = a_i - 1

        self.M_nghb_test_test = M_nghb
        # assert np.all(self.M_nghb_test_test >= 0)
        # return M_nghb

    def A_can(self):
        '''N_I * N_lbl '''
        logging.info('X_can')
        W = []
        for (i, row) in enumerate(self.M_nghb_test_train):
            nb = np.where(row > 0)
            j_idx = [self.labels_train[j] for j in nb[0]]
            w = np.zeros(self.N_lbl, dtype='int16')
            w[j_idx] = 1
            W.append(w)

        self.can = np.array(W)
        # return self.can

    def A_vio(self, col_target, alpha=0.8):
        '''N_lbl * N_lbl '''
        logging.info('A_vio')
        J_train = self.extract(TRAIN, col_target)
        J_valid = self.extract(VALID, col_target)
        J_test = self.extract(TEST, col_target)

        J = []
        J.extend(J_train)
        J.extend(J_valid)
        J.extend(J_test)

        L = []
        L.extend(self.labels_train)
        L.extend(self.labels_test)

        d = dict(set(zip(L, J)))
        n_lbl = self.N_lbl
        V = np.zeros((n_lbl, n_lbl), dtype='int16')
        print(self.N_lbl)
        for (i, j) in itertools.product(range(n_lbl), range(n_lbl)):
            sim = jellyfish.jaro_distance(d.get(i, ''), d.get(j, ''))
            # if i % 1000 == 0:
            #     print(i)
            if sim < alpha:
                V[i, j] = 1
        self.A_V = V
        # return V

    def A_linPrg(self):
        '''线性规划'''
        '''产生目标属性值冲突矩阵<M_v>: 约束(3)'''
        logging.info('A_linPrg')
        self.A_vio(JOURNAL)
        N_lbl = self.N_lbl
        N_i = self.N_I

        '''构建候选值参数矩阵 P_w'''
        self.A_can()
        P_w = Parameter(N_i, N_lbl, sign='positive', value=self.can)

        '''构建线性规划矩阵<X_lp>'''
        X_lp = Variable(N_i, N_lbl)

        '''约束: 公式(2)'''
        ONE = Parameter(N_lbl, sign='positive')
        ONE.value = np.ones(N_lbl, dtype='int16')

        P_v = Parameter(N_lbl, N_lbl, sign='positive', value=self.A_V)

        P_mask = P_w * P_v
        '''约束(4)'''
        prob = Problem(Maximize(sum_entries(mul_elemwise(P_w, X_lp))),
                       [
                           X_lp * ONE <= 1  # 约束(2)
                           , mul_elemwise(P_mask, X_lp) <= 1  # 约束(3)
                           , X_lp >= 0, X_lp <= 1  # 约束(4)
                       ])

        prob.solve()
        self.X_lp = X_lp

    def _Pr(self, epsilon=0.1):
        '''
        公式(5)
        公式(6)
        '''
        X_lp = np.array(self.X_lp.value)
        X_ij = X_lp + epsilon

        Pr = X_ij.T / (np.sum(X_ij, axis=1) + 1)
        Pr = Pr.T
        Pr_ = 1 / (np.sum(X_ij, axis=1) + 1)

        self.Pr = Pr
        self.Pr_ = Pr_

        # assert np.all(Pr.T + Pr_ == 1)

        return (Pr, Pr_)

    def _PrDD_I(self):
        '''
        公式(7)
        公式(8)
        '''
        Pr = self.Pr
        Pr_ = self.Pr_
        # N_I = self.N_I
        M = self.M_nghb_test_test
        # for i in range(N_I):
        #     if M[i, i] >= 1:
        #         M[i, i] = M[i, i] - 1

        Prdd_l = (M.dot(Pr).T + Pr_.T).T
        Prdd_ln = np.log(Prdd_l)
        Prdd = M.dot(Prdd_ln)

        self.Prdd7 = np.exp(Prdd_l)
        self.Prdd8 = np.exp(Prdd)
        # return self.Prdd

    def _expct(self):
        '''
        公式(9)
        '''
        E_w = self.Pr * self.Prdd8
        return E_w

    def conexp(self, E_i_1, i, j=None):
        # print(i)
        E = E_i_1
        Pr_i = self.Pr[i]
        E = E - np.sum(self.Prdd8[i] * Pr_i)

        if j is not None:
            E = E + 1

        nghb = np.where(self.M_nghb_test_test[i] > 0)[0]

        l_nghb = filter(lambda l: l > i, nghb)

        for l in l_nghb:

            can_l = np.where(self.can[l] > 0)[0]

            for k in can_l:
                '''代码实现'''
                Pr_l_k = self.Pr[l, k]
                E = E - self.Prdd8[l, k] * Pr_l_k
                if j is None or self.A_V[j, k] == 0:
                    self.Prdd8[l, k] = self.Prdd8[l, k] / self.Prdd7[l, k]
                    E = E + Pr_l_k * self.Prdd8[l, k]

        return E

    def derand(self):
        test = self.feature_test

        # 产生邻近矩阵
        self.X_nghb_test_train()
        self.X_nghb_test_test()

        # 采用线性规划, 计算候选值的得分
        self.A_linPrg()

        # 计算条件概率 P(r_i[A] = a_j) 和 P(r_i[A] = _)
        self._Pr()

        self._PrDD_I()

        E_0 = np.sum(self._expct())
        print(E_0)

        I_A = np.array([None] * self.N_I)
        E = np.zeros(self.N_I)
        # E_max = E_0
        # E[0] = E_0

        for i in range(self.N_I):
            # I_A[i] = None
            # E_i = E[i]

            if i == 0:
                E_i_1 = E_0
            else:
                E_i_1 = E[i - 1]

            E_max = self.conexp(E_i_1, i)

            for j in range(self.N_lbl):
                E_i = self.conexp(E_i_1, i, j)
                if E_i > E_max:
                    E_max = E_i
                    I_A[i] = j

            E[i] = E_max
            print('i = {0}, E_max = {1}, I_A{0}={2}'.format(i, E_max, I_A[i]))

        return I_A

    def _jaccard(self, s1, s2):
        a1 = s1.split(' ')
        a2 = s2.split(' ')
        pass

    def featuring(self, texts_train, texts_valid, texts_test):
        '''
        将文本数据向量化
        :param texts_train:
        :param texts_test:
        :return:
        '''
        # 将每个文本属性转换为独立的特征序列

        txt = []
        txt.extend(texts_train)
        txt.extend(texts_valid)

        self.cv.fit(txt)
        vocab_train = self.cv.vocabulary_

        tmpcv_test = CountVectorizer(min_df=1, max_df=self.cv.max_df, ngram_range=self.cv.ngram_range,
                                     dtype=self.cv.dtype)
        tmpcv_test.fit(texts_test)
        vocab_test = tmpcv_test.vocabulary_

        vocab = sorted(set(vocab_train.keys()).intersection(set(vocab_test.keys())))

        index = [(k, i) for (i, k) in enumerate(vocab)]
        self.cv.vocabulary_ = dict(index)
        features_train = self.cv.transform(txt)

        logging.info('{0} features extracted!'.format(len(self.cv.vocabulary_)))
        features_valid = self.cv.transform([''])
        features_test = self.cv.transform(texts_test)

        self.feature_train = features_train
        self.feature_test = features_test

        return (features_train, features_valid, features_test)


if __name__ == "__main__":
    cv12 = CountVectorizer(min_df=1, max_df=0.5, ngram_range=(1, 2), dtype='int16', stop_words='english')
    db = client['accuracy']
    model = Derand(db, cv12)
    titles_train = model.extract(TRAIN, 'title')
    titles_valid = model.extract(VALID, 'title')
    titles_test = model.extract(TEST, 'title')

    target_train = model.extract(TRAIN, JOURNAL)
    target_valid = model.extract(VALID, JOURNAL)
    target_test = model.extract(TEST, JOURNAL)

    targets = []
    targets.extend(target_train)
    targets.extend(target_valid)
    targets.extend(target_test)

    # X_train, X_test = model.featuring(titles_train, titles_test),
    X_train, X_valid, X_test = model.featuring(titles_train, titles_valid, titles_test)
    y_train = model.extract(TRAIN, 'lbl')
    y_valid = model.extract(VALID, 'lbl')
    y_test = model.extract(TEST, 'lbl')

    lbls = []
    lbls.extend(y_train)
    lbls.extend(y_valid)
    lbls.extend(y_test)

    model.training(X_train, y_train)
    model.validating(X_valid, y_valid)
    # model.compatible(targets, lbls)
    model.inference(X_test, y_test)

    pass
