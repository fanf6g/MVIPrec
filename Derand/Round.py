import logging

import cvxpy
import jellyfish
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
NAME = 'name'
ADDR = 'addr'
CITY = 'city'
TYPE = 'type'
LBL = 'lbl'


class Round(Model):
    def __init__(self, db, cv=CountVectorizer(min_df=2, ngram_range=(1, 2), dtype='int16')):
        super().__init__(db, cv)

    def training(self, features, labels):
        cities_train = self.extract(TRAIN, CITY)
        names_train = self.extract(TRAIN, NAME)
        addrs_train = self.extract(TRAIN, ADDR)
        types_train = self.extract(TRAIN, TYPE)

        cities_train.extend(self.extract(VALID, CITY))
        names_train.extend(self.extract(VALID, NAME))
        addrs_train.extend(self.extract(VALID, ADDR))
        types_train.extend(self.extract(VALID, TYPE))

        self.cities_train = list(map(lambda x: str(x), cities_train))
        self.names_train = list(map(lambda x: str(x), names_train))
        self.addrs_train = list(map(lambda x: str(x), addrs_train))
        self.types_train = list(map(lambda x: str(x), types_train))
        pass

    def inference(self, features, labels):
        cities_test = self.extract(TEST, CITY)
        names_test = self.extract(TEST, NAME)
        addrs_test = self.extract(TEST, ADDR)
        types_test = self.extract(TEST, TYPE)

        self.cities_test = list(map(lambda x: str(x), cities_test))
        self.names_test = list(map(lambda x: str(x), names_test))
        self.addrs_test = list(map(lambda x: str(x), addrs_test))
        # self.types_test = list(map(lambda x: str(x), types_test))

        Nghb_name_test_train = self._edit_dist(self.names_train, self.names_test) <= 10
        Nghb_addr_test_train = self._edit_dist(self.addrs_train, self.addrs_test) <= 7
        # Nghb_type_test_train = self._edit_dist(self.types_train, self.types_test) <= 5

        Nghb_name_test_test = self._edit_dist(self.names_test, self.names_test) <= 10
        Nghb_addr_test_test = self._edit_dist(self.names_test, self.addrs_test) <= 7
        # Nghb_type_test_test = self._edit_dist(self.names_test, self.types_test) <= 5

        Nghb_test_train = np.array(Nghb_name_test_train * Nghb_addr_test_train)
        Nghb_test_test = Nghb_name_test_test * Nghb_addr_test_test

        cities = np.array(sorted(set(self.cities_train)))
        n_lbl = len(cities)
        V = np.array(self._edit_dist(cities, cities) > 3, dtype='int8')
        for i in range(n_lbl):
            V[i, i] = 0

        city2id = dict([(v, k) for (k, v) in enumerate(cities)])

        n_test = 80
        n_train = 720

        def compress(city2id, nghb, city_train):
            np_city = np.array(city_train)
            tmp = []
            for i, row in enumerate(nghb):
                nb = np_city[row]
                r = np.zeros(n_lbl, dtype='int8')
                for c in nb:
                    r[city2id[c]] = 1

                tmp.append(r)

            return np.array(tmp)

        W_test_train = compress(city2id, Nghb_test_train, self.cities_train)
        P_w_test_train = cvxpy.Parameter(n_test, n_lbl, name='candidates', sign='positive', value=W_test_train)

        P_w_test_test = cvxpy.Parameter(n_test, n_test, name='candidates', sign='positive', value=Nghb_test_test)
        print(Nghb_test_test.shape)

        print(np.sum(np.sum(W_test_train, axis=1) > 0))

        X_lp = cvxpy.Variable(n_test, n_lbl)
        objective = cvxpy.Maximize(cvxpy.sum_entries(cvxpy.mul_elemwise(P_w_test_train, X_lp)))

        '''约束: 公式(2)'''
        ONE = cvxpy.Parameter(n_lbl, sign='positive')
        ONE.value = np.ones(n_lbl, dtype='int16')

        X_lp * ONE
        P_v = cvxpy.Parameter(n_lbl, n_lbl, sign='positive', value=V)

        constraint = [X_lp * ONE <= 1,  # 约束(2)
                      P_w_test_test * X_lp * P_v <= 1,  # 约束(3)
                      X_lp >= 0, X_lp <= 1  # 约束(4)]
                      ]

        prob = cvxpy.Problem(objective, constraint)
        prob.solve()
        pred = np.argmax(X_lp.value, axis=1).flatten()
        print(pred)
        print(self.cities_test)
        gtrue = np.array([city2id.get(city, 0) for city in self.cities_test])
        print(gtrue)
        print(np.sum(gtrue == pred))

        # P_v = cvxpy.Parameter(n_lbl, n_lbl, sign='positive', value=M_city)
        #
        # cons2 = X_lp * ONE <= 1

        # P_mask = P_w * X_lp * P_v
        #
        # print(P_mask.value)
        #
        # objective = cvxpy.Maximize(cvxpy.sum_entries(X_lp))

        # prob = cvxpy.Problem(objective,
        #                      [
        #                          X_lp * ONE <= 1  # 约束(2)
        #                          , cvxpy.mul_elemwise(P_mask, X_lp) <= 1  # 约束(3)
        #                          , X_lp >= 0, X_lp <= 1  # 约束(4)
        #                      ])
        #
        # prob.solve()
        #
        # # print(np.sum(np.sum(M_city, axis=1) > 1))
        #
        #
        # id2city = dict(enumerate(sorted(set(self.cities_train))))
        # city2id = dict([(v, k) for (k, v) in enumerate(sorted(set(self.cities_train)))])
        #
        # print(np.array(M_city, dtype='int8'))

        pass

    def _edit_dist(self, tests, train):
        V_edit = np.vectorize(jellyfish.levenshtein_distance)
        M_sim = np.array([V_edit(tests, t) for t in train])
        return M_sim


        # self.N_I = len(labels)


        #     lbl = []
        #     lbl.extend(self.labels_train)
        #     lbl.extend(self.labels_test)
        #
        #     self.N_lbl = len(set(lbl))
        #
        #     I_A = self.derand()
        #
        #     lbl_test = np.array(labels)
        #
        #     prd = np.argmax(np.array(I_A.value), axis=1)
        #     truth = prd == lbl_test
        #
        #     ind = np.where(truth == True)
        #     print(np.unique(ind, return_counts=True))
        #
        # def _neighbour_jaccard(self, csr_matrix1, csr_matrix2, threshold=0.8):
        #
        #     M_sim = csr_matrix1.dot(csr_matrix2.T).toarray()
        #
        #     V_test = np.sum(csr_matrix1.toarray(), axis=1)
        #     V_train = np.sum(csr_matrix2.toarray(), axis=1)
        #     denorm = np.log(np.outer(np.exp(V_test), np.exp(V_train))) + 1.0e-8 - M_sim
        #     SIM = M_sim / denorm
        #     i_sim = np.where(SIM >= threshold)
        #
        #     logging.info('test.shape={0}, train.shape={1}'.format(csr_matrix1.shape, csr_matrix2.shape))
        #     M_nghb = csr_matrix((np.ones(len(i_sim[0]), dtype='int16'), (i_sim[0], i_sim[1])),
        #                         shape=M_sim.shape,
        #                         dtype='int16').toarray()
        #     return M_nghb
        #
        # def X_nghb_test_train(self, alpha=0.8):
        #     '''产生近邻矩阵<M_nghb>, N_I * N_C '''
        #     logging.info('X_nghb')
        #     test = self.feature_test
        #     train = self.feature_train
        #
        #     self.M_nghb_test_train = self._neighbour_jaccard(test, train, threshold=alpha)
        #     # return M_nghb
        #
        # def X_nghb_test_test(self, alpha=0.8):
        #     '''产生近邻矩阵<M_nghb>, N_I * N_I '''
        #     logging.info('X_nghb')
        #
        #     def _jaro(s1, s2):
        #         return jellyfish.jaro_distance(s1, s2)
        #
        #     V_jaro = np.vectorize(_jaro)
        #
        #     test = self.texts_test
        #     train = self.texts_test
        #
        #     M_sim = []
        #     for (i, t_i) in enumerate(test):
        #         v_sim = V_jaro(t_i, train)
        #         M_sim.append(v_sim)
        #         print(i, t_i)
        #
        #     M_sim = np.array(M_sim)
        #
        #     for i in range(self.N_I):
        #         M_sim[i, i] = 0
        #
        #     i_sim = np.where(M_sim >= alpha)
        #
        #     # logging.info('test.shape={0}, train.shape={1}'.format(test.shape, train.shape))
        #     M_nghb = csr_matrix((np.ones(len(i_sim[0]), dtype='int16'), (i_sim[0], i_sim[1])),
        #                         shape=M_sim.shape,
        #                         dtype='int16').toarray()
        #     self.M_nghb_test_test = M_nghb
        #
        #     # assert np.all(self.M_nghb_test_test >= 0)
        #     # return M_nghb
        #
        # def A_can(self):
        #     '''N_I * N_lbl '''
        #     logging.info('X_can')
        #     W = []
        #     for (i, row) in enumerate(self.M_nghb_test_train):
        #         nb = np.where(row > 0)
        #         j_idx = [self.labels_train[j] for j in nb[0]]
        #         w = np.zeros(self.N_lbl, dtype='int16')
        #         w[j_idx] = 1
        #         W.append(w)
        #
        #     self.can = np.array(W)
        #     # return self.can
        #
        # def A_vio(self, col_target, alpha=0.9):
        #     '''N_lbl * N_lbl '''
        #     logging.info('A_vio')
        #     J_train = self.extract(TRAIN, col_target)
        #     J_valid = self.extract(VALID, col_target)
        #     J_test = self.extract(TEST, col_target)
        #
        #     J = []
        #     J.extend(J_train)
        #     J.extend(J_valid)
        #     J.extend(J_test)
        #
        #     L = []
        #     L.extend(self.labels_train)
        #     L.extend(self.labels_test)
        #
        #     d = dict(set(zip(L, J)))
        #     n_lbl = self.N_lbl
        #     V = np.zeros((n_lbl, n_lbl), dtype='int16')
        #     print(self.N_lbl)
        #     for (i, j) in itertools.product(range(n_lbl), range(n_lbl)):
        #         sim = jellyfish.jaro_distance(d.get(i, ''), d.get(j, ''))
        #         # if i % 1000 == 0:
        #         #     print(i)
        #         if sim < alpha:
        #             V[i, j] = 1
        #     self.A_V = V
        #     # return V
        #
        # def A_linPrg(self):
        #     '''线性规划'''
        #     '''产生目标属性值冲突矩阵<M_v>: 约束(3)'''
        #     logging.info('A_linPrg')
        #     self.A_vio(CITY)
        #     N_lbl = self.N_lbl
        #     N_i = self.N_I
        #
        #     '''构建候选值参数矩阵 P_w'''
        #     self.A_can()
        #     P_w = Parameter(N_i, N_lbl, sign='positive', value=self.can)
        #
        #     '''构建线性规划矩阵<X_lp>'''
        #     X_lp = Variable(N_i, N_lbl)
        #
        #     '''约束: 公式(2)'''
        #     ONE = Parameter(N_lbl, sign='positive')
        #     ONE.value = np.ones(N_lbl, dtype='int16')
        #
        #     P_v = Parameter(N_lbl, N_lbl, sign='positive', value=self.A_V)
        #
        #     P_mask = P_w * P_v
        #     '''约束(4)'''
        #     prob = Problem(Maximize(sum_entries(mul_elemwise(P_w, X_lp))),
        #                    [
        #                        X_lp * ONE <= 1  # 约束(2)
        #                        , mul_elemwise(P_mask, X_lp) <= 1  # 约束(3)
        #                        , X_lp >= 0, X_lp <= 1  # 约束(4)
        #                    ])
        #
        #     prob.solve()
        #     self.X_lp = X_lp
        #
        # def _Pr(self, epsilon=0.1):
        #     '''
        #     公式(5)
        #     公式(6)
        #     '''
        #     X_lp = np.array(self.X_lp.value)
        #     X_ij = X_lp + epsilon
        #
        #     Pr = X_ij.T / (np.sum(X_ij, axis=1) + 1)
        #     Pr = Pr.T
        #     Pr_ = 1 / (np.sum(X_ij, axis=1) + 1)
        #
        #     self.Pr = Pr
        #     self.Pr_ = Pr_
        #
        #     # assert np.all(Pr.T + Pr_ == 1)
        #
        #     return (Pr, Pr_)
        #
        # def _PrDD_I(self):
        #     '''
        #     公式(7)
        #     公式(8)
        #     '''
        #     Pr = self.Pr
        #     Pr_ = self.Pr_
        #     # N_I = self.N_I
        #     M = self.M_nghb_test_test
        #     # for i in range(N_I):
        #     #     if M[i, i] >= 1:
        #     #         M[i, i] = M[i, i] - 1
        #
        #     Prdd_l = (M.dot(Pr).T + Pr_.T).T
        #     Prdd_ln = np.log(Prdd_l)
        #     Prdd = M.dot(Prdd_ln)
        #
        #     self.Prdd7 = np.exp(Prdd_l)
        #     self.Prdd8 = np.exp(Prdd)
        #     # return self.Prdd
        #
        # def _expct(self):
        #     '''
        #     公式(9)
        #     '''
        #     E_w = self.Pr * self.Prdd8
        #     return E_w
        #
        # def conexp(self, E_i_1, i, j=None):
        #     # print(i)
        #     E = E_i_1
        #     Pr_i = self.Pr[i]
        #     E = E - np.sum(self.Prdd8[i] * Pr_i)
        #
        #     if j is not None:
        #         E = E + 1
        #
        #     nghb = np.where(self.M_nghb_test_test[i] > 0)[0]
        #
        #     l_nghb = filter(lambda l: l > i, nghb)
        #
        #     for l in l_nghb:
        #
        #         can_l = np.where(self.can[l] > 0)[0]
        #
        #         for k in can_l:
        #             '''代码实现'''
        #             Pr_l_k = self.Pr[l, k]
        #             E = E - self.Prdd8[l, k] * Pr_l_k
        #             if j is None or self.A_V[j, k] == 0:
        #                 self.Prdd8[l, k] = self.Prdd8[l, k] / self.Prdd7[l, k]
        #                 E = E + Pr_l_k * self.Prdd8[l, k]
        #
        #     return E
        #
        # def derand(self):
        #     test = self.feature_test
        #
        #     # 产生邻近矩阵
        #     self.X_nghb_test_train()
        #     self.X_nghb_test_test()
        #
        #     # 采用线性规划, 计算候选值的得分
        #     self.A_linPrg()
        #
        #     I_A = self.X_lp
        #
        #     return I_A
        #
        # def _jaccard(self, s1, s2):
        #     a1 = s1.split(' ')
        #     a2 = s2.split(' ')
        #     pass
        #
        # def featuring(self, texts_train, texts_valid, texts_test):
        #     '''
        #     将文本数据向量化
        #     :param texts_train:
        #     :param texts_test:
        #     :return:
        #     '''
        #     # 将每个文本属性转换为独立的特征序列
        #
        #     self.texts_train = texts_train
        #     self.texts_valid = texts_valid
        #     self.texts_test = texts_test
        #
        #     txt = []
        #     txt.extend(texts_train)
        #     txt.extend(texts_valid)
        #     txt.extend(texts_test)
        #
        #     self.cv.fit(txt)
        #     logging.info('{0} features extracted!'.format(len(self.cv.vocabulary_)))
        #     features_train = self.cv.transform(texts_train)
        #     features_valid = self.cv.transform(texts_valid)
        #     features_test = self.cv.transform(texts_test)
        #
        #     self.feature_train = features_train
        #     self.feature_test = features_test
        #
        #     return (features_train, features_valid, features_test)


def restaurant():
    cv12 = CountVectorizer(dtype='int16', stop_words='english')
    db = client['restaurant']
    model = Round(db, cv12)
    titles_train = model.extract(TRAIN, ADDR)
    author_train = model.extract(TRAIN, NAME)
    at_train = [a + ' ' + t for (a, t) in zip(author_train, titles_train)]

    titles_valid = model.extract(VALID, ADDR)
    author_valid = model.extract(VALID, NAME)
    at_valid = [a + ' ' + t for (a, t) in zip(titles_valid, author_valid)]

    titles_test = model.extract(TEST, ADDR)
    author_test = model.extract(TEST, NAME)
    at_test = [a + ' ' + t for (a, t) in zip(author_test, titles_test)]

    # X_train, X_test = model.featuring(titles_train, titles_test),
    X_train, X_valid, X_test = model.featuring(at_train, at_valid, at_test)
    y_train = model.extract(TRAIN, LBL)
    y_valid = model.extract(VALID, LBL)
    y_test = model.extract(TEST, LBL)

    lbls = []
    lbls.extend(y_train)
    lbls.extend(y_valid)
    lbls.extend(y_test)

    model.training(X_train, y_train)
    model.validating(X_valid, y_valid)
    # model.compatible(targets, lbls)
    model.inference(X_test, y_test)


if __name__ == "__main__":
    cv12 = CountVectorizer(dtype='int16', stop_words='english')
    db = client['restaurant']
    model = Round(db, cv12)
    model.training([], [])
    model.inference([], [])

    pass
