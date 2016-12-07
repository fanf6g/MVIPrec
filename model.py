import logging

from sklearn.feature_extraction.text import CountVectorizer

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')


class Model:
    def __init__(self, db, cv=CountVectorizer(dtype='int16')):
        self.db = db
        self.cv = cv

    def extract(self, tbl_name, col):
        '''
        从数据表 tbl_name 中读取 col 列的属性值
        :param tbl_name: 数据表名
        :param col: 属性列名
        :return:
        '''
        logging.info('reading the values of column [{0}] in table [{1}] from db [{2}]'.format(col, tbl_name, self.db))
        tbl = self.db.get_collection(tbl_name)
        values = [rec.get(col) for rec in tbl.find()]
        return values

    def featuring(self, texts_train, texts_valid, texts_test):
        '''
        将文本数据向量化
        :param texts_train:
        :param texts_test:
        :return:
        '''
        # 将每个文本属性转换为独立的特征序列

        self.cv.fit(texts_train)
        vocab_train = self.cv.vocabulary_

        tmpcv_valid = CountVectorizer(max_df=self.cv.max_df, ngram_range=self.cv.ngram_range, dtype=self.cv.dtype)
        vocab_valid = tmpcv_valid.fit(texts_valid).vocabulary_

        tmpcv_test = CountVectorizer(max_df=self.cv.max_df, ngram_range=self.cv.ngram_range, dtype=self.cv.dtype)
        vocab_test = tmpcv_test.fit(texts_test).vocabulary_

        vocab_valid_test = set(vocab_valid.keys()).union(vocab_test.keys())

        vocab = sorted(vocab_valid_test.intersection(vocab_train.keys()))
        index = [(k, i) for (i, k) in enumerate(vocab)]
        self.cv.vocabulary_ = dict(index)
        features_train = self.cv.transform(texts_train)

        logging.info('{0} features extracted!'.format(len(self.cv.vocabulary_)))
        features_valid = self.cv.transform(texts_valid)
        features_test = self.cv.transform(texts_test)
        return (features_train, features_valid, features_test)

    def training(self, features, labels):
        pass

    def validating(self, features, labels):
        pass

    def inference(self, features, labels):
        pass
