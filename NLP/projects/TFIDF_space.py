#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer
from Tools import readfile, readbunchobj, writebunchobj


def vector_space(stopword_path, bunch_path, space_path, train_tfidf_path=None):
    """创建TF-IDF词向量空间

    Args:
        sublinear_tf: 用1+log(tf)代替tf，即采用亚线性策略
        max_df: 临时停用词的阈值
    Returns:
        无

    """
    # 读取停用词
    stop_word_list = readfile(stopword_path).splitlines()
    # 导入分词后的词向量bunch对象
    bunch = readbunchobj(bunch_path)
    # 构建tf-idf词向量空间对象
    tfidf_space = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[],
                        vocabulary={})

    # 训练集
    if train_tfidf_path is not None:
        train_bunch = readbunchobj(train_tfidf_path)
        tfidf_space.vocabulary = train_bunch.vocabulary
        vectorizer = TfidfVectorizer(stop_words=stop_word_list, sublinear_tf=True, max_df=0.5,
                                     vocabulary=train_bunch.vocabulary)
        tfidf_space.tdm = vectorizer.fit_transform(bunch.contents)

    # 测试集
    else:
        vectorizer = TfidfVectorizer(stop_words=stop_word_list, sublinear_tf=True, max_df=0.5)
        tfidf_space.tdm = vectorizer.fit_transform(bunch.contents)
        tfidf_space.vocabulary = vectorizer.vocabulary_

    writebunchobj(space_path, tfidf_space)
    print('tf-idf词向量空间实例创建成功！！！')


if __name__ == '__main__':
    stopword_path = 'C:/My_python/NLP/corpus/train_word_bag/hlt_stop_words.txt'
    bunch_path = 'C:/My_python/NLP/corpus/train_word_bag/train_set.dat'
    space_path = 'C:/My_python/NLP/corpus/train_word_bag/tfidfspace.dat'
    vector_space(stopword_path, bunch_path, space_path)

    bunch_path = 'C:/My_python/NLP/corpus/test_word_bag/test_set.dat'
    space_path = 'C:/My_python/NLP/corpus/test_word_bag/testspace.dat'
    train_tfidf_path = 'C:/My_python/NLP/corpus/train_word_bag/tfidfspace.dat'
    vector_space(stopword_path, bunch_path, space_path, train_tfidf_path)
