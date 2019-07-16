#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def load_dataset():
    """函数说明: 创建实验样本

    Returns:
        posting_list: 实验样本切分的词条
        class_vec: 类别标签向量
    """
    # 切分的词条
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 类别标签向量
    class_vec = [0, 1, 0, 1, 0, 1]
    return posting_list, class_vec


def set_of_words2vec(vocab_list, input_set):
    """函数说明: 根据vocab_list词汇表，将input_set向量化，向量的每个元素为1或0

    Args:
        vocab_list: create_vocab_list返回的列表
        input_set: 切分的词条列表

    Returns:
        return_vec: 文档向量，词集模型
    """
    # 创建一个其中所含元素都为0的向量
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        # 如果词条存在于词汇表中，则置1
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print("The word: %s is not in my Vocabulary!" % word)
    return return_vec


def create_vocab_list(dataset):
    """函数说明: 将切分的实验样本词条整理成不重复的词条列表，也就是词汇表

    Args:
        dataset: 整理的样本数据集

    Returns:
        vocab_list: 返回不重复的词条列表，也就是词汇表
    """
    # 创建一个空的不重复列表
    vocab_set = set([])
    for document in dataset:
        # 每一次循环都取并集
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def train_nb(train_mat, train_category):
    """函数说明: 朴素贝叶斯分类器训练函数

    Args:
        train_mat: 训练文档矩阵，即set_of_word2vec返回的return_vec构成的矩阵
        train_category: 训练类别标签向量，即load_dataset返回的class_vec

    Returns:
        p0_vec: 非侮辱类的条件概率数组
        p1_vec: 侮辱类的条件概率数组
        p_abusive: 文档属于侮辱类的概率
    """
    # 计算训练的文档数目
    num_train_docs = len(train_mat)
    # 计算每篇文档的词条数
    num_words = len(train_mat[0])
    # 文档属于侮辱类的概率
    p_abusive = sum(train_category) / float(num_train_docs)
    # 初始化两个概率数组
    p0_num = np.ones(num_words)
    p1_num = np.ones(num_words)
    # 分母初始化为2(0+类别数目)，拉普拉斯平滑
    p0_den = 2.0
    p1_den = 2.0
    for i in range(num_train_docs):
        # 统计属于非侮辱类的条件概率所需的数据，即P(w0|0)，P(w1|0)，P(w2|0)...
        if train_category[i] == 0:
            p0_num += train_mat[i]
            p0_den += sum(train_mat[i])
        # 统计属于侮辱类的条件概率所需的数据，即P(w0|1)，P(w1|1)，P(w2|1)...
        else:
            p1_num += train_mat[i]
            p1_den += sum(train_mat[i])
    # 取对数，防止下溢出
    p0_vec = np.log(p0_num / p0_den)
    p1_vec = np.log(p1_num / p1_den)
    # 返回属于非侮辱类的条件概率数组，属于侮辱类的条件概率数组，文档属于侮辱类的概率
    return p0_vec, p1_vec, p_abusive


if __name__ == '__main__':
    my_posting_list, my_class_vec = load_dataset()
    my_vocab_list = create_vocab_list(my_posting_list)
    print("my_vocab_list:\n", my_vocab_list)
    train_matrix = []
    for posting_doc in my_posting_list:
        train_matrix.append(set_of_words2vec(my_vocab_list, posting_doc))
    p0_v, p1_v, p_abuse = train_nb(train_matrix, my_class_vec)
    print("p0v:\n", p0_v)
    print("p1v:\n", p1_v)
    print("class_vec:\n", my_class_vec)
    print("p_abuse:\n", p_abuse)
