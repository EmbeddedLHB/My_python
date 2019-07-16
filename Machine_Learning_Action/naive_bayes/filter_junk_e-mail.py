#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import re
import numpy as np
from Naive_Bayes import create_vocab_list, set_of_words2vec, train_nb
from Naive_Bayes_test import classify_nb
from functools import reduce


def text_parse(big_string):
    """函数说明: 接受一个大字符串并将其解析为字符串列表

    Args:
        big_string: 待接受的大字符串

    Returns:
        除I外全部转化为小写的字符串列表
    """
    # 将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    list_of_tokens = re.split(r'\W+', big_string)
    # 除了单个字母，例如大写的I，其它单词变成小写
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]


def spam_test():
    """函数说明: 测试朴素贝叶斯分类器

    Returns:
        无
    """
    doc_list = []
    class_list = []
    # 遍历25个txt文件
    for i in range(1, 26):
        # 读取每个垃圾邮件，并将字符串转换成字符串列表
        # 读取非垃圾邮件，0表示非垃圾邮件
        word_list = text_parse(open('email/ham/%d.txt' % i, 'r').read())
        doc_list.append(word_list)
        class_list.append(0)

        # 读取垃圾邮件，1表示垃圾邮件
        word_list = text_parse(open('email/spam/%d.txt' % i, 'r').read())
        doc_list.append(word_list)
        class_list.append(1)

    # 创建词汇表，不重复
    vocab_list = create_vocab_list(doc_list)
    # 创建存储训练集的索引值的列表和测试集的索引值的列表
    training_set = list(range(50))
    test_set = []

    # 从50个邮件中，随机挑选出40个作为训练集，10个作为测试集
    for i in range(10):
        # 随机选取索引值
        rand_index = random.randint(0, len(training_set) - 1)
        # 添加测试集的索引值
        test_set.append(training_set[rand_index])
        # 在训练集列表中删除添加到测试集的索引值
        del (training_set[rand_index])
    # 创建训练集矩阵和训练集类别标签系向量
    train_mat = []
    train_classes = []

    # 遍历训练集
    for doc_index in training_set:
        # 将生成的词集模型添加到训练矩阵中
        train_mat.append(set_of_words2vec(vocab_list, doc_list[doc_index]))
        # 将类别添加到训练集类别标签系向量中
        train_classes.append(class_list[doc_index])
    # 训练朴素贝叶斯模型
    p0_v, p1_v, p_spam = train_nb(np.array(train_mat), np.array(train_classes))
    # 错误分类计数
    error_count = 0.0

    # 遍历测试集
    for doc_index in test_set:
        # 测试集的词集模型
        word_vector = set_of_words2vec(vocab_list, doc_list[doc_index])
        # 如果分类错误，错误计数+1
        if classify_nb(np.array(word_vector), p0_v, p1_v, p_spam) != class_list[doc_index]:
            error_count += 1
            print("分类错误的测试集: ", doc_list[doc_index])
    print("错误率: %.2f%%" % (error_count / len(test_set) * 100))


if __name__ == '__main__':
    spam_test()
