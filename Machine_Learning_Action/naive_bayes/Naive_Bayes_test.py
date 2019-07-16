#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from Naive_Bayes import load_dataset, set_of_words2vec, create_vocab_list, train_nb


def classify_nb(vec2_classify, p0_vec, p1_vec, p_class1):
    """函数说明: 朴素贝叶斯分类器的分类函数

    Args:
        vec2_classify: 待分类的词条数组
        p0_vec: 非侮辱类的条件概率数组
        p1_vec: 侮辱类的条件概率数组
        p_class1: 文档属于侮辱类的概率

    Returns:
        0: 属于非侮辱类
        1: 属于侮辱类
    """
    # 对应元素相乘，log(A*B) = logA + logB
    p0 = sum(vec2_classify * p0_vec) + np.log(1.0 - p_class1)
    p1 = sum(vec2_classify * p1_vec) + np.log(p_class1)
    if p0 > p1:
        return 0
    else:
        return 1


def testing_bayes():
    """函数说明: 测试朴素贝叶斯分类器

    Returns:
        无
    """
    # 创建实验样本
    list_posts, list_classes = load_dataset()
    # 创建词汇表
    my_vocab_list = create_vocab_list(list_posts)
    train_mat = []
    for posting_doc in list_posts:
        # 将实验样本向量化
        train_mat.append(set_of_words2vec(my_vocab_list, posting_doc))
    # 训练朴素贝叶斯分类器
    p0_v, p1_v, p_abuse = train_nb(np.array(train_mat), np.array(list_classes))
    # 测试样本
    test_entries = [['love', 'my', 'dalmation'], ['stupid', 'garbage']]
    for test_entry in test_entries:
        # 测试样本向量化
        this_doc = np.array(set_of_words2vec(my_vocab_list, test_entry))
        if classify_nb(this_doc, p0_v, p1_v, p_abuse):
            print(test_entry, "属于侮辱类")
        else:
            print(test_entry, "属于非侮辱类")


if __name__ == '__main__':
    testing_bayes()
