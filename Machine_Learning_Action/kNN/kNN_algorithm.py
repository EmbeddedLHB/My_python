#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from kNN import classify0
from operator import itemgetter


def create_dataset():
    """创建数据集

    Returns:
        create_group: 数据集
        create_labels: 分类标签
    """
    create_group = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])  # 四组二维特征
    create_labels = ['爱情片', '爱情片', '动作片', '动作片']  # 四组特征的标签
    return create_group, create_labels


def classify_old_method(inx, classify_dataset, classify_labels, k):
    """kNN算法，分类器

    Args:
        inx: 用于分类的数据(测试集)
        classify_dataset: 用于训练的数据(训练集)
        classify_labels: 分类标签
        k: kNN算法参数,选择距离最小的k个点

    Returns:
        sortedClassCount[0][0]: 分类结果
    """
    # train_set的行数
    train_set_size = classify_dataset.shape[0]
    # 二维特征的差
    diff_mat = np.tile(inx, (train_set_size, 1)) - classify_dataset
    # 样本点到训练集各点的距离
    distances = ((diff_mat ** 2).sum(axis=1)) ** 0.5
    # distance中元素从小到大排序后的索引值
    sorted_dist = distances.argsort()
    # 新建一个记录类别次数的字典
    class_count = {}

    # 选择距离最小的k个点
    for i in range(k):
        # 前k个元素的类别
        vote_label = classify_labels[sorted_dist[i]]
        # 更新类别次数
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


if __name__ == '__main__':
    group, labels = create_dataset()
    test = [101, 20]  # 测试集
    test_class = classify_old_method(test, group, labels, 3)  # kNN分类
    test_class2 = classify0(test, group, labels, 3)
    print(test_class)
    print(test_class2)
