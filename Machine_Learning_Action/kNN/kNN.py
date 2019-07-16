#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from collections import Counter


def classify0(inx, data_set, classify_labels, k):
    """kNN算法，分类器

    Args:
        inx: 用于分类的数据(测试集)
        data_set: 用于训练的数据(训练集)
        classify_labels: 分类标签
        k: kNN算法参数,选择距离最小的k个点

    Returns:
        sortedClassCount[0][0]: 分类结果
    """
    # 计算距离
    dist = np.sum((inx - data_set) ** 2, axis=1) ** 0.5
    # k个最近的标签
    k_labels = [classify_labels[index] for index in dist.argsort()[0:k]]
    # 出现次数最多的标签即为最终类别
    label = Counter(k_labels).most_common(1)[0][0]
    return label


def file2matrix(filename):
    """打开并解析文件，对数据进行分类：1代表不喜欢，2代表魅力一般，3代表极具魅力

    Args:
        filename: 文件名

    Returns:
        return_mat: 特征矩阵
        class_label_vector: 分类Label向量
    """
    fr = open(filename, 'r', encoding='utf-8')
    array_of_lines = fr.readlines()
    # 数据行数
    number_of_lines = len(array_of_lines)
    # numpy矩阵，解析完成的数据：numberOfLines行，3列
    return_mat = np.zeros((number_of_lines, 3))
    # 返回的分类标签向量
    class_label_vector = []
    # 行的索引值
    index = 0

    for line in array_of_lines:
        line = line.strip()
        list_from_line = line.split('\t')
        # 提取数据前三列
        return_mat[index, :] = list_from_line[0:3]
        # 根据喜欢程度进行分类，1代表不喜欢，2代表魅力一般，3代表极具魅力
        if list_from_line[-1] == 'didntLike':
            class_label_vector.append(1)
        elif list_from_line[-1] == 'smallDoses':
            class_label_vector.append(2)
        elif list_from_line[-1] == 'largeDoses':
            class_label_vector.append(3)
        # 移动到下一行
        index += 1
    return return_mat, class_label_vector


def auto_norm(data_set):
    """对数据进行归一化

    Args:
        data_set: 特征矩阵

    Returns:
        norm_data_set: 归一化后的特征矩阵
        ranges: 数据范围
        min: 数据最小值
    """
    # 数据的最大值、最小值
    min_value = data_set.min(0)
    max_value = data_set.max(0)
    # 最大值和最小值的范围
    ranges = max_value - min_value
    # data_set的行数
    m = data_set.shape[0]
    # 原始值减去最小值
    norm_data_set = data_set - np.tile(min_value, (m, 1))
    norm_data_set = norm_data_set / np.tile(ranges, (m, 1))
    # 返回归一化数据结果，数据范围，最小值
    return norm_data_set, ranges, min_value
