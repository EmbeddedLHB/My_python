#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from AdaBoost import adaboost_train_ds, adaboost_classify


def load_data(filename):
    """函数说明: 导入马疝病数据集

    Args:
        filename: 文件名

    Returns:
        data_mat: 数据矩阵
        label_mat: 标签矩阵
    """
    # 特征的数量
    num_feat = len((open(filename).readline().split('\t')))
    # 初始化数据矩阵和标签矩阵
    data_mat = []
    label_mat = []
    with open(filename, 'r') as fr:
        # 对于文件的每一行
        for line in fr:
            line_arr = []
            cur_line = line.strip().split('\t')
            # 添加数据
            for i in range(num_feat - 1):
                line_arr.append(float(cur_line[i]))
            data_mat.append(line_arr)
            # 添加标签
            label_mat.append(float(cur_line[-1]))
    return data_mat, label_mat


if __name__ == '__main__':
    # 初始化训练集的数据矩阵
    dataArr, labelArr = load_data('horseColicTraining2.txt')
    # 训练分类器
    weakClassArr, aggClassRes = adaboost_train_ds(dataArr, labelArr)
    # 初始化测试集的数据矩阵
    testArr, testLabelArr = load_data('horseColicTest2.txt')
    print(weakClassArr)

    # 对训练集进行错误率测试
    predictions = adaboost_classify(dataArr, weakClassArr)
    errArr = np.mat(np.ones((len(dataArr), 1)))
    print('训练集的错误率: %.3f%%' %
          float(errArr[predictions != np.mat(labelArr).T].sum() / len(dataArr) * 100))

    # 对测试集进行错误率测试
    predictions = adaboost_classify(testArr, weakClassArr)
    errArr = np.mat(np.ones((len(testArr), 1)))
    print('测试集的错误率: %.3f%%' %
          float(errArr[predictions != np.mat(testLabelArr).T].sum() / len(testArr) * 100))
