#!/usr/bin/env python
# -*- coding: utf-8 -*-
from kNN import auto_norm, classify0, file2matrix


def dating_class_test():
    dating_data_mat, dating_labels = file2matrix('datingTestSet.txt')  # 返回的特征矩阵dating_data_mat和dating_labels
    ratio = 0.10  # 取所有数据的10%
    norm_mat, ranges, min_value = auto_norm(dating_data_mat)  # 数据归一化
    m = norm_mat.shape[0]  # norm_mat的行数
    num_of_test = int(m * ratio)  # 10%测试数据的个数
    error_count = 0.0  # 分类错误计数

    for i in range(num_of_test):
        # 前num_of_test个数据作为测试集，后m-num_of_test个数据作为训练集
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_of_test:m, :], dating_labels[num_of_test:m], 4)
        print("The classifier came back with: %d, the real answer is: %d" % (classifier_result, dating_labels[i]))
        if classifier_result != dating_labels[i]:
            error_count += 1.0
    print("The total error rate is: %.2f" % (error_count / float(num_of_test)))


if __name__ == '__main__':
    dating_class_test()
