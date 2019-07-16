#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from kNN import classify0
from os import listdir
# from sklearn.neighbors import KNeighborsClassifier as kNN


def img_vector(filename):
    """将手写图片转化为向量

    Args:
        filename: 文件名

    Returns:
        return_vec: 返回的向量
    """
    return_vec = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()  # 读取第i行
        for j in range(32):
            # return_vec第0行，32*i+j列被赋值为文本中第i行第j列的数
            return_vec[0, 32 * i + j] = int(line_str[j])
    return return_vec


def handwriting_class_test():
    """手写分类数字测试
    函数有两种表示形式
    """
    labels = []
    training_file_list = listdir('trainingDigits')  # 文件列表
    m = len(training_file_list)  # 文件数量
    training_mat = np.zeros((m, 1024))  # m*1024矩阵
    for i in range(m):
        # 第i个训练文件，如'0_0.txt'
        filename_str = training_file_list[i]
        # 用.对文件名进行切片，并且返回第一个，如'0_0'
        file_str = filename_str.split('.')[0]
        # 用_进行切片，返回训练集的真实标签
        class_num_str = int(file_str.split('_')[0])
        # 依次添加到标签列表
        labels.append(class_num_str)
        training_mat[i, :] = img_vector('trainingDigits/%s' % filename_str)

    #
    # # 构建kNN分类器
    # neigh = kNN(n_neighbors=3, algorithm='auto')
    # # 拟合模型，training_mat为训练矩阵，labels位对应的标签
    # neigh.fit(training_mat, labels)
    #

    # 返回test_digits目录下的文件列表
    test_file_list = listdir('testDigits')
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        # 第i个测试文件，如'0_0.txt'
        filename_str = test_file_list[i]
        # 用.对文件名进行切片，并且返回第一个，如'0_0'
        file_str = filename_str.split('.')[0]
        # 用_进行切片，返回训练集的真实标签
        class_num_str = int(file_str.split('_')[0])
        # 将文本文件转化为向量
        vector_under_test = img_vector('testDigits/%s' % filename_str)
        # 使用kNN分类器对向量进行分类
        classifier_result = classify0(vector_under_test, training_mat, labels, 3)
        #
        # 可把上面那一行改为下面的函数(import的地方也要改)
        # classifier_result = neigh.predict(vector_under_test)
        #
        print("The classifier came back with:%d, the real answer is :%d" % (classifier_result, class_num_str))
        if classifier_result != class_num_str:
            error_count += 1.0
    print("\nThe total number of errors is :%d" % error_count)
    print("\nThe total error rate is :%f" % (error_count / float(m_test)))


if __name__ == '__main__':
    handwriting_class_test()
