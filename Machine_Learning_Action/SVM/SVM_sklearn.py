#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import operator
import os
from sklearn.svm import SVC
from SVM_digits import img2vector


def head_writing_test():
    """函数说明: 手写数字分类测试

    Returns:
        无
    """
    # 测试集的labels
    labels = []
    # 所有文件的目录
    train_file = os.listdir('trainingDigits')
    m = len(train_file)
    # 初始化数据矩阵
    train_mat = np.zeros((m, 1024))
    for i in range(m):
        # 文件名
        filename = train_file[i]
        # 将获得的类别添加到labels中
        labels.append(int(filename[0]))
        train_mat[i, :] = img2vector('trainingDigits/%s' % filename)

    clf = SVC(C=200, kernel='rbf')
    clf.fit(train_mat, labels)

    # 返回testDigits目录下的文件列表
    test_file = os.listdir('testDigits')
    # 错误数
    error = 0
    # 测试数据的数量
    m_test = len(test_file)
    # 从文件中解析出测试集的类别并进行分类测试
    for i in range(m_test):
        # 文件名
        filename = test_file[i]
        # 分类的数字
        class_num = int(filename[0])
        # 获得向量
        vec = img2vector('testDigits/%s' % filename)
        # 获得预测结果
        result = clf.predict(vec)
        print('分类返回结果: %d\t真实结果: %d' % (result, class_num))
        if result != class_num:
            error += 1
    print('总共错了%d个数据' % error)
    print('错误率为%f%%' % (error / m_test * 100))


if __name__ == '__main__':
    head_writing_test()
