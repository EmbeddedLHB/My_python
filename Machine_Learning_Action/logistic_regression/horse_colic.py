#!usr/bin/env python
# -*- coding: utf-8 -*-
import random
import numpy as np
from Logistic import sigmoid


def classify_vector(inx, weights):
    """函数说明: 分类函数

        inx: <type: 'array'>
    """
    prob = sigmoid(sum(inx * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def grad_ascent(data_mat_in, class_labels):
    """函数说明: 梯度上升算法

    Args:
        data_mat_in: 数据集(100*3数组)
        class_labels: 数据标签(100*1数组)

    Returns:
        weights.getA(): 求得的权重数组
    """
    # 转换为numpy.mat
    data_mat = np.mat(data_mat_in)
    # 转换为numpy.mat并转置
    label_mat = np.mat(class_labels).transpose()
    m, n = np.shape(data_mat)
    # 移动步长，也就是学习速率，控制更新的幅度
    alpha = 0.01
    # 最大迭代次数
    max_cycles = 500
    # 初始化权重数组
    weights = np.ones((n, 1))
    # 存储每次更新的回归系数
    weights_arr = np.array([])
    for i in range(max_cycles):
        # 梯度上升矢量化公式
        h = sigmoid(data_mat * weights)
        error = label_mat - h
        weights = weights + alpha * data_mat.transpose() * error
    # 返回权重数组
    return weights.getA()


def stochastic_grad_ascent(data_mat, class_labels, num=150):
    """函数说明: 改进的随机梯度上升算法

    Args:
        data_mat: 数据数组
        class_labels: 数据标签
        num: 迭代次数

    Returns:
        weights: 求得的回归系数数组(最优参数)
    """
    m, n = np.shape(data_mat)
    # 参数初始化
    weights = np.ones(n)
    for i in range(num):
        data_index = list(range(m))
        for j in range(m):
            # 降低alpha的大小，每次减小1/(i+j)
            alpha = 4 / (1.0 + i + j) + 0.01
            # 随机选取样本
            rand_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_mat[rand_index] * weights))
            # 计算误差
            error = class_labels[rand_index] - h
            # 更新回归系数
            weights = weights + alpha * error * data_mat[rand_index]
            # 删除已经使用的样本
            del (data_index[rand_index])
    return weights


def colic_test():
    """函数说明: 使用python写的logistic分类器做预测

    Returns:
        无
    """
    training_set = []
    training_labels = []
    with open('horseColicTraining.txt', 'r') as fp_train:
        for line in fp_train.readlines():
            curr_line = line.strip().split('\t')
            line_arr = []
            for i in range(len(curr_line) - 1):
                line_arr.append(float(curr_line[i]))
            training_set.append(line_arr)
            training_labels.append(float(curr_line[-1]))
    train_weights = grad_ascent(training_set, training_labels)
    num_test_vec = 0.0
    error_count = 0.0
    with open('horseColicTest.txt', 'r') as fp_test:
        for line in fp_test.readlines():
            num_test_vec += 1.0
            curr_line = line.strip().split('\t')
            line_arr = []
            for i in range(len(curr_line) - 1):
                line_arr.append(float(curr_line[i]))
            if int(classify_vector(np.array(line_arr), train_weights[:, 0])) != int(curr_line[-1]):
                error_count += 1
    error_rate = (error_count / num_test_vec) * 100
    print('测试集错误率为： %.2f%%' % error_rate)


if __name__ == '__main__':
    colic_test()
