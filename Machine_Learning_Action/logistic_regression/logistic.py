#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from scipy.special import expit


def load_data():
    """函数说明: 加载数据

    Returns:
        data_mat: 数据列表
        label_mat: 标签列表
    """
    # 数据列表
    data_mat = []
    # 标签列表
    label_mat = []
    with open('testSet.txt', 'r') as fp:
        for line in fp.readlines():
            line_arr = line.strip().split()
            # 将文件里的数据添加到列表
            data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
            label_mat.append(int(line_arr[2]))
    return data_mat, label_mat


def sigmoid(inx):
    # return 1.0 / (1 + np.exp(-inx))
    return expit(inx)


def grad_ascent(data_mat_in, class_labels):
    """函数说明: 梯度上升算法

    Args:
        data_mat_in: 数据集(100*3数组)
        class_labels: 数据标签(100*1数组)

    Returns:
        weights.getA(): 求得的权重数组
        weights_arr: 回归系数的数组
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
        weights_arr = np.append(weights_arr, weights)
    weights_arr = weights_arr.reshape(max_cycles, n)
    # 返回权重数组
    return weights.getA(), weights_arr


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
    # 存储每次更新的回归系数
    weights_arr = np.array([])
    # i是迭代次数，j是样本点的下标
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
            # 添加回归系数到数组中
            weights_arr = np.append(weights_arr, weights)
            # 删除已经使用的样本
            del (data_index[rand_index])
    weights_arr = weights_arr.reshape(num * m, n)
    return weights, weights_arr


def plot_weights(weights_arr1, weights_arr2):
    """函数说明: 绘制回归系数与迭代次数的关系

    Args:
        weights_arr1: 回归系数数组1
        weights_arr2: 回归系数数组2
    Returns:
        无
    """
    font = FontProperties(fname=r'C:/windows/fonts/simsun.ttc', size=14)
    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, figsize=(20, 10))

    # 绘制梯度上升算法回归系数与迭代次数关系
    x1 = np.arange(0, len(weights_arr1), 1)
    # 绘制w0与迭代次数的关系
    axs[0][0].plot(x1, weights_arr1[:, 0])
    axs0_title_text = axs[0][0].set_title(u'改进的随机梯度上升算法: 回归系数与迭代系数关系', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'W0', Fontproperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][0].plot(x1, weights_arr1[:, 1])
    axs1_ylabel_text = axs[1][0].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][0].plot(x1, weights_arr1[:, 2])
    axs2_xlabel_text = axs[2][0].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[2][0].set_ylabel(u'W2', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    # 绘制改进的随机梯度上升算法的回归系数与迭代系数关系
    x2 = np.arange(0, len(weights_arr2), 1)
    # 绘制w0与迭代次数的关系
    axs[0][1].plot(x2, weights_arr2[:, 0])
    axs0_title_text = axs[0][1].set_title(u'梯度上升算法: 回归系数与迭代系数关系', FontProperties=font)
    axs0_ylabel_text = axs[0][1].set_ylabel(u'W0', Fontproperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][1].plot(x2, weights_arr2[:, 1])
    axs1_ylabel_text = axs[1][1].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][1].plot(x2, weights_arr2[:, 2])
    axs2_xlabel_text = axs[2][1].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[2][1].set_ylabel(u'W2', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    plt.show()


if __name__ == '__main__':
    data_mat1, label_mat1 = load_data()
    weights1, weights_array1 = stochastic_grad_ascent(np.array(data_mat1), label_mat1)
    weights2, weights_array2 = grad_ascent(data_mat1, label_mat1)
    plot_weights(weights_array1, weights_array2)
