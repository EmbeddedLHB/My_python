#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random

import matplotlib.pyplot as plt
import numpy as np

from SVM import clip_alpha, load_data
from SVM_optimize import update_ek


class OptStruct:
    """类说明: 数据结构， 维护所有需要操作的值"""

    def __init__(self, data_mat, class_labels, c, toler, k_tup):
        # 数据矩阵
        self.X = data_mat
        # 数据标签
        self.label_mat = class_labels
        # 松弛变量
        self.C = c
        # 容错率
        self.toler = toler
        # 数据矩阵的行数
        self.m = np.shape(data_mat)[0]
        # 根据矩阵行数初始化alpha参数为0
        self.alphas = np.mat(np.zeros((self.m, 1)))
        # 初始化参数b为0
        self.b = 0
        # 根据矩阵行数初始化误差缓存， 第一列为是否有效的标志位， 第二列为实际的误差E的值
        self.ecache = np.mat(np.zeros((self.m, 2)))
        # 初始化核K
        self.K = np.mat(np.zeros((self.m, self.m)))
        # 计算所有数据的核K
        for i in range(self.m):
            self.K[:, i] = kernel_trans(self.X, self.X[i, :], k_tup)


def kernel_trans(x, a, k_tup):
    """函数说明: 通过核函数将数据转换到更高维的空间

    Args:
        x: 数据矩阵
        a: 单个数据的向量
        k_tup: 包含核函数信息的元组

    Returns:
        k: 计算的核k
    """
    m, n = np.shape(x)
    k = np.mat(np.zeros((m, 1)))
    # 线性核函数， 只进行内积
    if k_tup[0] == 'lin':
        k = x * a.T
    # 高斯核函数， 根据高斯核函数公式进行计算
    elif k_tup[0] == 'rbf':
        for j in range(m):
            delta_row = x[j, :] - a
            k[j] = delta_row * delta_row.T
        k = np.exp(k / (-1 * k_tup[1] ** 2))
    else:
        raise NameError('核函数无法识别')
    return k


def calculate_ek(op, k):
    """计算标号为k的数据误差
    与SVM_optimize.py中的函数有区别
    """
    fx_k = float(np.multiply(op.alphas, op.label_mat).T * op.K[:, k] + op.b)
    ek = fx_k - float(op.label_mat[k])
    return ek


def inner(i, op):
    """
    优化的SMO算法
    Parameters：
        i - 标号为i的数据的索引值
        oS - 数据结构
    Returns:
        1 - 有任意一对alpha值发生变化
        0 - 没有任意一对alpha值发生变化或变化太小
    """
    # 步骤1：计算误差Ei
    ei = calculate_ek(op, i)
    # 优化alpha,设定一定的容错率。
    if ((op.label_mat[i] * ei < -op.toler) and (op.alphas[i] < op.C)) or (
            (op.label_mat[i] * ei > op.toler) and (op.alphas[i] > 0)):
        # 使用内循环启发方式2选择alpha_j,并计算Ej
        j, ej = select_j(i, op, ei)
        # 保存更新前的alpha值
        alpha_i_old = op.alphas[i].copy()
        alpha_j_old = op.alphas[j].copy()
        # 步骤2：计算上下界L和H
        if op.label_mat[i] != op.label_mat[j]:
            high = min(op.C, op.C + op.alphas[j] - op.alphas[i])
            low = max(0, op.alphas[j] - op.alphas[i])
        else:
            high = min(op.C, op.alphas[j] + op.alphas[i])
            low = max(0, op.alphas[j] + op.alphas[i] - op.C)
        if low == high:
            print("low==high")
            return 0
        # 步骤3：计算eta
        eta = 2.0 * op.K[i, j] - op.K[i, i] - op.K[j, j]
        if eta >= 0:
            print("eta>=0")
            return 0
        # 步骤4：更新alpha_j
        op.alphas[j] -= op.label_mat[j] * (ei - ej) / eta
        # 步骤5：修剪alpha_j
        op.alphas[j] = clip_alpha(op.alphas[j], high, low)
        # 更新Ej至误差缓存
        update_ek(op, j)
        if abs(op.alphas[j] - alpha_j_old) < 0.00001:
            print("alpha_j变化太小")
            return 0
        # 步骤6：更新alpha_i
        op.alphas[i] += op.label_mat[j] * op.label_mat[i] * (alpha_j_old - op.alphas[j])
        # 更新Ei至误差缓存
        update_ek(op, i)
        # 步骤7：更新b_1和b_2
        b1 = op.b - ei - op.label_mat[i] * (op.alphas[i] - alpha_i_old) * op.K[i, i] - op.label_mat[j] * (
                op.alphas[j] - alpha_j_old) * op.K[i, j]
        b2 = op.b - ej - op.label_mat[i] * (op.alphas[i] - alpha_i_old) * op.K[i, j] - op.label_mat[j] * (
                op.alphas[j] - alpha_j_old) * op.K[j, j]
        # 步骤8：根据b_1和b_2更新b
        if (0 < op.alphas[i]) and (op.C > op.alphas[i]):
            op.b = b1
        elif (0 < op.alphas[j]) and (op.C > op.alphas[j]):
            op.b = b2
        else:
            op.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def select_j_rand(i, m):
    """
    函数说明:随机选择alpha_j的索引值

    Parameters:
        i - alpha_i的索引值
        m - alpha参数个数
    Returns:
        j - alpha_j的索引值
    """
    j = i  # 选择一个不等于i的j
    while j == i:
        j = int(random.uniform(0, m))
    return j


def select_j(i, op, ei):
    """
    内循环启发方式2
    Parameters：
        i - 标号为i的数据的索引值
        op - 数据结构
        ei - 标号为i的数据误差
    Returns:
        j, maxK - 标号为j或maxK的数据的索引值
        Ej - 标号为j的数据误差
    """
    max_k = -1
    max_delta_e = 0
    ej = 0  # 初始化
    op.ecache[i] = [1, ei]  # 根据Ei更新误差缓存
    valid_ecache_list = np.nonzero(op.ecache[:, 0].A)[0]  # 返回误差不为0的数据的索引值
    if (len(valid_ecache_list)) > 1:  # 有不为0的误差
        for k in valid_ecache_list:  # 遍历,找到最大的Ek
            if k == i:
                continue  # 不计算i,浪费时间
            ek = calculate_ek(op, k)  # 计算Ek
            delta_e = abs(ei - ek)  # 计算|ei-Ek|
            if delta_e > max_delta_e:  # 找到maxDeltaE
                max_k = k
                max_delta_e = delta_e
                ej = ek
        return max_k, ej  # 返回maxK,Ej
    else:  # 没有不为0的误差
        j = select_j_rand(i, op.m)  # 随机选择alpha_j的索引值
        ej = calculate_ek(op, j)  # 计算Ej
    return j, ej  # j,Ej


def smo_p(data_mat, class_labels, c, toler, max_iter, k_tup=('lin', 0)):
    """函数说明: 完整的线性SMO算法

    Args:
        data_mat: 数据矩阵
        class_labels: 数据标签
        c: 松弛变量
        toler: 容错率
        max_iter: 最大迭代次数
        k_tup: 包含核函数信息的元组

    Returns:
        op.b: SMO算法计算的b
        op.alphas: SMO算法计算的alpha
    """
    # 初始化数据结构
    op = OptStruct(np.mat(data_mat), np.mat(class_labels).transpose(), c, toler, k_tup)
    the_iter = 0
    alpha_pairs_changed = 0
    entire_set = True
    # 遍历整个数据集alpha也没有更新或者超过最大迭代次数， 则退出循环
    while the_iter < max_iter and (alpha_pairs_changed > 0 or entire_set):
        alpha_pairs_changed = 0
        # 遍历边界值
        if entire_set:
            for i in range(op.m):
                alpha_pairs_changed += inner(i, op)
                print('全样本遍历: 第%d次迭代 样本: %d, alpha优化次数: %d' % (the_iter, i, alpha_pairs_changed))
            the_iter += 1
        # 遍历非边界值
        else:
            # 遍历不在边界0和c的alpha
            non_bound = np.nonzero((op.alphas.A > 0) * (op.alphas.A < c))[0]
            for i in non_bound:
                alpha_pairs_changed += inner(i, op)
                print('非边界遍历: 第%d次迭代 样本: %d, alpha优化次数: %d' % (the_iter, i, alpha_pairs_changed))
            the_iter += 1
        # 遍历一次后改为非边界遍历
        if entire_set:
            entire_set = False
        # 如果alpha没有更新， 计算全样本遍历
        elif alpha_pairs_changed == 0:
            entire_set = True
        print('迭代次数: %d' % the_iter)
    return op.b, op.alphas


def test_rbf(k1=1.3):
    """函数说明: 测试函数

    Args:
        k1: 使用高斯核函数的时候表示到达率

    Returns:
        无
    """
    # 加载训练集并根据训练集得到b和alphas
    data_arr, label_arr = load_data('testSetRBF.txt')
    b, alphas = smo_p(data_arr, label_arr, 200, 0.0001, 100, ('rbf', k1))
    data_mat = np.mat(data_arr)
    label_mat = np.mat(label_arr).transpose()
    # 获得支持向量
    sv_ind = np.nonzero(alphas.A > 0)[0]
    svs = data_mat[sv_ind]
    label_sv = label_mat[sv_ind]
    print('支持向量个数: %d' % np.shape(svs)[0])

    m, n = np.shape(data_mat)
    error = 0
    for i in range(m):
        # 计算各个点的核
        kernel = kernel_trans(svs, data_mat[i, :], ('rbf', k1))
        # 根据支持向量的点， 计算超平面， 返回预测结果
        predict = kernel.T * np.multiply(label_sv, alphas[sv_ind]) + b
        # 返回数组中各元素的正负符号， 用1和-1表示， 并统计错误个数
        if np.sign(predict) != np.sign(label_arr[i]):
            error += 1
    print('训练集错误率: %.2f%%' % (float(error) / m * 100))

    # 加载测试集
    data_arr, label_arr = load_data('testSetRBF2.txt')
    data_mat = np.mat(data_arr)
    m, n = np.shape(data_mat)
    error = 0
    for i in range(m):
        # 计算各个点的核
        kernel = kernel_trans(svs, data_mat[i, :], ('rbf', k1))
        # 根据支持向量的点， 计算超平面， 返回预测结果
        predict = kernel.T * np.multiply(label_sv, alphas[sv_ind]) + b
        # 返回数组中各元素的正负符号， 用1和-1表示， 并统计错误个数
        if np.sign(predict) != np.sign(label_arr[i]):
            error += 1
    print('测试集错误率: %.2f%%' % (float(error) / m * 100))


def show_data(data_mat, label_mat):
    """函数说明: 数据可视化

    Args:
        data_mat: 数据矩阵
        label_mat: 数据标签

    Returns:
        无
    """
    # 正负样本
    data_plus = []
    data_minus = []
    # 将信息添加到正负样本
    for i in range(len(data_mat)):
        if label_mat[i] > 0:
            data_plus.append(data_mat[i])
        else:
            data_minus.append(data_mat[i])
    # 转化为numpy矩阵
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    # 绘制散点图
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1])
    plt.show()


if __name__ == '__main__':
    test_rbf()
    _data_arr, _label_arr = load_data('testSetRBF.txt')
    show_data(_data_arr, _label_arr)
