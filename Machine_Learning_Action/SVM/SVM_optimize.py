#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from SVM import clip_alpha, load_data, select_j_rand, show_classifier


class OptStruct:
    """类说明: 数据结构， 维护所有需要操作的值"""

    def __init__(self, data_mat, class_labels, c, toler):
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


def calculate_ek(op, k):
    """函数说明: 计算标号为k的数据的误差ek"""
    fx_k = float(np.multiply(op.alphas, op.label_mat).T * (op.X * op.X[k, :].T) + op.b)
    ek = fx_k - float(op.label_mat[k])
    return ek


def select_j(i, op, ei):
    """函数说明: 内循环启发方式2

    Args:
        i: 标号为i的数据的索引值
        op: 数据结构
        ei: 标号为i的数据误差

    Returns:
        j, max_k: 标号为j或max_k的数据的索引值
        ej: 标号为j的数据误差
    """
    # 初始化
    max_k = -1
    max_delta_e = 0
    ej = 0
    # 根据ei更新误差缓存
    op.ecache[i] = [1, ei]
    # 返回误差不为0的数据索引值
    valid_ecache_list = np.nonzero(op.ecache[:, 0].A)[0]
    # 有不为0的误差
    if len(valid_ecache_list) > 1:
        # 遍历， 找到最大的ek
        for k in valid_ecache_list:
            if k == i:
                continue
            # 计算ek和|ei - ek|
            ek = calculate_ek(op, k)
            delta_e = abs(ei - ek)
            # 找到max_k
            if delta_e > max_delta_e:
                max_k = k
                max_delta_e = delta_e
                ej = ek
        return max_k, ej
    # 没有不为0的误差
    else:
        # 分别计算j和ej
        j = select_j_rand(i, op.m)
        ej = calculate_ek(op, j)
    return j, ej


def update_ek(op, k):
    """函数说明: 计算标号为k的数据的误差ek， 并更新误差缓存"""
    ek = calculate_ek(op, k)
    op.ecache[k] = [1, ek]


def inner(i, op):
    """函数说明: 优化的SMO算法

    Args:
        i: 标号为i的数据的索引值
        op: 数据结构

    Returns:
        1: 有任意一对alpha值发生变化
        0: 没有任意一对alpha值发生变化
    """
    # 步骤1: 计算误差ei
    ei = calculate_ek(op, i)
    # 优化alpha， 设定一定的容错率
    if (op.label_mat[i] * ei < -op.toler and op.alphas[i] < op.C) or (
            op.label_mat[i] * ei > op.toler and op.alphas[i] > 0):
        # 使用内循环启发方式2喧杂alpha_j， 并计算ej
        j, ej = select_j(i, op, ei)
        # 保存更新前的alpha值， 使用浅拷贝
        alpha_i_old = op.alphas[i].copy()
        alpha_j_old = op.alphas[j].copy()
        # 步骤2: 计算上下界high和low
        if op.label_mat[i] != op.label_mat[j]:
            high = min(op.C, op.C + op.alphas[j] - op.alphas[i])
            low = max(0, op.alphas[j] - op.alphas[i])
        else:
            high = min(op.C, op.alphas[j] + op.alphas[i])
            low = max(0, op.alphas[j] + op.alphas[i] - op.C)
        if low == high:
            print('low == high')
            return 0

        # 步骤3: 计算eta
        eta = 2.0 * op.X[i, :] * op.X[j, :].T - op.X[i, :] * op.X[i, :].T - op.X[j, :] * op.X[j, :].T
        if eta >= 0:
            print('eta >= 0')
            return 0
        # 步骤4: 更新alpha_j
        op.alphas[j] -= op.label_mat[j] * (ei - ej) / eta
        # 步骤5: 修剪alpha_j
        op.alphas[j] = clip_alpha(op.alphas[j], high, low)
        # 更新ej至误差缓存
        update_ek(op, j)
        if abs(op.alphas[j] - alpha_j_old) < 0.00001:
            # print('alpha_j变化太小')
            return 0
        # 步骤6: 更新alpha_i
        op.alphas[i] += op.label_mat[j] * op.label_mat[i] * (alpha_j_old - op.alphas[j])
        # 更新ei至误差缓存
        update_ek(op, i)

        # 步骤7: 更新b_1和b_2
        b1 = (op.b - ei - op.label_mat[i] * (op.alphas[i] - alpha_i_old) * op.X[i, :] * op.X[i, :].T - op.label_mat[j]
              * (op.alphas[j] - alpha_j_old) * op.X[i, :] * op.X[j, :].T)
        b2 = (op.b - ej - op.label_mat[i] * (op.alphas[i] - alpha_i_old) * op.X[i, :] * op.X[j, :].T - op.label_mat[j]
              * (op.alphas[j] - alpha_j_old) * op.X[j, :] * op.X[j, :].T)
        # 步骤8: 根据b_1和b_2更新b
        if 0 < op.alphas[i] < op.C:
            op.b = b1
        elif 0 < op.alphas[j] < op.C:
            op.b = b2
        else:
            op.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smo_p(data_mat, class_labels, c, toler, max_iter):
    """函数说明: 完整的先行SMO算法

    Args:
        data_mat: 数据矩阵
        class_labels: 数据标签
        c: 松弛变量
        toler: 容错率
        max_iter: 最大迭代次数

    Returns:
        op.b: SMO算法计算的b
        op.alphas: SMO算法计算的alpha
    """
    # 初始化数据结构
    op = OptStruct(np.mat(data_mat), np.mat(class_labels).transpose(), c, toler)
    the_iter = 0
    entire_set = True
    alpha_pairs_changed = 0
    # 遍历整个数据集alpha也没有更新或者超过最大迭代次数， 则退出循环
    while the_iter < max_iter and (alpha_pairs_changed > 0 or entire_set):
        alpha_pairs_changed = 0
        if entire_set:
            for i in range(op.m):
                alpha_pairs_changed += inner(i, op)
                print('全样本遍历: 第%d次迭代 样本: %d, alpha优化次数: %d' % (the_iter, i, alpha_pairs_changed))
            the_iter += 1
        # 遍历非边界值
        else:
            # 遍历不再边界0和c的alpha
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


def calculate_w(alphas, data_arr, class_labels):
    """函数说明: 计算w

    Args:
        alphas: alphas值
        data_arr: 数据矩阵
        class_labels: 数据标签

    Returns:
        w: 计算得到的w
    """
    x = np.mat(data_arr)
    label_mat = np.mat(class_labels).transpose()
    m, n = np.shape(x)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * label_mat[i], x[i, :].T)
    return w


if __name__ == '__main__':
    my_data, my_labels = load_data('testSet.txt')
    _b, _alphas = smo_p(my_data, my_labels, 0.6, 0.001, 40)
    _w = calculate_w(_alphas, my_data, my_labels)
    show_classifier(my_data, my_labels, _w, _b, _alphas)
