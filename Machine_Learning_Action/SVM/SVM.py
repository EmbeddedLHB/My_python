#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random

import matplotlib.pyplot as plt
import numpy as np


def load_data(filename):
    """函数说明: 读取数据

    Args:
        filename: 文件名

    Returns:
        data_mat: 数据矩阵
        label_mat: 数据标签
    """
    data_mat = []
    label_mat = []
    with open(filename, 'r') as fp:
        for line in fp.readlines():
            # 逐行读取，滤除空格等
            line_arr = line.strip().split('\t')
            # 分别添加数据和标签
            data_mat.append([float(line_arr[0]), float(line_arr[1])])
            label_mat.append(float(line_arr[2]))
    return data_mat, label_mat


def select_j_rand(i, m):
    """函数说明: 得到一个(0, m)之间，不等于i的随机值"""
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    # 返回一个不等于i的小于m的值
    return j


def clip_alpha(alpha, high, low):
    """函数说明: 修剪alpha， 使alpha取值在high和low之间"""
    if alpha > high:
        alpha = high
    if alpha < low:
        alpha = low
    return alpha


def smo_simple(data_mat_in, class_labels, c, toler, max_iter):
    """函数说明: 简化版SMO算法

    Args:
        data_mat_in: 数据矩阵
        class_labels: 数据标签
        c: 惩罚参数
        toler: 松弛变量
        max_iter: 最大迭代次数

    Returns:
        无
    """
    # 转换为numpy的mat存储
    data_mat = np.mat(data_mat_in)
    label_mat = np.mat(class_labels).transpose()
    # 初始化b参数
    b = 0
    m, n = np.shape(data_mat)
    # 初始化alpha参数
    alphas = np.mat(np.zeros((m, 1)))
    # 初始化迭代次数
    num_iter = 0

    # 最多迭代max_iter次
    while num_iter < max_iter:
        alpha_pairs_changed = 0
        for i in range(m):
            # 步骤1: 计算误差Ei
            fxi = float(np.multiply(alphas, label_mat).T * (data_mat * data_mat[i, :].T)) + b
            ei = fxi - float(label_mat[i])
            # 优化alpha，设定一定的容错率
            if (label_mat[i] * ei < -toler and alphas[i] < c) or (label_mat[i] * ei > toler and alphas[i] > 0):
                # 随机选择另一个与alpha_i成对优化的alpha_j
                j = select_j_rand(i, m)
                # 步骤1: 计算误差Ej
                fxj = float(np.multiply(alphas, label_mat).T * (data_mat * data_mat[j, :].T)) + b
                ej = fxj - float(label_mat[j])
                # 保存更新前的alpha值，使用深拷贝
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()

                # 步骤2: 计算上下界L和H
                if label_mat[i] != label_mat[j]:
                    low = max(0, alphas[j] - alphas[i])
                    high = min(c, c + alphas[j] - alphas[i])
                else:
                    low = max(0, alphas[j] + alphas[i] - c)
                    high = min(c, alphas[j] + alphas[i])
                if low == high:
                    print('low==high')
                    continue

                # 步骤3: 计算eta
                eta = (2.0 * data_mat[i, :] * data_mat[j, :].T - data_mat[i, :] * data_mat[i, :].T
                       - data_mat[j, :] * data_mat[j, :].T)
                if eta >= 0:
                    print('eta>=0')
                    continue
                # 步骤4: 更新alpha_j
                alphas[j] -= label_mat[j] * (ei - ej) / eta
                # 步骤5: 修剪alpha_j
                alphas[j] = clip_alpha(alphas[j], high, low)
                if abs(alphas[j] - alpha_j_old) < 0.00001:
                    # print('alpha_j变化太小')
                    continue

                # 步骤6: 更新alpha_i
                alphas[i] += label_mat[j] * label_mat[i] * (alpha_j_old - alphas[j])
                # 步骤7: 更新b1和b2
                b1 = (b - ei - label_mat[i] * (alphas[i] - alpha_i_old) * data_mat[i, :] * data_mat[i, :].T
                      - label_mat[j] * (alphas[j] - alpha_j_old) * data_mat[i, :] * data_mat[j, :].T)
                b2 = (b - ej - label_mat[i] * (alphas[i] - alpha_i_old) * data_mat[i, :] * data_mat[j, :].T
                      - label_mat[j] * (alphas[j] - alpha_j_old) * data_mat[j, :] * data_mat[j, :].T)
                # 步骤8: 根据b1和b2更新b
                if 0 < alphas[i] < c:
                    b = b1
                elif 0 < alphas[j] < c:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                # 统计优化次数
                alpha_pairs_changed += 1
                # 打印统计信息
                print('第%d次迭代 样本%d， alpha优化次数: %d' % (num_iter, i, alpha_pairs_changed))
        # 更新迭代次数
        if alpha_pairs_changed == 0:
            num_iter += 1
        else:
            num_iter = 0
        print('迭代次数: %d' % num_iter)
    return b, alphas


def show_classifier(data_mat, label_mat, w, b, alphas):
    """函数说明: 分类结果可视化

    Args:
        data_mat: 数据矩阵
        label_mat: 标签矩阵
        w: 直线法向量
        b: 超平面参数中的常数值
        alphas: 超平面参数中的其他值

    Returns:
        无
    """
    # 正样本和负样本
    data_plus = []
    data_minus = []
    # 将数据分类
    for i in range(len(data_mat)):
        if label_mat[i] > 0:
            data_plus.append(data_mat[i])
        else:
            data_minus.append(data_mat[i])
    # 将数据列表numpy化
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)

    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7)
    x1 = max(data_mat)[0]
    x2 = min(data_mat)[0]
    a1, a2 = w

    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    plt.plot([x1, x2], [y1, y2])

    # 找出支持向量点
    for i, alpha in enumerate(alphas):
        if alpha > 0:
            x, y = data_mat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()


def get_w(data_mat, label_mat, alphas):
    """函数说明: 计算w

    Args:
        data_mat: 数据矩阵
        label_mat: 数据标签
        alphas: alpha值

    Returns:
        w.tolist(): 将numpy类型的w转化为列表
    """
    alphas, data_mat, label_mat = np.array(alphas), np.array(data_mat), np.array(label_mat)
    w = np.dot((np.tile(label_mat.reshape(1, -1).T, (1, 2)) * data_mat).T, alphas)
    return w.tolist()


if __name__ == '__main__':
    # 加载数据
    data_mat1, label_mat1 = load_data('testSet.txt')
    # 得到超平面的参数
    b_, alphas1 = smo_simple(data_mat1, label_mat1, 0.6, 0.001, 40)
    # 得到w的值
    w1 = get_w(data_mat1, label_mat1, alphas1)
    show_classifier(data_mat1, label_mat1, w1, b_, alphas1)
