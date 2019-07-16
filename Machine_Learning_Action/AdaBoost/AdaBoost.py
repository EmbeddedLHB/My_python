#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


def load_data():
    """函数说明: 创建单层决策树的数据集

    Returns:
        data_mat: 数据矩阵
        class_labels: 数据标签
    """
    data_mat = np.mat([[1.0, 2.1],
                       [1.5, 1.6],
                       [1.3, 1.0],
                       [1.0, 1.0],
                       [2.0, 1.0]])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_mat, class_labels


def show_data(data_mat, label_mat):
    """函数说明: 数据可视化

    Args:
        data_mat: 数据矩阵
        label_mat: 数据标签

    Returns:
        无
    """
    # 正样本和负样本
    plus = []
    minus = []
    for i in range(len(data_mat)):
        if label_mat[i] > 0:
            plus.append(data_mat[i])
        else:
            minus.append(data_mat[i])
    # 转换为numpy矩阵
    plus_np = np.array(plus)
    minus_np = np.array(minus)
    plt.scatter(np.transpose(plus_np)[0], np.transpose(plus_np)[1])
    plt.scatter(np.transpose(minus_np)[0], np.transpose(minus_np)[1])
    plt.show()


def stump_classify(data_mat, dim, threshold, mark):
    """函数说明: 单层决策树分类函数

    Args:
        data_mat: 数据矩阵
        dim: 第dim列
        threshold: 阈值
        mark: 标志

    Returns:
        ret_array: 分类结果
    """
    # 初始化ret_array为1
    ret_array = np.ones((np.shape(data_mat)[0], 1))
    # 将小于等于阈值的部分赋值为-1
    if mark == 'lt':
        ret_array[data_mat[:, dim] <= threshold] = -1.0
    # 将大于阈值的部分赋值为-1
    else:
        ret_array[data_mat[:, dim] > threshold] = -1.0
    return ret_array


def build_stump(data_arr, class_labels, d):
    """函数说明: 找到数据集上最佳的单层决策树

    Args:
        data_arr: 数据矩阵
        class_labels: 数据标签
        d: 样本权重

    Returns:
        best_stump: 最佳单层决策树信息
        min_error: 最小误差
        best_result: 最佳的分类结果
    """
    data_mat = np.mat(data_arr)
    label_mat = np.mat(class_labels).T
    m, n = np.shape(data_mat)
    num_steps = 10.0
    # 初始化最佳单层决策树和最佳分类结果
    best_stump = {}
    best_result = np.mat(np.zeros((m, 1)))
    # 初始化最小误差为正无穷
    min_error = float('inf')
    # 对于每一个特征
    for i in range(n):
        # 找到特征中最小值和最大值
        range_min = data_mat[:, i].min()
        range_max = data_mat[:, i].max()
        # 迭代的步长
        step_size = (range_max - range_min) / num_steps

        # 这里之所以从-1开始迭代，是因为DT分类时是">"而不是">="
        for j in range(-1, int(num_steps) + 1):
            # 遍历小于和大于的情况
            for mark in ['lt', 'gt']:
                # 计算阈值
                threshold = range_min + float(j) * step_size
                # 计算分类结果
                predict_val = stump_classify(data_mat, i, threshold, mark)
                # 初始化误差矩阵
                error_mat = np.mat(np.ones((m, 1)))
                # 分类正确的， 赋值为0
                error_mat[predict_val == label_mat] = 0
                # 计算误差
                error = d.T * error_mat
                # print('split: dim %d, thresh %.2f, mark: %s, the error is %.3f' %
                #       (i, thresh_val, mark, error))

                # 找到误差最小的分类方式
                if error < min_error:
                    min_error = error
                    best_result = predict_val.copy()
                    best_stump['dim'] = i
                    best_stump['threshold'] = threshold
                    best_stump['mark'] = mark
    return best_stump, min_error, best_result


def adaboost_train_ds(data_arr, class_labels, num=40):
    """函数说明: 使用AdaBoost算法提升分类器性能
    其中d是一个概率分布向量，所有的元素都会被初始化为1/m

    Args:
        data_arr: 数据矩阵
        class_labels: 标签矩阵
        num: 迭代次数，也是弱分类器的数量

    Returns:
        weak_class_arr: 弱分类器的矩阵表示
        agg_class_res: 类别估计累积值
    """
    weak_class_arr = []
    m = np.shape(data_arr)[0]
    # 初始化权重
    d = np.mat(np.ones((m, 1)) / m)
    agg_class_res = np.mat(np.zeros((m, 1)))
    # 对于每一个弱分类器
    for i in range(num):
        # 当前的最佳单层决策树、错误率、分类标签
        best_stump, error, class_res = build_stump(data_arr, class_labels, d)
        # print('D:', d.T)
        # 计算弱学习算法权重alpha，其中error不等于0
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        best_stump['alpha'] = alpha
        # 存储单层决策树
        weak_class_arr.append(best_stump)
        # print('class_result:', class_res.T)
        # 计算e的指数项
        exp_on = -1 * alpha * np.multiply(np.mat(class_labels).T, class_res)
        d = np.multiply(d, np.exp(exp_on))
        # 根据样本权重公式，更新样本权重
        d = d / d.sum()

        # 计算adaBoost误差，当误差为0的时候，退出循环
        agg_class_res += alpha * class_res
        # print('agg class result:', agg_class_res.T)
        # 计算误差
        agg_errors = np.multiply(np.sign(agg_class_res) != np.mat(class_labels).T, np.ones((m, 1)))
        error_rate = agg_errors.sum() / m
        print('total error:', error_rate, '\n')
        # 误差为0，退出循环
        if error_rate == 0.0:
            break
    return weak_class_arr, agg_class_res


def adaboost_classify(dat_to_class, classifier):
    """函数说明: AdaBoost分类函数

    Args:
        dat_to_class: 待分类样例
        classifier: 训练好的分类器

    Returns:
        分类结果
    """
    data_mat = np.mat(dat_to_class)
    m = np.shape(data_mat)[0]
    agg_class_res = np.mat(np.zeros((m, 1)))
    # 遍历所有分类器，进行分类
    for i in range(len(classifier)):
        class_res = stump_classify(data_mat, classifier[i]['dim'], classifier[i]['threshold'],
                                   classifier[i]['mark'])
        agg_class_res += classifier[i]['alpha'] * class_res
        # print('agg class res:\n', agg_class_res)
    return np.sign(agg_class_res)


if __name__ == '__main__':
    dataArr, classLabels = load_data()
    weakClassArr, aggClassRes = adaboost_train_ds(dataArr, classLabels)
    print(adaboost_classify([[0, 0], [5, 5]], weakClassArr))
