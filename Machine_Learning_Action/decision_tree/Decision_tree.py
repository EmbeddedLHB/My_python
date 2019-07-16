#!/usr/bin/env python
# -*- coding: utf-8 -*-
from math import log
from operator import itemgetter


def calc_shannon_ent(dataset):
    """函数说明: 计算给定数据集的经验熵(香农熵)

    在此程序中，将分别计算每一个特征的每一个类别的香农熵，
    如: 计算 '青年' 分类下的 '放贷' 的香农熵

    Args:
        dataset: 数据集

    Returns:
        shannon_ent: 经验熵(香农熵)
    """
    num_entries = len(dataset)  # 数据集行数
    label_counts = {}
    for feat_vec in dataset:
        current_label = feat_vec[-1]  # 标签信息
        if current_label not in label_counts.keys():  # 添加新的标签
            label_counts[current_label] = 0
        label_counts[current_label] += 1  # label计数
    shannon_ent = 0.0  # 香农熵
    # 计算香农熵
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries  # 该label的概率
        shannon_ent -= prob * log(prob, 2)  # 利用公式计算香农熵
    return shannon_ent


def create_dataset():
    """函数说明: 创建测试数据集

    年龄: 0代表青年，1代表中年，2代表老年
    有工作: 0代表否，1代表是
    有自己的房子: 0代表否，1代表是
    信贷情况: 0代表一般，1代表好，2代表非常好
    类别（是否给贷款）: no代表否，yes代表是

    Returns:
        origin_dataset: 数据集
        labels: 分类属性
    """
    origin_dataset = [[0, 0, 0, 0, 'no'],
                      [0, 0, 0, 1, 'no'],
                      [0, 1, 0, 1, 'yes'],
                      [0, 1, 1, 0, 'yes'],
                      [0, 0, 0, 0, 'no'],
                      [1, 0, 0, 0, 'no'],
                      [1, 0, 0, 1, 'no'],
                      [1, 1, 1, 1, 'yes'],
                      [1, 0, 1, 2, 'yes'],
                      [1, 0, 1, 2, 'yes'],
                      [2, 0, 1, 2, 'yes'],
                      [2, 0, 1, 1, 'yes'],
                      [2, 1, 0, 1, 'yes'],
                      [2, 1, 0, 2, 'yes'],
                      [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']  # 分类属性
    return origin_dataset, labels


def split_dataset(dataset, axis, value):
    """函数说明: 按照给定特征划分数据集

    Args:
        dataset: 待划分的数据集
        axis: 划分数据集的特征
        value: 需要返回的特征的值

    Returns:
        ret_dataset: 划分后的数据集，如 '中年' 类别下的 '放贷'
    """
    ret_dataset = []
    for feat_vec in dataset:
        if feat_vec[axis] == value:
            # 分别将下标axis之前和之后的部分添加进了reduce_feat_vec，从而去除feat_vec[axis](特征值)
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis + 1:])
            ret_dataset.append(reduced_feat_vec)
    return ret_dataset


def choose_best_feature(dataset):
    """函数说明: 选择最优特征

    Args:
        dataset: 数据集

    Returns:
        best_feature: 信息增益最大特征的索引值

    """
    num_features = len(dataset[0]) - 1  # 特征数量
    base_entropy = calc_shannon_ent(dataset)  # 数据集的香农熵
    best_info_gain = 0.0  # 信息增益
    best_feature = -1  # 最优特征索引值
    for i in range(num_features):
        # dataset的第i个特征
        feat_list = [example[i] for example in dataset]
        unique_vals = set(feat_list)
        new_entropy = 0.0  # 经验条件熵
        for value in unique_vals:
            # dataset划分后的子集
            sub_dataset = split_dataset(dataset, i, value)
            prob = len(sub_dataset) / float(len(dataset))  # 子集概率
            # 根据公式计算经验条件熵
            new_entropy += prob * calc_shannon_ent(sub_dataset)
        info_gain = base_entropy - new_entropy  # 信息增益
        #
        # print("第%d个特征的增益为%.3f" % (i, info_gain))
        #
        if info_gain > best_info_gain:  # 更新信息增益
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_cnt(class_list):
    """函数说明: 统计class_list中出现次数最多的元素(类标签)

    Args:
        class_list: 类标签列表

    Returns:
        sorted_class_count[0][0]: 出现次数最多的元素
    """
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=itemgetter(1), reverse=True)  # 降序排序
    return sorted_class_count[0][0]


def create_tree(dataset, labels, feat_labels):
    """函数说明: 创建决策树

    Args:
        dataset: 训练数据集
        labels: 分类属性标签
        feat_labels: 存储选择的最优特征标签

    Returns:
        my_tree: 决策树
    """
    # 取分类标签(是否放贷: yes or no)
    class_list = [example[-1] for example in dataset]
    # 如果类别完全相同，则停止继续划分
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 遍历完所有特征时，返回出现次数最多的类标签
    if len(dataset[0]) == 1 or len(labels) == 0:
        return majority_cnt(class_list)
    best_feat = choose_best_feature(dataset)  # 最优特征
    best_feat_label = labels[best_feat]  # 最优特征标签
    feat_labels.append(best_feat_label)
    # 根据最优特征的标签生成树
    tree = {best_feat_label: {}}
    # 删除已经使用了特征的标签
    del(labels[best_feat])
    # 得到训练集中所有最优特征的属性值
    feat_values = [example[best_feat] for example in dataset]
    unique_vals = set(feat_values)  # 去掉重复属性值
    # 遍历特征，创建决策树
    for value in unique_vals:
        tree[best_feat_label][value] = create_tree(split_dataset(dataset, best_feat, value),
                                                   labels, feat_labels)
    return tree


if __name__ == '__main__':
    my_dataset, my_labels = create_dataset()
    my_feat_labels = []
    my_tree = create_tree(my_dataset, my_labels, my_feat_labels)
    print(my_tree)
