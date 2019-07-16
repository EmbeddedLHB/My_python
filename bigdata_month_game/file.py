#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from snownlp import SnowNLP
from tqdm import tqdm


def judge(x):
    """函数说明: 判断概率的函数"""
    if x > 0.5:
        return 1
    return 0


def pre_process(train_file):
    """函数说明: 对训练集进行预处理，分别生成积极和消极的txt文件

    Args:
        train_file: 训练集地址

    Returns:
        无
    """
    df = pd.read_csv(train_file)
    # 分别读取正负样本的评论
    pos = df[df.label == 1]['review']
    neg = df[df.label == 0]['review']
    # 保存评论
    pos.to_csv('pos.txt', sep='\n', index=False, header=False)
    neg.to_csv('neg.txt', sep='\n', index=False, header=False)


def read_analysis(test_set, result_set):
    """函数说明: 使用snownlp进行情感分析并保存结果集合

    Args:
        test_set: 测试集
        result_set: 分类结果

    Returns:
        无
    """
    i = 1
    # 初始化结果的DataFrame数据结构
    result = pd.DataFrame({'ID': [], 'label': []})
    # 读取测试集中的数据
    test = pd.read_csv(test_set)['review']
    for line in tqdm(test):
        result = result.append({'ID': i, 'label': judge(SnowNLP(line).sentiments)}, ignore_index=True)
        i += 1
    # 保存为csv文件
    result.to_csv(result_set, index=False)


if __name__ == '__main__':
    my_train_file = 'TrainSet.csv'
    my_test_set = 'TestSet.csv'
    my_result_set = 'my_result.csv'
    pre_process(my_train_file)
    read_analysis(my_test_set, my_result_set)
