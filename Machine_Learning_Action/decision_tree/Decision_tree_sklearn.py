#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pydotplus
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.preprocessing import LabelEncoder


def note():
    """
    在文件中:
        第一列: 年龄
        第二列: 症状
        第三列: 是否散光
        第四列: 眼泪数量
        第五列: 分类结果

    Returns:
        无
    """
    pass


if __name__ == '__main__':
    with open('lenses.txt', 'r') as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    # 提取每组数据的类别，保存在列表里
    lenses_target = []
    for each in lenses:
        lenses_target.append(each[-1])

    lenses_labels = ['age', 'prescript', 'astigmatic', 'tearRate']  # 特征标签
    # 保存lenses数据的临时列表
    lenses_list = []
    # 保存lenses数据的字典，用于生成pandas
    lenses_dict = {}
    for each_label in lenses_labels:
        for each in lenses:
            lenses_list.append(each[lenses_labels.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lenses_list = []
    # print(lenses_dict)  # 打印字典信息
    # 生成pandas.DataFrame
    lenses_pd = pd.DataFrame(lenses_dict)
    # print(lenses_pd)  # 打印pandas.DataFrame
    # 创建LabelEncoder()对象，用于序列化
    le = LabelEncoder()
    # 对每一列进行序列化
    for col in lenses_pd.columns:
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    # print(lenses_pd)  # 打印编码信息

    # 创建DecisionTreeClassifier()类
    clf = tree.DecisionTreeClassifier(max_depth=4)
    # 使用数据，构建决策树
    clf = clf.fit(lenses_pd.values.tolist(), lenses_target)
    dot_data = StringIO()
    # 绘制决策树
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=lenses_pd.keys(),
                         class_names=clf.classes_,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # 将绘制好的决策树保存为pdf文件
    graph.write_pdf("tree.pdf")
    print(clf.predict([[1, 1, 1, 0]]))
