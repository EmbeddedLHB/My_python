#!/usr/bin/env python
# -*- coding: utf-8 -*-
from Decision_tree import create_dataset, create_tree


def classify(input_tree, feat_labels, test_vec):
    """函数说明: 使用决策树分类

    Args:
        input_tree: 已经生成的决策树
        feat_labels: 存储选择的最优特征标签
        test_vec: 测试数据列表，顺序对应最优特征标签

    Returns:
        class_label: 分类结果
    """
    class_label = 0
    first_str = next(iter(input_tree))  # 决策树结点
    second_dict = input_tree[first_str]  # 下一个字典
    feat_index = feat_labels.index(first_str)
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]
    return class_label


if __name__ == '__main__':
    dataset, labels = create_dataset()
    my_feat_labels = []
    my_tree = create_tree(dataset, labels, my_feat_labels)
    my_test_vec = [0, 1]
    result = classify(my_tree, my_feat_labels, my_test_vec)
    if result == 'yes':
        print("放贷")
    if result == 'no':
        print("不放贷")
