#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle


def store_tree(input_tree, filename):
    """函数说明: 存储决策树

    Args:
        input_tree: 已经生成的决策树
        filename: 决策树的存储文件名

    Returns:
        无
    """
    with open(filename, 'wb') as fw:
        pickle.dump(input_tree, fw)


if __name__ == '__main__':
    my_tree = {'有自己的房子': {0: {'有工作': {0: 'no', 1: 'yes'}}, 1: 'yes'}}
    store_tree(my_tree, 'classifier_storage.txt')
