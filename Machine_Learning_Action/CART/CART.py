#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from prune import split_data, prune
from model import model_leaf, model_err


def load_data(filename):
    """Function description: Load data from file

    Args:
        filename: The name of the file

    Returns:
        data_mat: The mat of dataset
    """
    data_mat = []
    fr = open(filename)
    for line in fr.readlines():
        cur_line = line.strip().split('\t')
        flt_line = list(map(float, cur_line))
        data_mat.append(flt_line)
    return data_mat


def reg_leaf(dataset):
    """Generate leaf node, use the mean of values"""
    return np.mean(dataset[:, -1])


def reg_err(dataset):
    """Calculate the total variance: variance * number of samples"""
    return np.var(dataset[:, -1]) * np.shape(dataset)[0]


def choose_best_split(dataset, leaf=reg_leaf, error=reg_err, ops=(1, 4)):
    """Split the dataset and generate the corresponding leaf nodes in the best way

    ops=(1, 4) is very important because it determines the threshold value at
    which decision tree split stops

    Args:
        dataset: The loaded raw dataset
        leaf: Function of create leaf node
        error: Function of calculate error
        ops: Allowable error reduction value,
        the minimum number of sample to be split

    Returns:
        best_index: feature's index coordinate
        best_value: The best value of split
    """
    # The minimum error reduction value, the error after split is reduced less
    # than this difference, so there is no need to continue to split
    tol_s = ops[0]
    # If the minimum size < tol_n, don't continue to split
    tol_n = ops[1]
    if len(set(dataset[:, -1].T.tolist()[0])) == 1:
        return None, leaf
    m, n = np.shape(dataset)
    # Total variance without classification error
    s = error(dataset)
    best_s = float('inf')
    best_index = 0
    best_value = 0

    for feature in range(n - 1):
        # For each x column
        for value in set(dataset[:, feature].T.A.tolist()[0]):
            mat0, mat1 = split_data(dataset, feature, value)
            if np.shape(mat0)[0] < tol_n or np.shape(mat1)[0] < tol_n:
                continue
            new_s = error(mat0) + error(mat1)
            if new_s < best_s:
                best_index = feature
                best_value = value
                best_s = new_s

    if s - best_s < tol_s:
        return None, leaf(dataset)
    mat0, mat1 = split_data(dataset, best_index, best_value)
    if np.shape(mat0)[0] < tol_n or np.shape(mat1)[0] < tol_n:
        return None, leaf(dataset)
    return best_index, best_value


def create_tree(dataset, leaf=reg_leaf, error=reg_err, ops=(1, 4)):
    """Split the dataset
    generate the corresponding leaf nodes in the best way

    Assume dataset is numpy mat so we can array filtering

    Args:
        dataset: The loaded raw dataset
        leaf: Function of create leaf node
        error: Function of calculate error
        ops: Allowable error reduction value,
        the minimum number of sample to be split

    Returns:
        ret_tree: The last result of decision tree
    """
    # Choose the best split
    feature, value = choose_best_split(dataset, leaf, error, ops)
    if feature is None:
        return value
    ret_tree = {}
    ret_tree['index'] = feature
    ret_tree['value'] = value
    l_set, r_set = split_data(dataset, feature, value)
    ret_tree['left'] = create_tree(l_set, leaf, error, ops)
    ret_tree['right'] = create_tree(r_set, leaf, error, ops)
    return ret_tree


if __name__ == '__main__':
    MyDat = load_data('exp2.txt')
    MyMat = np.mat(MyDat)

    tree = create_tree(MyMat)
    print(tree)
    TestData = np.mat(load_data('ex2test.txt'))
    print(prune(tree, TestData))

    # tree = create_tree(MyMat, model_leaf, model_err, (1, 10))
    # print(tree)
