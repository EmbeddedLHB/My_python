#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def split_data(dataset, feature_index, value):
    """Binary dataset according to the value of the feature column
    Given a feature and eigenvalue, the function splits the above dataset into
    two subsets by array filtering and returns

    Args:
        dataset: The dataset
        feature_index: Feature column waiting to be split
        value: The value to be compared by the feature column

    Returns:
        mat0: <= value's dataset is on the left
        mat1: > value's dataset is on the right
    """
    mat0 = dataset[np.nonzero(dataset[:, feature_index] <= value)[0], :]
    mat1 = dataset[np.nonzero(dataset[:, feature_index] > value)[0], :]
    return mat0, mat1


def is_dict(t):
    """Judge if the node is a dictionary"""
    return type(t).__name__ == 'dict'


def get_mean(tree):
    """Calculate left child and right child's mean"""
    if is_dict(tree['right']):
        tree['right'] = get_mean(tree['right'])
    if is_dict(tree['left']):
        tree['left'] = get_mean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, test_data):
    """Find the leaf nodes from top to bottom, and use the test dataset to
    determine whether combining these leaf nodes can reduce the test error

    Args:
        tree: Tree to be pruned
        test_data: Test data required for pruning

    Returns:
        tree: Pruned tree
    """
    l_set = []
    r_set = []
    if np.shape(test_data)[0] == 0:
        return get_mean(tree)
    if is_dict(tree['left']) or is_dict(tree['right']):
        l_set, r_set = split_data(test_data, tree['index'], tree['value'])

    if is_dict(tree['left']):
        tree['left'] = prune(tree['left'], l_set)
    if is_dict(tree['right']):
        tree['right'] = prune(tree['right'], r_set)

    # The above series of operations essentially separates the test dataset
    # according to the training completed tree, and the corresponding values are
    # placed in the corresponding nodes
    # If true:
    #   * Calculate the total variance and the total variance of the result set
    #     itself is not branched
    if not is_dict(tree['left']) and not is_dict(tree['right']):
        l_set, r_set = split_data(test_data, tree['index'], tree['value'])
        error = (np.sum(np.power(l_set[:, -1] - tree['left'], 2)) +
                 np.sum(np.power(r_set[:, -1] - tree['right'], 2)))
        tree_mean = (tree['left'] + tree['right']) / 2.0
        error_post = np.sum(np.power(test_data[:, -1] - tree_mean, 2))

        # If the total variance merged < the total variance not merged, then merge
        if error < error_post:
            return tree_mean
        else:
            return tree
    else:
        return tree
