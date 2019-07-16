#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def is_dict(t):
    """Judge if the node is a dictionary"""
    return type(t).__name__ == 'dict'


def regression_tree(model, data):
    return float(model)


def model_tree(model, dataset):
    """Predict the model tree

    Args:
        model: Input model, the optional value is the regression tree model or
        the model tree model, here is the model tree model
        dataset: Input test data

    Returns:
        float(x * model): test data * coefficient
    """
    n = np.shape(dataset)[1]
    x = np.mat(np.ones((1, n + 1)))
    x[:, 1: n + 1] = dataset
    return float(x * model)


def tree_forecast(tree, dataset, model=regression_tree):
    """Forecasting the tree of a particular model, either a regression tree
    or a model tree

    Args:
        tree: Model of a tree that has been trained
        dataset: Input test data
        model: The model type of the predicted tree. The optional value is
        regression tree or model tree.

    Returns:
        Predicted value
    """
    if not is_dict(tree):
        return model_tree(tree, dataset)
    if dataset[tree['index']] <= tree['value']:
        if is_dict(tree['left']):
            return tree_forecast(tree['left'], dataset, model)
        else:
            return model_tree(tree['left'], dataset)
    else:
        if is_dict(tree['right']):
            return tree_forecast(tree['right'], dataset, model)
        else:
            return model_tree(tree['right'], dataset)


def forecast(tree, test_data, model=regression_tree):
    """Call tree_forecast to predict the tree of a particular model,
    either a regression tree or a model tree

    Args:
        tree: Model of a tree that has been trained
        test_data: Input test data
        model: The model type of the predicted tree. The optional value is
        regression tree or model tree.

    Returns:
        Predictive matrix
    """
    m = len(test_data)
    y_mat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        y_mat[i, 0] = tree_forecast(tree, np.mat(test_data[i]), model)
    return y_mat


if __name__ == '__main__':
    pass
