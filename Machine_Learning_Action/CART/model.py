#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def model_leaf(dataset):
    """A model of the leaf node is generated when the data no longer needs to be
    segmented"""
    w, x, y = linear_solve(dataset)
    return w


def model_err(dataset):
    """Return the error of dataset"""
    w, x, y = linear_solve(dataset)
    y_value = x * w
    return sum(np.power(y - y_value, 2))


def linear_solve(dataset):
    """Format the dataset into the target variable y and the independent
    variable x, perform a simple linear regression, and get w

    Args:
        dataset: The input data

    Returns:
        w: Perform regression coefficients for linear regression
    """
    m, n = np.shape(dataset)
    x = np.mat(np.ones((m, n)))
    x[:, 1: n] = dataset[:, 0: n-1]
    y = dataset[:, -1]

    xTx = x.T * x
    if np.linalg.det(xTx) == 0:
        raise NameError("This matrix is singular, can't do inverse")
    w = xTx.I * (x.T * y)
    return w, x, y
