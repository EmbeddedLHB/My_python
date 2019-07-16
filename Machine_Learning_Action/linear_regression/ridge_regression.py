#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from linear_regression import load_data


def ridge_regression(x_mat, y_mat, lam=0.2):
    """Calculate the ridge regression's coefficient

    Args:
        x_mat: The dataset of x
        y_mat: The dataset of y
        lam: Reduction coefficient

    Returns:
        w: The regression coefficient
    """
    xTx = x_mat.T * x_mat
    denominator = xTx + np.eye(np.shape(x_mat)[1]) * lam
    if np.linalg.det(denominator) == 0.0:
        print("The matrix is singular, can't reverse")
        return
    w = denominator.I * (x_mat.T * y_mat)
    return w


def ridge_test(dataset_x, dataset_y):
    """The ridge regression test

    Args:
        dataset_x: The x dataset
        dataset_y: The y dataset

    Returns:
        w_mat: The matrix of regression coefficient
    """
    x_mat = np.mat(dataset_x)
    y_mat = np.mat(dataset_y).T
    # Data regularize
    y_mean = np.mean(y_mat, 0)
    y_mat -= y_mean
    x_means = np.mean(x_mat, 0)
    x_var = np.var(x_mat, 0)
    x_mat = (x_mat - x_means) / x_var

    # The test number of lambda
    num = 30
    w_mat = np.zeros((num, np.shape(x_mat)[1]))
    for i in range(num):
        w = ridge_regression(x_mat, y_mat, np.exp(i-10))
        w_mat[i, :] = w.T
    return w_mat


def plot_w_mat():
    """Plot the ridge regression's coefficient matrix

    Returns:
        None
    """
    dataset_x, dataset_y = load_data('abalone.txt')
    ridge_weights = ridge_test(dataset_x, dataset_y)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridge_weights)

    title = ax.set_title('The relationship between log(lambda) and regression coefficient')
    x_label = ax.set_xlabel('log(lambda)')
    y_label = ax.set_ylabel('Regression coefficient')
    plt.setp(title, size=30, color='red')
    plt.setp(x_label, size=20, color='black')
    plt.setp(y_label, size=20, color='black')
    plt.show()


if __name__ == '__main__':
    plot_w_mat()
