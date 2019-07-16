#!/usr/bin/env python
# -*- coding: utf-8 -*-
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
from linear_regression import load_data


def regularize(x_mat, y_mat):
    """Data regularize

    Args:
        x_mat: Dataset x
        y_mat: Dataset y

    Returns:
        r_x_mat: The dataset x after regularize
        r_y_mat: The dataset y after regularize
    """
    rx_mat = x_mat.copy()
    y_mean = np.mean(y_mat, 0)
    ry_mat = y_mat - y_mean

    x_means = np.mean(rx_mat, 0)
    x_var = np.var(rx_mat, 0)
    rx_mat = (rx_mat - x_means) / x_var
    return rx_mat, ry_mat


def square_error(y_arr, y_value):
    """Calculate the square error

    """
    return ((y_arr - y_value) ** 2).sum()


def stage_wise(dataset_x, dataset_y, len_step=0.01, num=100):
    """Forward

    Args:
        dataset_x: Input data x
        dataset_y: Predicted data y
        len_step: Step size to be adjusted for each iteration
        num: Number of iterations

    Returns:
        return_mat: Regression coefficient matrix of num sub-iterations
    """
    x_mat = np.mat(dataset_x)
    y_mat = np.mat(dataset_y).T
    x_mat, y_mat = regularize(x_mat, y_mat)
    m, n = np.shape(x_mat)

    return_mat = np.zeros((num, n))
    w = np.zeros((n, 1))
    w_max = w.copy()

    for i in range(num):
        min_error = float('inf')
        for j in range(n):
            for sign in [-1, 1]:
                w_test = w.copy()
                w_test[j] += len_step * sign
                y_test = x_mat * w_test
                error = square_error(y_mat.A, y_test.A)

                if error < min_error:
                    min_error = error
                    w_max = w_test
        w = w_max.copy()
        return_mat[i, :] = w.T
    return return_mat


def plot_stage_wise_mat():
    """Plot forward stage-wise linear regression's coefficient matrix

    Returns:
        None
    """
    dataset_x, dataset_y = load_data('abalone.txt')
    ridge_weights = stage_wise(dataset_x, dataset_y, 0.005, 1000)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridge_weights)

    title = ax.set_title('The relationship between log(lambda) and regression coefficient')
    x_label = ax.set_xlabel('Number of iterations')
    y_label = ax.set_ylabel('Regression coefficient')
    plt.setp(title, size=30, color='red')
    plt.setp(x_label, size=20, color='black')
    plt.setp(y_label, size=20, color='black')
    plt.show()


if __name__ == '__main__':
    plot_stage_wise_mat()
