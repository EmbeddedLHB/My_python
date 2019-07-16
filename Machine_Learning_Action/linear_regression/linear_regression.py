#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


def load_data(filename):
    """Function description: Load data from file

    Args:
        filename: The name of the file

    Returns:
        dataset_x: The dataset of x
        dataset_y: The dataset of y
    """
    num_feat = len(open(filename).readline().split('\t')) - 1
    dataset_x = []
    dataset_y = []
    with open(filename) as fp:
        for line in fp:
            # Current line
            line_arr = []
            cur_line = line.strip().split('\t')
            for i in range(num_feat):
                line_arr.append(float(cur_line[i]))
            # Add x0, x1 to the feature array
            dataset_x.append(line_arr)
            # Add x2, which is y, to the label array
            dataset_y.append(float(cur_line[-1]))
    return dataset_x, dataset_y


def calc_regression(dataset_x, dataset_y):
    """Function description: Calculate the regression coefficient w

    Args:
        dataset_x: The dataset of x
        dataset_y: The dataset of y

    Returns:
        w: The regression coefficient
    """
    x_mat = np.mat(dataset_x)
    y_mat = np.mat(dataset_y).T
    xTx = x_mat.T * x_mat
    if np.linalg.det(xTx) == 0.0:
        print("The matrix is singular matrix, can't reverse")
        return
    w = xTx.I * (x_mat.T * y_mat)
    return w


def plot_regression():
    """Function description: Plot the regression line and the data point
    In this function, we have also calculate the correlation coefficient

    Returns:
        None
    """
    dataset_x, dataset_y = load_data('ex0.txt')
    w = calc_regression(dataset_x, dataset_y)
    x_mat = np.mat(dataset_x)
    y_mat = np.mat(dataset_y)
    x_sort = x_mat.copy()
    x_sort.sort(0)
    y_value = x_sort * w
    print(w)
    # The correlation coefficient of x_mat * w and y_mat
    print(np.corrcoef((x_mat * w).T, y_mat))

    plt.figure()
    plt.plot(x_sort[:, 1], y_value, c='red')
    plt.scatter(x_mat[:, 1].flatten().A[0], y_mat.flatten().A[0], s=20, c='blue', alpha=0.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()


if __name__ == '__main__':
    plot_regression()
