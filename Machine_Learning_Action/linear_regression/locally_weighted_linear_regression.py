#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from linear_regression import load_data, calc_regression


def locally_weighted_linear_regression(test_point, dataset_x, dataset_y, k=1.0):
    """Calculate the regression coefficient w
    using locally weighted linear regression

    Args:
        test_point: Test set
        dataset_x: The dataset of x
        dataset_y: The dataset of y
        k: The parameter of Gaussian core, custom parameter

    Returns:
        w: The regression coefficient
    """
    x_mat = np.mat(dataset_x)
    y_mat = np.mat(dataset_y).T
    m = np.shape(x_mat)[0]
    # Create the weight diagonal matrix
    weights = np.mat(np.eye(m))
    # Calculate every sample's weight
    for i in range(m):
        diff_mat = test_point - x_mat[i, :]
        weights[i, i] = np.exp(diff_mat * diff_mat.T / (-2.0 * k ** 2))

    xTx = x_mat.T * (weights * x_mat)
    if np.linalg.det(xTx) == 0.0:
        print("The matrix is singular, can't reverse")
        return
    w = xTx.I * (x_mat.T * (weights * y_mat))
    return test_point * w


def lwlr_test(dataset_test, dataset_x, dataset_y, k=1.0):
    """The locally weight linear regression test

    Args:
        dataset_test: Test set
        dataset_x: The dataset of x
        dataset_y: The dataset of y
        k: The parameter of Gaussian core, custom parameter

    Returns:
        w: The regression coefficient
    """
    m = np.shape(dataset_test)[0]
    y_value = np.zeros(m)
    for i in range(m):
        y_value[i] = locally_weighted_linear_regression(dataset_test[i], dataset_x, dataset_y, k)
    return y_value


def plot_lwlr_regression():
    """Plot multiple local weighted regression curves

    Returns:
        None
    """
    dataset_x, dataset_y = load_data('ex0.txt')
    y_value1 = lwlr_test(dataset_x, dataset_x, dataset_y, 1.0)
    y_value2 = lwlr_test(dataset_x, dataset_x, dataset_y, 0.01)
    y_value3 = lwlr_test(dataset_x, dataset_x, dataset_y, 0.003)
    x_mat = np.mat(dataset_x)
    y_mat = np.mat(dataset_y)
    index_sorted = x_mat[:, 1].argsort(0)
    x_sorted = x_mat[index_sorted][:, 0, :]

    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    axs[0].plot(x_sorted[:, 1], y_value1[index_sorted], c='red')
    axs[1].plot(x_sorted[:, 1], y_value2[index_sorted], c='red')
    axs[2].plot(x_sorted[:, 1], y_value3[index_sorted], c='red')
    axs[0].scatter(x_mat[:, 1].flatten().A[0], y_mat.flatten().A[0], s=20, c='blue', alpha=0.5)
    axs[1].scatter(x_mat[:, 1].flatten().A[0], y_mat.flatten().A[0], s=20, c='blue', alpha=0.5)
    axs[2].scatter(x_mat[:, 1].flatten().A[0], y_mat.flatten().A[0], s=20, c='blue', alpha=0.5)

    axs0_title = axs[0].set_title('Locally weighted regression curve, k=1.0')
    axs1_title = axs[1].set_title('Locally weighted regression curve, k=0.01')
    axs2_title = axs[2].set_title('Locally weighted regression curve, k=0.003')
    plt.setp(axs0_title, size=15, color='red')
    plt.setp(axs1_title, size=15, color='red')
    plt.setp(axs2_title, size=15, color='red')
    plt.show()


def error_size(dataset_y, y_value):
    """Calculate the error size of the value

    """
    return ((dataset_y - y_value) ** 2).sum()


def abalone_age():
    """Predict abalones' age

    Returns:
        None
    """
    dataset_x, dataset_y = load_data('abalone.txt')
    print('The training set is the same as the test set')
    print('Locally weighted linear regression, the influence of the kernel k size:')
    y_value1 = lwlr_test(dataset_x[0: 99], dataset_x[0: 99], dataset_y[0: 99], 0.1)
    y_value2 = lwlr_test(dataset_x[0: 99], dataset_x[0: 99], dataset_y[0: 99], 1)
    y_value3 = lwlr_test(dataset_x[0: 99], dataset_x[0: 99], dataset_y[0: 99], 10)
    print('When k=0.1, the error size is: ', error_size(dataset_y[0: 99], y_value1.T))
    print('When k=1  , the error size is: ', error_size(dataset_y[0: 99], y_value2.T))
    print('When k=10 , the error size is: ', error_size(dataset_y[0: 99], y_value3.T))

    print('\nChange dataset: The training set is different from the test set')
    print('Locally weighted linear regression, the influence of the kernel k size:')
    y_value1 = lwlr_test(dataset_x[100: 199], dataset_x[0: 99], dataset_y[0: 99], 0.1)
    y_value2 = lwlr_test(dataset_x[100: 199], dataset_x[0: 99], dataset_y[0: 99], 1)
    y_value3 = lwlr_test(dataset_x[100: 199], dataset_x[0: 99], dataset_y[0: 99], 10)
    print('When k=0.1, the error size is: ', error_size(dataset_y[100: 199], y_value1.T))
    print('When k=1  , the error size is: ', error_size(dataset_y[100: 199], y_value2.T))
    print('When k=10 , the error size is: ', error_size(dataset_y[100: 199], y_value3.T))

    print('\nThe training set is different from the test set')
    print('Compare the simple linear regression and when k=1, locally weighted linear regression:')
    print('When k=1, the error size is: ', error_size(dataset_y[100: 199], y_value2.T))
    w = calc_regression(dataset_x[0: 99], dataset_y[0: 99])
    y_value = np.mat(dataset_x[100: 199]) * w
    print("The simple linear regression 's error size is: ",
          error_size(dataset_y[100: 199], y_value.T.A))


if __name__ == '__main__':
    abalone_age()
    plot_lwlr_regression()
