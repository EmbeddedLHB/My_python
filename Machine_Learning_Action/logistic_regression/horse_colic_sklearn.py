#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.linear_model import LogisticRegression


def colic_sklearn():
    """函数说明: 使用sklearn构建logistic回归分类器

    Returns:
        无
    """
    training_set = []
    training_labels = []
    with open('horseColicTraining.txt', 'r') as fp_train:
        for line in fp_train.readlines():
            curr_line = line.strip().split('\t')
            line_arr = []
            for i in range(len(curr_line) - 1):
                line_arr.append(float(curr_line[i]))
            training_set.append(line_arr)
            training_labels.append(float(curr_line[-1]))

    test_set = []
    test_labels = []
    with open('horseColicTest.txt', 'r') as fp_test:
        for line in fp_test.readlines():
            curr_line = line.strip().split('\t')
            line_arr = []
            for i in range(len(curr_line) - 1):
                line_arr.append(float(curr_line[i]))
            test_set.append(line_arr)
            test_labels.append(float(curr_line[-1]))
    classifier = LogisticRegression(solver='sag', max_iter=5000).fit(training_set, training_labels)
    test_accuracy = classifier.score(test_set, test_labels) * 100
    print('正确率：%f%%' % test_accuracy)


if __name__ == '__main__':
    colic_sklearn()
