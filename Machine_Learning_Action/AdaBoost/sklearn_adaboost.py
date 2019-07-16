#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from horse_adaboost import load_data


if __name__ == '__main__':
    # 载入数据
    dataArr, classLabels = load_data('horseColicTraining2.txt')
    testArr, testLabels = load_data('horseColicTest2.txt')

    # 训练分类器
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), algorithm='SAMME', n_estimators=10)
    bdt.fit(dataArr, classLabels)

    # 对训练集进行预测
    predictions = bdt.predict(dataArr)
    errArr = np.mat(np.ones((len(dataArr), 1)))
    print('训练集的错误率: %.3f%%' % float(errArr[predictions != classLabels].sum() / len(dataArr) * 100))

    # 对测试集进行预测
    predictions = bdt.predict(testArr)
    errArr = np.mat(np.ones((len(testArr), 1)))
    print('测试集的错误率: %.3f%%' % float(errArr[predictions != testLabels].sum() / len(testArr) * 100))
