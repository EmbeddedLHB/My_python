#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from AdaBoost import adaboost_train_ds
from horse_adaboost import load_data


def plot_roc(pred_strengths, class_labels):
    """函数说明: 绘制ROC

    Args:
        pred_strengths: 分类器的预测强度
        class_labels: 类别

    Returns:
        无
    """
    font = FontProperties(fname='c:/windows/fonts/simsun.ttc', size=14)
    # 从(1.0, 1.0)开始计算
    cur = (1.0, 1.0)
    # 用于计算AUC
    y_sum = 0.0
    # 正类的数量
    num_pos = np.sum(np.array(class_labels) == 1.0)
    # x轴及y轴步长
    x_step = 1 / float(len(class_labels) - num_pos)
    y_step = 1 / float(num_pos)

    # 预测强度排序
    sorted_indices = pred_strengths.argsort()
    plt.subplot(111)

    for i in sorted_indices.tolist()[0]:
        # 多了一个true positive，向y轴移动一步
        if class_labels[i] == 1.0:
            del_x = 0
            del_y = y_step
        # 多了一个false positive，向x轴移动一步，y轴的总高度上升
        else:
            del_x = x_step
            del_y = 0
            y_sum += cur[1]
        plt.plot([cur[0], cur[0] - del_x], [cur[1], cur[1] - del_y], c='b')
        cur = (cur[0] - del_x, cur[1] - del_y)

    plt.plot([0, 1], [0, 1], 'b--')
    plt.title('AdaBoost马疝病检测系统的ROC曲线', FontProperties=font)
    plt.xlabel('假阳率:', FontProperties=font)
    plt.ylabel('真阳率:', FontProperties=font)
    plt.axis([0, 1, 0, 1])
    print('AUC面积为:', y_sum * x_step)
    plt.show()


if __name__ == '__main__':
    dataArr, labelArr = load_data('horseColicTraining2.txt')
    weakClassArr, aggClassRes = adaboost_train_ds(dataArr, labelArr, 50)
    plot_roc(aggClassRes.T, labelArr)
