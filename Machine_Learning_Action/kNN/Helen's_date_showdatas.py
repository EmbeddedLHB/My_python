#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from kNN import file2matrix


def showdatas(dating_data_mat, dating_labels):
    """可视化数据

    Args:
        dating_data_mat: 特征矩阵
        dating_labels: 分类Label
    """
    font = FontProperties(fname="C:/windows/fonts/simsun.ttc", size=14)  # 汉字格式
    # 将fig画布分隔成2*2的区域
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(13, 8))
    labels_colors = []
    for i in dating_labels:
        if i == 1:
            labels_colors.append('black')
        if i == 2:
            labels_colors.append('orange')
        if i == 3:
            labels_colors.append('red')

    # 画出散点图，以dating_data_mat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据，散点大小为15，透明度为0.5
    axs[0][0].scatter(x=dating_data_mat[:, 0], y=dating_data_mat[:, 1], color=labels_colors, s=15, alpha=.5)
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比', FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占比', FontProperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图， 以dating_data_mat矩阵的第一(飞行常客里程)， 第三列(冰淇淋)数据画散点数据
    axs[0][1].scatter(x=dating_data_mat[:, 0], y=dating_data_mat[:, 2], color=labels_colors, s=15, alpha=.5)
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰淇淋公升数', FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰淇淋公升数', FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图， 以dating_data_mat矩阵的第二(玩游戏)， 第三列(冰淇淋)数据画散点数据
    axs[1][0].scatter(x=dating_data_mat[:, 1], y=dating_data_mat[:, 2], color=labels_colors, s=15, alpha=.5)
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰淇淋公升数', FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比', FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰淇淋公升数', FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')

    # 设置并添加图例
    didnt_like = mlines.Line2D([], [], color='black', marker='.', markersize=6, label='didntLike')
    small_doses = mlines.Line2D([], [], color='orange', marker='.', markersize=6, label='smallDoses')
    large_doses = mlines.Line2D([], [], color='red', marker='.', markersize=6, label='largeDoses')
    axs[0][0].legend(handles=[didnt_like, small_doses, large_doses])
    axs[0][1].legend(handles=[didnt_like, small_doses, large_doses])
    axs[1][0].legend(handles=[didnt_like, small_doses, large_doses])
    plt.show()


if __name__ == '__main__':
    filename = "datingTestSet.txt"
    data_mat, labels = file2matrix(filename)
    showdatas(data_mat, labels)
