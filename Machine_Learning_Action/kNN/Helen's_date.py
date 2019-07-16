#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from kNN import auto_norm, classify0, file2matrix


def classify_person():
    """输入一个人的三维特征，进行分类输出

    """
    # 输出结果
    result_list = ['讨厌', '有点喜欢', '非常喜欢']
    # 三维特征用户输入
    video_game = float(input("玩视频游戏所耗时间百分比:"))
    fly_miles = float(input("每年获得的飞行常客里程数:"))
    ice_cream_litre = float(input("每周消费的冰淇淋公升数:"))
    filename = "datingTestSet.txt"  # 打开的文件名
    # 打开并处理数据
    dating_data_mat, dating_labels = file2matrix(filename)
    # 训练集归一化
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    # 生成numpy数组， 测试集
    in_arr = np.array([fly_miles, video_game, ice_cream_litre])
    # 测试集归一化
    norm_in_arr = (in_arr - min_vals) / ranges
    # 返回分类结果
    classifier_result = classify0(norm_in_arr, norm_mat, dating_labels, 3)
    print("你可能%s这个人" % (result_list[classifier_result - 1]))


if __name__ == '__main__':
    classify_person()
