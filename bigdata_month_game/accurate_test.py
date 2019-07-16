#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd


def main():
    """函数说明: 判断当前csv文件答案的准确度"""

    # 将标准答案的标签添加到数组中
    result = pd.read_csv('result.csv')
    result_num = result['label']

    # 将自己答案的标签添加到数组中
    my_result = pd.read_csv('my_result.csv')
    my_num = my_result['label']

    # 若标准答案和自己答案数组长度不同，则说明算法运行中遗漏了数据
    if len(result_num) != len(my_num):
        print("ERROR")
        exit()

    # 错误的个数
    error = 0
    for i in range(len(result_num)):
        if result_num[i] != my_num[i]:
            error += 1

    print('错误数:   ', error)
    print('总标签数: ', len(my_num))
    print('正确率:   ', 1 - error / len(my_num))


if __name__ == '__main__':
    main()
