#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
from sklearn.datasets.base import Bunch
from Tools import readfile


def corpus2Bunch(wordbag_path, seg_path):
    """对文本进行Bunch化操作

    Args:
        wordbag_path:
        seg_path:
    Returns:

    """
    # seg_path下的所有子目录，即分类信息
    cate_list = os.listdir(seg_path)
    # 创建一个Bunch实例
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    bunch.target_name.extend(cate_list)

    # 每个目录下所有的文件
    for my_dir in cate_list:
        # 拼出分类子目录的路径
        class_path = seg_path + my_dir + '/'
        # 获取class_path下的所有文件
        file_list = os.listdir(class_path)
        # 遍历类别目录下文件
        for file_path in file_list:
            # 拼出文件名全路径
            fullname = class_path + file_path
            bunch.label.append(my_dir)
            bunch.filenames.append(fullname)
            # 读取文件内容
            bunch.contents.append(readfile(fullname))

    # 将bunch存储到wordbag_path路径中
    with open(wordbag_path, 'wb') as file_obj:
        pickle.dump(bunch, file_obj)
    print('构建文本对象结束！！！')


if __name__ == '__main__':
    # 对训练集进行Bunch化操作：
    # Bunch存储路径
    wordbag_path = 'C:/My_python/NLP/corpus/train_word_bag/train_set.dat'
    # 分词后分类语料库路径
    seg_path = 'C:/My_python/NLP/corpus/train_corpus_seg/'
    corpus2Bunch(wordbag_path, seg_path)

    # 对测试集进行Bunch化操作：
    # Bunch存储路径
    wordbag_path = 'C:/My_python/NLP/corpus/test_word_bag/test_set.da'
    # 分词后分类语料库路径
    seg_path = 'C:/My_python/NLP/corpus/test_corpus_seg/'
    corpus2Bunch(wordbag_path, seg_path)