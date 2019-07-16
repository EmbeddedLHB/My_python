#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import random
import jieba
from operator import itemgetter
from sklearn.naive_bayes import MultinomialNB


def text_processing(folder_path, test_size=0.2):
    """函数说明: 中文文本处理

    Args:
        folder_path: 文本存放的路径
        test_size: 测试集占比，默认占所有数据集的20%

    Returns:
        tf_list: 按词频降序排序的训练集列表
        train_data_list: 训练集列表
        test_data_list: 测试集列表
        train_class_list: 训练集标签列表
        test_class_list: 测试集标签列表
    """
    # 查看folder_path下的文件
    folder_list = os.listdir(folder_path)
    # 训练集及其类别
    data_list = []
    class_list = []

    # 遍历每个子文件夹
    for folder in folder_list:
        # 根据子文件夹，生成新的路径
        new_folder_path = os.path.join(folder_path, folder)
        # 存放子文件夹下的txt文件的列表
        files = os.listdir(new_folder_path)

        j = 1
        # 遍历每个txt文件
        for file in files:
            if j > 100:
                break
            with open(os.path.join(new_folder_path, file), 'r', encoding='utf-8') as fp:
                raw = fp.read()

            # 精简模式，返回一个可迭代的generator
            word_cut = jieba.cut(raw, cut_all=False)
            # generator转换为list后添加到训练集里
            data_list.append(list(word_cut))
            class_list.append(folder)
            j += 1

    # zip压缩合并，将数据与标签对应压缩
    data_class_list = list(zip(data_list, class_list))
    # 将data_class_list乱序
    random.shuffle(data_class_list)
    # 训练集和测试集切分的索引值
    index = int(len(data_class_list) * test_size) + 1
    # 训练集、测试集及其解压缩
    train_list = data_class_list[index:]
    test_list = data_class_list[:index]
    train_data_list, train_class_list = zip(*train_list)
    test_data_list, test_class_list = zip(*test_list)

    # 统计训练集词频
    words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if word in words_dict.keys():
                words_dict[word] += 1
            else:
                words_dict[word] = 1

    # 根据键的值倒序排序
    sorted_word_dict = sorted(words_dict.items(), key=itemgetter(1), reverse=True)
    # 解压缩并转换成列表
    tf_list, all_words_nums = zip(*sorted_word_dict)
    tf_list = list(tf_list)
    return tf_list, train_data_list, test_data_list, train_class_list, test_class_list


def make_words_set(words_file):
    """函数说明: 读取文件里的内容，并去重

    Args:
        words_file: 文件路径

    Returns:
        words_set: 读取内容的set集合
    """
    words_set = set()
    with open(words_file, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            word = line.strip()
            # 若有文本，则添加到words_set中
            if len(word) > 0:
                words_set.add(word)
    return words_set


def text_vectorization(train_data_list, test_data_list, feature_words):
    """函数说明: 根据feature_words将文本向量化

    Args:
        train_data_list: 训练集
        test_data_list: 测试集
        feature_words: 特征集

    Returns:
        train_vec_list: 训练集向量化列表
        test_vec_list: 测试集向量化列表
    """

    def text_vectorization_t(text, feature_words_t):
        text_words = set(text)
        # 若出现在特征集中，置1，否则置0
        features = [1 if word in text_words else 0 for word in feature_words_t]
        return features

    train_vec_list = [text_vectorization_t(text, feature_words) for text in train_data_list]
    test_vec_list = [text_vectorization_t(text, feature_words) for text in test_data_list]
    return train_vec_list, test_vec_list


def text_feature_select(all_words_list, delete_num, stopwords_set=None):
    """函数说明: 文本特征选取

    Args:
        all_words_list: 训练集所有文本列表
        delete_num: 删除词频最高的delete_num个词
        stopwords_set: 停用词

    Returns:
        features_words: 特征集
    """
    feature_words = []  # 特征列表
    n = 1
    for t in range(delete_num, len(all_words_list), 1):
        # feature_words的维度为1000
        if n > 1000:
            break
        this_word = all_words_list[t]
        # 如果这个词不是数字，而且不是停用词，且1<单词长度<5，那么这个词就可以作为特征词
        if not this_word.isdigit() and this_word not in stopwords_set and 1 < len(this_word) < 5:
            feature_words.append(this_word)
        n += 1
    return feature_words


def text_classifier(train_vec_list, test_vec_list, train_class_list, test_class_list):
    """函数说明: 新闻分类器

    Args:
        train_vec_list: 训练集向量化的特征文本
        test_vec_list: 测试集向量化的特征文本
        train_class_list: 训练集分类标签
        test_class_list: 测试集分类标签

    Returns:
        test_accuracy: 分类器精度
    """
    classifier = MultinomialNB().fit(train_vec_list, train_class_list)
    test_accuracy = classifier.score(test_vec_list, test_class_list)
    return test_accuracy


if __name__ == '__main__':
    # 文本预处理
    folder_path1 = './SogouC/Sample'
    tf_list1, train_data_list1, test_data_list1, train_class_list1, test_class_list1 = \
        text_processing(folder_path1, test_size=0.2)

    # 生成stopwords_set
    stopwords_file = './stopwords_cn.txt'
    stopwords_set1 = make_words_set(stopwords_file)

    # 得到特征词
    feature_words1 = text_feature_select(tf_list1, 450, stopwords_set1)
    train_vec_list1, test_vec_list1 = text_vectorization(train_data_list1, test_data_list1, feature_words1)

    # 分类的精度
    test_accuracy1 = text_classifier(train_vec_list1, test_vec_list1, train_class_list1, test_class_list1)
    print(test_accuracy1)
