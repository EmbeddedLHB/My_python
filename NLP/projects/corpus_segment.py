#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import jieba
from Tools import savefile, readfile


def corpus_segment(corpus_path, seg_path):
    """中文分词

    Args:
        corpus_path: 未分词语料库路径
        seg_path: 是分词后语料库存储路径

    Returns:
        无
    """
    # 获取corpus_path下的所有目录
    cate_list = os.listdir(corpus_path)
    """
    其中子目录的名字就是类别名，例如：
    C:/My_python/NLP/projects/train_corpus/art/21.txt中，'.../train_corpus/'是corpus_path，'art'是catelist中的一个成员
    """
    print('分词中...')
    # 获取每个目录（类别）下所有的文件
    # 这里my_dir就是train_corpus/art/21.txt中的art（即cate_list中的一个类别）
    for my_dir in cate_list:
        # 分类子目录路径：C:/My_python/NLP入门/corpus/train_corpus/art/
        class_path = corpus_path + my_dir + "/"
        # 分词后存储的对应目录路径：C:/My_python/NLP入门/corpus/train_corpus_seg/art/
        seg_dir = seg_path + my_dir + "/"

        # 若不存在分词目录则创建该目录
        if not os.path.exists(seg_dir):
            os.makedirs(seg_dir)

        # 获取未分词语料库中某一类别中的所有文本
        file_list = os.listdir(class_path)
        '''
        如train_corpus/art/中的
        21.txt, 22.txt, 23.txt...
        file_list=['21.txt', '22.txt', ...]
        '''

        # 对类别目录下的每一个文件
        for file_path in file_list:
            # 文件名全路径如：C:/My_python/NLP入门/corpus/train_corpus/art/21.txt
            fullname = class_path + file_path
            # 读取文件内容
            content = readfile(fullname)
            '''
            此时，content里面存贮的是原文本的所有字符，例如多余的空格、空行、回车等等，
            接下来，我们需要把这些无关痛痒的字符统统去掉，变成只有标点符号做间隔的紧凑的文本内容
            '''
            # 删除换行
            content = content.replace('\r\n'.encode('utf-8'), ''.encode('utf-8')).strip()
            # 删除空行、多余的空格
            content = content.replace(' '.encode('utf-8'), ''.encode('utf-8')).strip()
            # 为文件内容分词
            content_seg = jieba.cut(content)
            # 将处理后的文件保存到分词后语料目录
            savefile(seg_dir + file_path, ' '.join(content_seg).encode('utf-8'))

    print('中文语料分词结束！！！')


if __name__ == '__main__':
    # 对训练集进行分词

    # 未分词分类语料库路径
    corpus_path = 'C:/My_python/NLP/corpus/train_corpus/'
    # 分词后分类语料库路径
    seg_path = 'C:/My_python/NLP/corpus/train_corpus_seg/'
    corpus_segment(corpus_path, seg_path)

    # 对测试集进行分词

    # 未分词分类语料库路径
    corpus_path = 'C:/My_python/NLP/corpus/test_corpus/'
    # 分词后分类语料库路径
    seg_path = 'C:/My_python/NLP/corpus/test_corpus_seg/'
    corpus_segment(corpus_path, seg_path)
