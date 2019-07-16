#!/usr/bin/env python
# -*- coding: utf-8 -*-
from snownlp import sentiment


def main():
    sentiment.train('C:/Users/卢洪波/AppData/Roaming/Python/Python37/site-packages/snownlp/sentiment/neg.txt',
                    'C:/Users/卢洪波/AppData/Roaming/Python/Python37/site-packages/snownlp/sentiment/pos.txt')
    sentiment.save('C:/Users/卢洪波/AppData/Roaming/Python/Python37/site-packages/snownlp/sentiment/sentiment.marshal')


if __name__ == '__main__':
    main()
