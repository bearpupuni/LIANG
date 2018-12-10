#-*- encoding:utf-8 -*-

from __future__ import (absolute_import, division, print_function,                                      # 在老版本的Python中兼顾新特性
                        unicode_literals)

import networkx as nx                                                                                   # 引入networkx包为nx
import numpy as np                                                                                      # 引入numpy包为np

from . import util                                                                                      # 调用同文件夹下的util.py程序
from .Segmentation import Segmentation                                                                  # 调用同文件夹下的Segmnetation.py中的Segmnetation类

class TextRank4Keyword(object):
    
        # 初始化函数
    def __init__(self, stop_words_file = None, 
                 allow_speech_tags = util.allow_speech_tags, 
                 delimiters = util.sentence_delimiters):
        """
        Keyword arguments: （关键词参数）
        stop_words_file  --  str类型，指定停用词文件的路径（一行一个停用词），若为其他类型，则使用默认的停用词文件
        allow_speech_tags:   词性列表，用于过滤某些词性的词
        delimiters       --  用于将文本拆分为句子的分隔符，默认值为`?!;？！。；…\n`
        
        Object Var:
        self.words_no_filter      --  对sentences中每个句子分词而得到的两级列表。
        self.words_no_stop_words  --  去掉words_no_filter中的停用词而得到的两级列表。
        self.words_all_filters    --  保留words_no_stop_words中指定词性的单词而得到的两级列表。
        """
        self.text = ''
        self.keywords = None
        
        self.seg = Segmentation(stop_words_file=stop_words_file,                 # 创建Segmentation类的实例
                                allow_speech_tags=allow_speech_tags, 
                                delimiters=delimiters)

        self.sentences = None                                                    # 句子列表
        self.words_no_filter = None     # 2维列表                                # 对sentences中每个句子分词而得到的2维列表
        self.words_no_stop_words = None                                          # 去掉words_no_filter中的停止词而得到的2维列表
        self.words_all_filters = None                                            # 保留words_no_stop_words中指定词性的单词而得到的两维列表


        # 分析文本的函数，体现算法思想的部分
    def analyze(self, text, 
                window = 2, 
                lower = False,
                vertex_source = 'all_filters',
                edge_source = 'no_stop_words',
                pagerank_config = {'alpha': 0.85,}):
        """分析文本

        Keyword arguments:
        text       --  文本内容，字符串。
        window     --  窗口大小，int（整型），用来构造单词之间的边。默认值为2。
        lower      --  是否将文本中的英文单词转换为小写。默认为False。
        vertex_source   --  选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来构造pagerank对应的图中的节点。
                            默认值为`'all_filters'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'`。关键词也来自`vertex_source`。
        edge_source     --  选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来构造pagerank对应的图中的节点之间的边。
                            默认值为`'no_stop_words'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'`。边的构造要结合`window`参数。
        pagerank_config:    pagerank算法参数配置，阻尼系数为0.85
        
        """
        
        # self.text = util.as_text(text)
        self.text = text
        self.word_index = {}
        self.index_word = {}

        # 关键词列表
        self.keywords = []                                                      
        self.graph = None
        
        result = self.seg.segment(text=text, lower=lower)
        self.sentences = result.sentences
        self.words_no_filter = result.words_no_filter
        self.words_no_stop_words = result.words_no_stop_words
        self.words_all_filters   = result.words_all_filters

        # 调试
        util.debug(20*'*')
        util.debug('self.sentences in TextRank4Keyword:\n', ' || '.join(self.sentences))
        util.debug('self.words_no_filter in TextRank4Keyword:\n', self.words_no_filter)
        util.debug('self.words_no_stop_words in TextRank4Keyword:\n', self.words_no_stop_words)
        util.debug('self.words_all_filters in TextRank4Keyword:\n', self.words_all_filters)

        # 选项，几种模式
        options = ['no_filter', 'no_stop_words', 'all_filters']

        # 模式选择
        if vertex_source in options:
            _vertex_source = result['words_'+vertex_source]
        else:
            _vertex_source = result['words_all_filters']

        if edge_source in options:
            _edge_source   = result['words_'+edge_source]
        else:
            _edge_source   = result['words_no_stop_words']

        self.keywords = util.sort_words(_vertex_source, _edge_source, window = window, pagerank_config = pagerank_config)
 
        # 获取最重要的num个长度大于等于word_min_len的关键词。
    def get_keywords(self, num = 6, word_min_len = 1):

        """
        num: 返回的关键词个数
        word_min_len: 最小关键词长度
        Return:关键词列表。
        """
        result = []
        count = 0
        for item in self.keywords:
            if count >= num:
                break
            if len(item.word) >= word_min_len:
                result.append(item)
                count += 1
        return result

        # 获取关键短语。
    def get_keyphrases(self, keywords_num = 12, min_occur_num = 2): 

        """
        获取 keywords_num 个关键词构造的可能出现的短语，要求这个短语在原文本中至少出现的次数为min_occur_num。

        keywords_num: 返回的关键词短语个数
        min_occur_num: 短语在文本中的最小出现次数
        Return:关键短语的列表。
        """
        keywords_set = set([ item.word for item in self.get_keywords(num=keywords_num, word_min_len = 1)])     # 关键词集合
        keyphrases = set()                                                                                     # 关键词短语集合
        for sentence in self.words_no_filter:
            one = []
            for word in sentence:
                if word in keywords_set:
                    one.append(word)
                else:
                    if len(one) >  1:
                        keyphrases.add(''.join(one))     # 将关键词组成关键词短语
                    if len(one) == 0:
                        continue
                    else:
                        one = []
            # 兜底
            if len(one) >  1:
                keyphrases.add(''.join(one))

        return [phrase for phrase in keyphrases                  # 在原文本中至少出现min_occur_num词
                if self.text.count(phrase) >= min_occur_num]

# 主模块
if __name__ == '__main__':      # 空语句，保持程序结构的完整性
    pass
