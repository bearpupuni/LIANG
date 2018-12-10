#-*- encoding:utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import networkx as nx
import numpy as np

from . import util
from .Segmentation import Segmentation

class TextRank4Sentence(object):
    
    def __init__(self, stop_words_file = None, 
                 allow_speech_tags = util.allow_speech_tags,
                 delimiters = util.sentence_delimiters):
        """
        Keyword arguments:
        stop_words_file  --  str类型，指定停用词文件的路径（一行一个停用词），若为其他类型，则使用默认的停用词文件
        allow_speech_tags:   词性列表，用于过滤某些词性的词
        delimiters       --  用于将文本拆分为句子的分隔符，默认值为`?!;？！。；…\n`

        
        Object Var:
        self.sentences               --  由句子组成的列表。
        self.words_no_filter         --  对sentences中每个句子分词而得到的两级列表。
        self.words_no_stop_words     --  去掉words_no_filter中的停止词而得到的两级列表。
        self.words_all_filters       --  保留words_no_stop_words中指定词性的单词而得到的两级列表。
        """
        self.seg = Segmentation(stop_words_file=stop_words_file,
                                allow_speech_tags=allow_speech_tags,
                                delimiters=delimiters)
        
        self.sentences = None
        self.words_no_filter = None     # 2维列表
        self.words_no_stop_words = None
        self.words_all_filters = None
        
        self.key_sentences = None

        # 分析文本的函数
    def analyze(self, text, lower = False, 
              source = 'no_stop_words', 
              sim_func = util.get_similarity,
              pagerank_config = {'alpha': 0.85,}):
        """
        Keyword arguments:
        text                 --  文本内容，字符串。
        lower                --  是否将文本中的英文单词转换为小写。默认为False。
        source               --  选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来生成句子之间的相似度。
                                 默认值为`'all_filters'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'`。
        sim_func             --  指定计算句子相似度的函数。
        pagerank_config:    pagerank算法参数配置，阻尼系数为0.85
        
        """

        # 关键句子列表
        self.key_sentences = []
        
        result = self.seg.segment(text=text, lower=lower)
        self.sentences = result.sentences
        self.words_no_filter = result.words_no_filter
        self.words_no_stop_words = result.words_no_stop_words
        self.words_all_filters   = result.words_all_filters

        # 选项，几种模式
        options = ['no_filter', 'no_stop_words', 'all_filters']
        if source in options:
            _source = result['words_'+source]
        else:
            _source = result['words_no_stop_words']

        self.key_sentences = util.sort_sentences(sentences = self.sentences,
                                                 words     = _source,
                                                 sim_func  = sim_func,
                                                 pagerank_config = pagerank_config)

        # 获取最重要的num个长度大于等于sentence_min_len的句子用来生成摘要。
    def get_key_sentences(self, num = 6, sentence_min_len = 6):

        """
        num: 返回的关键句个数
        sentence_min_len ：关键句最短长度
        Return:多个句子组成的列表。
        """
        result = []
        count = 0
        for item in self.key_sentences:
            if count >= num:
                break
            if len(item['sentence']) >= sentence_min_len:
                result.append(item)
                count += 1
        return result
    

if __name__ == '__main__':
    pass
