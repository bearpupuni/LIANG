#-*- encoding:utf-8 -*-

from __future__ import (absolute_import, division, print_function,    # 兼容           division：分割
                        unicode_literals)

import os                # 引入包
import math
import networkx as nx
import numpy as np
import sys

try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass
    
sentence_delimiters = ['?', '!', ';', '？', '！', '。', '；', '……', '…', '\n']       # 句子默认分隔符（数组声明）
allow_speech_tags = ['an', 'i', 'j', 'l', 'n', 'nr', 'nrfg', 'ns', 'nt', 'nz', 't', 'v', 'vd', 'vn', 'eng']   # 词性列表（数组声明）

PY2 = sys.version_info[0] == 2
if not PY2:                                                # 非python2版本   
    # Python 3.x and up                                    python3及以上
    text_type    = str                                     # 文本类型为str字符串
    string_types = (str,)                                  # 字符串用，分割
    xrange       = range                                   # range 循环

    def as_text(v):  ## 生成unicode字符串
        if v is None:
            return None
        elif isinstance(v, bytes):                         # isinstance 判断对象的变量类型。 如果为bytes
            return v.decode('utf-8', errors='ignore')      # 返回 decode 转换为utf8 输出错误信息'ignore'
        elif isinstance(v, str):                           # 如果判断为str 
            return v
        else:
            raise ValueError('Unknown type %r' % type(v))  # 返回错误信息 不知道v的类型

    def is_text(v):
        return isinstance(v, text_type)                    # 返回v

else:
    # Python 2.x
    text_type    = unicode
    string_types = (str, unicode)
    xrange       = xrange

    def as_text(v):
        if v is None:
            return None
        elif isinstance(v, unicode):
            return v
        elif isinstance(v, str):
            return v.decode('utf-8', errors='ignore')
        else:
            raise ValueError('Invalid type %r' % type(v))

    def is_text(v):
        return isinstance(v, text_type)

__DEBUG = None   #  声明__DEBUG为空none

def debug(*args):                  # debug方法  *args可以接受序列的输入参数。当函数的参数不确定时，可以使用*args。 
    global __DEBUG                 # 定义为全局变量
    if __DEBUG is None:
        try:
            if os.environ['DEBUG'] == '1':    # 如果debug为1
                __DEBUG = True
            else:
                __DEBUG = False
        except:
            __DEBUG = False
    if __DEBUG:
        print( ' '.join([str(arg) for arg in args]) )     # 将arg转换为字符串输出  生成了元组、列表、字典后，可以用 join() 来转化为字符串  “for arg in args”？

# 这个类看不太懂 2018.10.28
class AttrDict(dict):                                     # 2018.11.4  继承dict类 ? 字典            
    """Dict that can get attribute by dot"""              # AttrDict类
    def __init__(self, *args, **kwargs):                  # *args表示任何多个无名参数，它是一个tuple； **kwargs表示关键字参数，它是一个dict。并且同时使用*args和**kwargs时，必须*args参数列要在**kwargs前
        super(AttrDict, self).__init__(*args, **kwargs)   # super() 函数是用于调用父类(超类)的一个方法
        self.__dict__ = self


def combine(word_list, window = 2):                         
    """构造在window下的单词组合，用来构造单词之间的边。
    
    Keyword arguments:
    word_list  --  list of str, 由单词组成的列表。
    windows    --  int, 窗口大小。
    """
    if window < 2: window = 2                             # 如果窗口大小小于2，则等于2
    for x in xrange(1, window):                           # xrange循环
        if x >= len(word_list):
            break
        word_list2 = word_list[x:]
        res = zip(word_list, word_list2)                  # 将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
        for r in res:
            yield r                                       # 在一个函数中，程序执行到yield语句的时候，程序暂停，返回yield后面表达式的值，在下一次调用的时候，从yield语句暂停的地方继续执行，如此循环，直到函数执行完。

def get_similarity(word_list1, word_list2):
    """默认的用于计算两个句子相似度的函数。

    Keyword arguments:
    word_list1, word_list2  --  分别代表两个句子，都是由单词组成的列表
    """
    words   = list(set(word_list1 + word_list2))                   # words为两个句子set的列表    
    vector1 = [float(word_list1.count(word)) for word in words]    #第一个句子
    vector2 = [float(word_list2.count(word)) for word in words]    #第二个句子
                                                                         # 分子部分的意思是同时出现在两个句子中的同一个词的数量
    vector3 = [vector1[x]*vector2[x]  for x in xrange(len(vector1))]     # for循环
    vector4 = [1 for num in vector3 if num > 0.]
    co_occur_num = sum(vector4)      # 求和（相同数量）                        

    if abs(co_occur_num) <= 1e-12:
        return 0.
    
    denominator = math.log(float(len(word_list1))) + math.log(float(len(word_list2))) # 分母  对句子中词的个数求对数后求和
    
    if abs(denominator) < 1e-12:
        return 0.
    
    return co_occur_num / denominator      # 返回相似度计算结果

def sort_words(vertex_source, edge_source, window = 2, pagerank_config = {'alpha': 0.85,}):   # 阻尼系数为0.85
    """将单词按关键程度从大到小排序

    Keyword arguments:
    vertex_source   --  二维列表，子列表代表句子，子列表的元素是单词，这些单词用来构造pagerank中的节点
    edge_source     --  二维列表，子列表代表句子，子列表的元素是单词，根据单词位置关系构造pagerank中的边
    window          --  一个句子中相邻的window个单词，两两之间认为有边
    pagerank_config --  pagerank的设置
    """
    sorted_words   = []
    word_index     = {}
    index_word     = {}
    _vertex_source = vertex_source            # 单词节点
    _edge_source   = edge_source              # 单词边
    words_number   = 0
    for word_list in _vertex_source:          # 循环  排序
        for word in word_list:
            if not word in word_index:           
                word_index[word] = words_number
                index_word[words_number] = word
                words_number += 1

    graph = np.zeros((words_number, words_number))     # numpy.zeros() 创建一个二维数组 
    
    for word_list in _edge_source:
        for w1, w2 in combine(word_list, window):        # 构造单词的边
            if w1 in word_index and w2 in word_index:
                index1 = word_index[w1]
                index2 = word_index[w2]
                graph[index1][index2] = 1.0
                graph[index2][index1] = 1.0

    debug('graph:\n', graph)
    
    nx_graph = nx.from_numpy_matrix(graph)
    scores = nx.pagerank(nx_graph, **pagerank_config)          # this is a dict
    sorted_scores = sorted(scores.items(), key = lambda item: item[1], reverse=True)
    for index, score in sorted_scores:
        item = AttrDict(word=index_word[index], weight=score)
        sorted_words.append(item)

    return sorted_words

def sort_sentences(sentences, words, sim_func = get_similarity, pagerank_config = {'alpha': 0.85,}):
    """将句子按照关键程度从大到小排序

    Keyword arguments:
    sentences         --  列表，元素是句子
    words             --  二维列表，子列表和sentences中的句子对应，子列表由单词组成
    sim_func          --  计算两个句子的相似性，参数是两个由单词组成的列表
    pagerank_config   --  pagerank的设置
    """
    sorted_sentences = []
    _source = words
    sentences_num = len(_source)        
    graph = np.zeros((sentences_num, sentences_num))      # 创建一个二维数组
    
    for x in xrange(sentences_num):                                 
        for y in xrange(x, sentences_num):
            similarity = sim_func( _source[x], _source[y] )
            graph[x, y] = similarity
            graph[y, x] = similarity
            
    nx_graph = nx.from_numpy_matrix(graph)
    scores = nx.pagerank(nx_graph, **pagerank_config)              # this is a dict  (字典？)
    sorted_scores = sorted(scores.items(), key = lambda item: item[1], reverse=True)

    for index, score in sorted_scores:
        item = AttrDict(index=index, sentence=sentences[index], weight=score)
        sorted_sentences.append(item)

    return sorted_sentences

if __name__ == '__main__':
    pass
