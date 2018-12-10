#-*- encoding:utf-8 -*-                                                # 指定文件编码为utf-8
  
from __future__ import print_function                                  # 为了在老版本的Python中兼顾新特性的一种方法

import sys                                                             # 在sys.path变量中所列目录中寻找sys.py模块 sys=system 引用sys模块
try:                                                                   # 异常处理
    reload(sys)                                                        # 重新加载sys模块
                                                                       # reload作用：对已经加载的模块进行重新加载，一般用于原模块有变化等特殊情况，reload前该模块必须已经import过。
                                                                       # Python2.5 初始化后删除了 sys.setdefaultencoding 方法，需要重新载入 
    sys.setdefaultencoding('utf-8')                                    # 设置python解析器默认的编码 仅本次有效，因为setdefaultencoding函数在被系统调用后即被删除
except:                                                               
    pass


"""
展示textrank4zh模块的主要功能：

提取关键词

提取关键短语（关键词组）

提取摘要（关键句）

"""

import codecs                                                          # 引用codecs模块（编码转换模块）
from textrank4zh import TextRank4Keyword, TextRank4Sentence            # 从textrank4zh模块中导入TextRank4Keyword（提取关键词）, TextRank4Sentence（生成摘要） 两个类

text = codecs.open('../test/doc/01.txt', 'r', 'utf-8').read()          # 打开并读取文本文件
                                                                       # codecs.open(filepath,method,encoding)
                                                                       # filepath--文件路径; method--打开方式，r为读，w为写，rw为读写; encoding--文件的编码，中文文件使用utf-8
tr4w = TextRank4Keyword()                                              # 创建TextRank4Keyword类的实例

tr4w.analyze(text=text, lower=True, window=2)                          # 对文本进行分析，设定窗口大小为2，并将英文单词转换为小写
                                                        # py2中text必须是utf8编码的str或者unicode对象，py3中必须是utf8编码的bytes或者str对象

print( '关键词：' )                                                    # 输出                      
for item in tr4w.get_keywords(20, word_min_len=1):                     # 从关键词列表中获取最重要的20个长度大于1的关键词
    print(item.word, item.weight)                                      # 输出每个关键词的内容及它的权重
 
print()
print( '关键短语：' )
for phrase in tr4w.get_keyphrases(keywords_num=20, min_occur_num= 2):  # 获取20个关键词构造的可能出现的短语，要求这个短语在原文本中至少出现的次数为2次
    print(phrase)                                                      # 输出关键短语 

tr4s = TextRank4Sentence()                                             # 创建TextRank4Sentence类的实例
tr4s.analyze(text=text, lower=True, source = 'all_filters')            # 对文本进行分析，英文单词小写，使用'all_filters'来生成句子之间的相似度

print()
print( '摘要：' )
for item in tr4s.get_key_sentences(num=3):                             # 获取三个句子作为摘要
    print(item.index, item.weight, item.sentence)                      # 输出句子的 索引、权重、内容
