TextModeling
================
基于Tensorflow的文本处理框架，支持文本分类、序列标注、文本自表示等。

# QuickStart
python train.py -h
python train.py -i configs/seg/pku_seg_config.py 

# 使用说明
本框架把用户参数都放到config文件中,通过在config中定义:
+ 数据data_config: 数据路径，解析器
+ 模型model_config: 网络结构，LOSS
+ 训练train_config
为了灵活性，config 本身也是一个python文件。

TODO: 详细补充说明每部分的功能

# More-Tasks
每种任务里可能存在多个子问题，同一个问题也可能有多个配置，也有相关的指标数据。
每个任务会有一个文档加以说明。

## pku-seg

## ner

## pos-seg

## 多任务：分词、词性、NER
example_configs/multi_task_config.py
输入字的序列，在分词、词性、NER3个任务上做序列标注

## graph-emb
图数据需要放在这里吗，原则上是通用的

## word-emb

## pairwise-ranking

## 
TODO: 给每个例子增加充足的注释
