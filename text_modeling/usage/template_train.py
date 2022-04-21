#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       pku_seg_train
#   Author:         xiahong xiahahaha01@gmail.com
#   Create:         2022/4/20

"""train seg model on pku dataset

这个文件是用户入口。
配置的顺序
1. 定义这个任务类型，数据加载
2. 定义模型网络，模型加载
3. 定义训练的方法、输出路径等
4. 进行训练

"""
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf

from utils.base import now
from utils.word2vec import getw2v

# ===================================================================
# GENERAL SETTING
batch_szie = 1000
OUTPUT_DIR = "result/{}".format(now())
TOP_LOG_PATH = "result/log"

# ===================================================================
# DATA SETTING

TRAIN_PATHS = ["icwb2-data/training/pku_training.train.conll"]
DEV_PATHS = ["icwb2-data/training/pku_training.dev.conll"]
TEST_PATHS = ["icwb2-data/gold/pku_test_gold.conll"]
TAG_PATHS = ["icwb2-data/tags.txt"]


class data_config():

    class seg_data():
        task_id = 1
        tags_paths = TAG_PATHS
        train_paths = TRAIN_PATHS
        dev_paths = DEV_PATHS
        test_paths = TEST_PATHS
        batch_size = batch_size
        vocab = None
        sub_vocab = None
        tag_vocab = [{k: name.strip() for k, name in enumerate(
            open(path))} for path in tags_paths]
        ngram_defs = NGRAM_DEFS
        ngram_words = [
            {k: w[:-1] for k, w in enumerate(open(path))} for path in NGRAM_PATHS]
        tok = "word"
        seq_len, sub_seq_len = 100, None
        data_type = "ner"
        names = ["input_zh_x", "input_zh_y_seg", "input_zh_x_length"]
        split = '\t'

     task2configs = {"seg": seg_data}

data = load_data(data_config)


# ===================================================================
# MODEL GRAPH SETTING
with tf.Session(config=train_config.session_conf) as sess:
    # here name is same with data 
    placeholders = [
        ("input_zh_x", tf.int64, [None, None, None]),
        ("input_zh_x_length", tf.int64, [None]),
        ("input_zh_y_seg", tf.int64, [None, None]),
        ("dropout", tf.float32, None)
    ]
    
    w2v_layer = Word2vec()
    rnn_layer = Rnn(rnn_cell = 'lstm', cell_size = 300, rnn_layer_num = 1, attn_type = None, bi =True)
    crf_layer = CRF(ngram_sizes = ngram_sizes_, tag_size = 6, init_embs = ngram_feature_vecs)
    output_layer = Outputs(objects = "seq_tag", num_classes = 8, use_crf = False)
    train_op = TrainOp(learning_method = "adam_decay", start_learning_rate = 0.01,
                       decay_steps = 6000,
                       decay_rate = 0.95,
                       grad_clip = 5)
    
    # 注意到如果2层计算共享参数，这里可能涉及到scope?
    # 同一层，但不同的输入与输出
    net = [  # 这里输入输出的name表示self.name,而不是计算图中的名字
        [("input_zh_x",), w2v_layer, ("words_vec",)],
        [("words_vec", "input_zh_x_length", "dropout", rnn_layer, ("words_vec2",)],
        [("words_vec2", "input_zh_y_seg", "input_zh_x_length"), output_layer, ("predictions_zh_seg", "loss_zh_seg")],
        [("loss_zh_seg",), train_op, ("train_op_zh_seg", "lr3")]
    ]


    # ==================================
    # TASK GRAPH INPUT AND OUTPUT SETTING
    train_task2io = {
        "seg": {
            "inputs": [
                "input_zh_x",
                "input_zh_x_length",
                "input_zh_y_seg",
                "dropout"],
            "outputs": [
                "loss_zh_seg",
                "train_op_zh_seg"]},
    }
    predict_task2io = {
        "seg": {
            "inputs": [
                "input_zh_x",
                "input_zh_x_length",
                "dropout"],
            "outputs": [
                "predictions_zh_seg",
                "input_zh_x_length"]},
    }

    # use tf.name_scope to manager variable_names
    model = GeneralModel(sess, placeholders, net, )
    model.inits(sess, train_config.restore)
    model.save_info(train_config.model_dir)
     
# ===================================================================
# TRAINING SETTING 
    class train_config():
        task_type = {"seg": "seq_tag"}
        epoch_num = 20
        summary_steps = 10
        session_conf = tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9),
            device_count={'CPU': 10, 'GPU': 1},
            allow_soft_placement=True,
            log_device_placement=False)
        lrs = ["lr3"]
        start_learning_rate = 0.01
        decay_steps = 3
        decay_rate = 0.9
        ask_for_del = False
        z = locals()
        tm = now()
        top_log_path = TOP_LOG_PATH
        model_dir = '{}/model'.format(OUTPUT_DIR)
        summary_dir = '{}/log'.format(OUTPUT_DIR)
        model_path = os.path.join(model_dir, 'model')
        print "MODEL_PATH=", model_path
        print "LOG_DIR=", summary_dir
        restore = False

        return Config
    
    train(sess, model,
          data.train_data, data.dev_data, data.test_data,
          train_config=train_config, tags=data.tags)
