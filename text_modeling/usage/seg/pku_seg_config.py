#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:
#   Author:         xiahong xiahahaha01@gmail.com
#   Create:

"""summary

description:

Usage:
foo = ClassFoo()
bar = foo.FunctionBar()
"""
# encoding=utf-8
'''
train seg model on pku dataset
'''


import json
import os
import sys
import time

import numpy as np
import tensorflow as tf

from utils.word2vec import getw2v

sys.path.append("../..")


def now():
    return time.strftime("%Y%m%d%H", time.localtime(time.time()))

# SEQUENCE TAGGING PROBLEM SETTING


TRAIN_PATHS = ["icwb2-data/training/pku_training.train.conll"]
DEV_PATHS = ["icwb2-data/training/pku_training.dev.conll"]
TEST_PATHS = ["icwb2-data/gold/pku_test_gold.conll"]
TAG_PATHS = ["icwb2-data/tags.txt"]

# OUTPUT SETTING
tm = now()
OUTPUT_DIR = "result/{}".format(tm)
TOP_LOG_PATH = "result/log"

# TOTAL SETTING (DATA, MODEL, TRAIN )
default_batch_szie = 1000
NGRAM_PATHS = [
    "icwb2-data/feature.unigram",
    "icwb2-data/feature.bigram",
    "icwb2-data/feature.trigram"]
NGRAM_DEFS = ["w[k];w[k-1];w[k+1]", "w[k],w[k+1];w[k-1],w[k]",
              "w[k-2],w[k-1],w[k];w[k],w[k+1],w[k+2];w[k-1],w[k],w[k+1]"]
NGRAM_FEATURE_PATHS = [
    "icwb2-data/feature.unigram.vals",
    "icwb2-data/feature.bigram.vals",
    "icwb2-data/feature.trigram.vals"]

# NGRAM_PATHS = ["icwb2-data/feature.unigram","icwb2-data/feature.bigram"]
# NGRAM_DEFS = ["w[k]", "w[k],w[k+1];w[k-1],w[k]"]
# NGRAM_FEATURE_PATHS = ["icwb2-data/feature.unigram.vals","icwb2-data/feature.bigram.vals"]

# NGRAM_PATHS = ["icwb2-data/feature.unigram","icwb2-data/feature.bigram"]
# NGRAM_DEFS = ["w[k];w[k-1];w[k+1]", "w[k],w[k+1];w[k-1],w[k]"]
# NGRAM_FEATURE_PATHS = ["icwb2-data/feature.unigram.vals","icwb2-data/feature.bigram.vals", "icwb2-data/feature.trigram.vals"]
#NGRAM_PATHS = ["icwb2-data/feature.unigram","icwb2-data/feature.bigram","icwb2-data/feature.trigram"]
#NGRAM_DEFS = ["w[k];w[k-1];w[k+1]", "w[k],w[k+1];w[k-1],w[k];w[k-2],w[k-1];w[k+1],w[k+2]", "w[k-2],w[k-1],w[k];w[k],w[k+1],w[k+2];w[k-1],w[k],w[k+1]"]


def get_emb(path, words_num, vec_size=2):
    emb = []
    for line in open(path):
        vals = [float(x) for x in line.strip().split(' ')]
        emb.append(vals)
    rval = np.array(emb + [[0, ] * vec_size, ] *
                    (words_num - len(emb)), dtype=np.float32)
    return rval


def get_config(branch="develop"):

    default_batch_size = 1000
    ngram_words = [{k: w[:-1]
                    for k, w in enumerate(open(path))} for path in NGRAM_PATHS]
    ngram_sizes_ = sum([[len(words) + 5, ] * len(NGRAM_DEFS[k].split(';'))
                       for k, words in enumerate(ngram_words)], [])
    ngram_feature_vecs = sum([[get_emb(path,
                                       len(ngram_words[k]) + 5),
                               ] * len(NGRAM_DEFS[k].split(';')) for k,
                              path in enumerate(NGRAM_FEATURE_PATHS)],
                             [])
    print ngram_sizes_
    for vec in ngram_feature_vecs:
        print vec.shape
    '''
    zh_chars_vec=getw2v(
        vec_path=vec_path,
        trainable=True,
        vocab_path=vec_path,
        vocab_skip_head=True,
        max_vocab_size=200000,
        vec_size=None) # generate vocab, vocab_size, init_emb, vec_size
    '''
    class Config():

        class data_config():

            class seg_data():
                task_id = 1
                tags_paths = TAG_PATHS
                train_paths = TRAIN_PATHS
                dev_paths = DEV_PATHS
                test_paths = TEST_PATHS
                batch_size = default_batch_size
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
            # train_sampling_args =

        class model_config():
            class crf_args():
                ngram_sizes = ngram_sizes_
                tag_size = 6
                init_embs = ngram_feature_vecs
                # print ngram_sizes

            class word2vec_args():
                init_emb = None
                w2v_shape = None
                sub_init_emb = None
                sub_w2v_shape = None
                sub_cnn = None

            class rnn_args():
                rnn_cell = 'lstm'
                cell_size = 300
                rnn_layer_num = 1
                attn_type = None
                bi = True

            class outputs_args_seg():
                objects = "seq_tag"
                num_classes = 8  # TODO
                use_crf = False

            class train_args():
                learning_method = "adam_decay"
                start_learning_rate = 0.01
                decay_steps = 6000
                decay_rate = 0.95
                grad_clip = 5

            placeholders = [
                ("input_zh_x", tf.int64, [None, None, None]),
                ("input_zh_x_length", tf.int64, [None]),
                ("input_zh_y_seg", tf.int64, [None, None]),
                ("dropout", tf.float32, None)
            ]

            net = [  # 这里输入输出的name表示self.name,而不是计算图中的名字
                [("input_zh_x",), "Word2Vec", word2vec_args,
                 "word2vec", ("words_vec",)],
                [("words_vec", "input_zh_x_length", "dropout"),
                 "Rnn", rnn_args, "rnn", ("words_vec2",)],
                [("words_vec2", "input_zh_y_seg", "input_zh_x_length"), "Outputs",
                 outputs_args_seg, "output_seg", ("predictions_zh_seg", "loss_zh_seg")],
                [("loss_zh_seg",), "TrainOp", train_args,
                 "train3", ("train_op_zh_seg", "lr3")]
            ]
            net_crf = [  # 这里输入输出的name表示self.name,而不是计算图中的名字
                [("input_zh_x", "input_zh_y_seg", "input_zh_x_length"), "CRF",
                 crf_args, "crf", ("predictions_zh_seg", "loss_zh_seg")],
                [("loss_zh_seg",), "TrainOp", train_args,
                 "train", ("train_op_zh_seg", "lr3")],
            ]

            net = net_crf

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

    return Config


if __name__ == "__main__":
    config = get_config()
    for name in dir(config):
        if name[0] == '_':
            continue
        print name, '\t', getattr(config, name)
        v = getattr(config, name)
        print '-' * 40
        for name_2 in dir(v):
            print '\t', name_2, getattr(v, name_2)
