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
sequence tagging(small) NER
'''


import json
import os
import sys
import time

import tensorflow as tf

from utils.word2vec import getw2v

sys.path.append("..")

default_batch_size = 1


def now():
    return time.strftime("%Y-%m-%d-%H", time.localtime(time.time()))


def get_config(ceph_path="/ceph_ai", mode="train", branch="develop"):

    default_batch_size = 1

    zh_words_vec = getw2v(
        vec_path="/ceph_ai/xiahong/data/w2v/bert/words_unq.txt.bert.vec",
        trainable=True,
        vocab_path="/ceph_ai/xiahong/data/w2v/bert/words_unq.txt.bert.vec",
        vocab_skip_head=True,
        max_vocab_size=None,
        vec_size=768)  # generate vocab, vocab_size, init_emb, vec_size

    class Config():

        class data_config():

            class pos_data():
                task_id = 1
                tags_paths = [
                    ceph_path +
                    "/xiahong/data/segment_corpus/pos_tokens/tags.txt"]
                train_paths = [
                    ceph_path +
                    "/xiahong/data/segment_corpus/pos_tokens/train.data"]
                dev_paths = [
                    ceph_path +
                    "/xiahong/data/segment_corpus/pos_tokens/test.data"]
                test_paths = [
                    ceph_path +
                    "/xiahong/data/segment_corpus/pos_tokens/test.data"]
                batch_size = default_batch_size
                vocab = zh_words_vec.vocab
                sub_vocab = None
                tag_vocab = [{k: name.strip() for k, name in enumerate(
                    open(path))} for path in tags_paths]
                tok = "word_char"
                seq_len, sub_seq_len = 20, 5
                data_type = "ner"
                names = ["input_zh_x", "input_zh_y_pos", "input_zh_x_length"]

            task2configs = {"pos": pos_data}
            # train_sampling_args =

        class model_config():

            class word2vec_args():
                init_emb = zh_words_vec.init_emb
                w2v_shape = None
                sub_init_emb = None
                sub_w2v_shape = None

            class rnn_args():
                pass

            class rnn_args2():
                rnn_cell = 'lstm'
                cell_size = 600
                rnn_layer_num = 1
                attn_type = None
                bi = True

            class outputs_args_pos():
                objects = "seq_tag"
                num_classes = 108  # TODO
                use_crf = True

            class train_args():
                learning_method = "adam_decay"
                start_learning_rate = 0.003
                decay_steps = 6000
                decay_rate = 0.75
                grad_clip = 5

            placeholders = [
                ("input_zh_x", tf.int64, [None, None]),
                ("input_zh_x_length", tf.int64, [None]),
                ("input_zh_y_pos", tf.int64, [None, None]),
                ("dropout", tf.float32, None)
            ]

            net = [  # ?????????????????????name??????self.name,??????????????????????????????
                [("input_zh_x", ), "Word2Vec",
                 word2vec_args, "word2vec", ("words_vec",)],
                [("words_vec", "input_zh_y_pos", "input_zh_x_length"), "Outputs",
                 outputs_args_pos, "output_pos", ("predictions_zh_pos", "loss_zh_pos")],
                [("loss_zh_pos",), "TrainOp", train_args,
                 "train2", ("train_op_zh_pos", "lr")],
            ]

            train_task2io = {
                "pos": {
                    "inputs": [
                        "input_zh_x",
                        "input_zh_x_length",
                        "input_zh_y_pos",
                        "dropout"],
                    "outputs": [
                        "loss_zh_pos",
                        "train_op_zh_pos"]},
            }
            predict_task2io = {
                "pos": {
                    "inputs": [
                        "input_zh_x",
                        "input_zh_x_length",
                        "dropout"],
                    "outputs": [
                        "predictions_zh_pos",
                        "input_zh_x_length"]},
            }

        class train_config():
            task_type = {"pos": "seq_tag"}
            epoch_num = 20
            summary_steps = 10
            session_conf = tf.ConfigProto(
                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9),
                device_count={'CPU': 10, 'GPU': 1},
                allow_soft_placement=True,
                log_device_placement=False)
            lrs = ["lr"]
            start_learning_rate = 0.003
            decay_steps = 2
            decay_rate = 0.75
            ask_for_del = False
            super_params = ['text_model', 'cnn_layer_num']
            z = locals()
            suffix = '-'.join(["{}={}".format(name, z.get(name, None))
                              for name in super_params])
            tm = now()
            top_log_path = ceph_path + '/xiahong/LOG'
            model_dir = ceph_path + \
                '/xiahong/RESULT/ner/{}/{}v2/model/{}'.format(branch, tm, suffix)
            summary_dir = ceph_path + \
                '/xiahong/RESULT/ner/{}/{}v2/log/{}'.format(branch, tm, suffix)
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
