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

sys.path.append(os.getcwd() + "/..")

default_batch_size = 1000


def now():
    return time.strftime("%Y-%m-%d-%H", time.localtime(time.time()))


def get_config(ceph_path="/ceph_ai", mode="train", branch="test"):

    default_batch_size = 1000

    zh_chars_vec = getw2v(
        vec_path=ceph_path + '/xiahong/data/ner_data/char_vec.txt',
        trainable=True,
        vocab_path=ceph_path + '/xiahong/data/ner_data/char_vec.txt',
        vocab_skip_head=True,
        max_vocab_size=200000,
        vec_size=None)  # generate vocab, vocab_size, init_emb, vec_size

    class Config():

        class data_config():
            class seg_data():
                task_id = 0
                tags_paths = [ceph_path +
                              "/xiahong/data/segment_corpus/seg_data/tags.txt"]
                # ceph_path + "/xiahong/data/ner_data/entity_train"][::-1]
                train_paths = [
                    ceph_path +
                    "/xiahong/data/segment_corpus/seg_data/train.data"]
                dev_paths = [
                    ceph_path +
                    "/xiahong/data/segment_corpus/seg_data/test.data"]
                test_paths = [
                    ceph_path +
                    "/xiahong/data/segment_corpus/seg_data/test.data"]
                batch_size = default_batch_size
                vocab = zh_chars_vec.vocab
                sub_vocab = None
                tag_vocab = [{k: name.strip() for k, name in enumerate(
                    open(path))} for path in tags_paths]
                tok = "char"
                seq_len, sub_seq_len = 200, 10
                data_type = "ner"
                names = ["input_zh_y_seg", "input_zh_x", "input_zh_x_length"]

            class ner_data():
                task_id = 0
                tags_paths = [ceph_path + "/xiahong/data/ner_data/tags.txt"]
                # ceph_path + "/xiahong/data/ner_data/entity_train"][::-1]
                train_paths = [ceph_path + "/xiahong/data/ner_data/train.data"]
                dev_paths = [ceph_path + "/xiahong/data/ner_data/test.data"]
                test_paths = [ceph_path + "/xiahong/data/ner_data/test.data"]
                batch_size = default_batch_size
                vocab = zh_chars_vec.vocab
                sub_vocab = None
                tag_vocab = [{k: name.strip() for k, name in enumerate(
                    open(path))} for path in tags_paths]
                tok = "char"
                seq_len, sub_seq_len = 200, 10
                data_type = "ner"
                names = ["input_zh_y_ner", "input_zh_x", "input_zh_x_length"]

            class pos_data():
                task_id = 1
                tags_paths = [ceph_path +
                              "/xiahong/data/segment_corpus/pos_data/tags.txt"]
                train_paths = [
                    ceph_path +
                    "/xiahong/data/segment_corpus/pos_data/train.data"]
                dev_paths = [
                    ceph_path +
                    "/xiahong/data/segment_corpus/pos_data/test.data"]
                test_paths = [
                    ceph_path +
                    "/xiahong/data/segment_corpus/pos_data/test.data"]
                batch_size = default_batch_size
                vocab = zh_chars_vec.vocab
                sub_vocab = None
                tag_vocab = [{k: name.strip() for k, name in enumerate(
                    open(path))} for path in tags_paths]
                tok = "char"
                seq_len, sub_seq_len = 200, 10
                data_type = "ner"
                names = ["input_zh_y_pos", "input_zh_x", "input_zh_x_length"]

            class senti_data():
                task_id = 2
                tags_paths = [ceph_path +
                              "/xiahong/data/sentiment_corpus/tags.txt"]
                train_paths = [ceph_path +
                               "/xiahong/data/sentiment_corpus/train.data"]
                dev_paths = [ceph_path +
                             "/xiahong/data/sentiment_corpus/test.data"]
                test_paths = [ceph_path +
                              "/xiahong/data/sentiment_corpus/test.data"]
                batch_size = default_batch_size
                vocab = None
                sub_vocab = None
                seq_len, sub_seq_len = 200, 10
                tok = "char"
                data_type = "clf"
                outputs = [
                    "input_x_zh",
                    "input_x_zh_length",
                    "input_y_zh_senti"]

            class en_ner_task():
                pass
                # TODO
                # pass

            task2configs = {"ner": ner_data, "pos": pos_data, "seg": seg_data}
            # train_sampling_args =

        class model_config():

            class word2vec_args():
                init_emb = zh_chars_vec.init_emb
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

            class rnn_args2():
                rnn_cell = 'lstm'
                cell_size = 600
                rnn_layer_num = 1
                attn_type = None
                bi = True

            class outputs_args_ner():
                objects = "seq_tag"
                num_classes = 19

            class outputs_args_pos():
                objects = "seq_tag"
                num_classes = 126  # TODO
                use_crf = False

            class outputs_args_seg():
                objects = "seq_tag"
                num_classes = 4  # TODO
                use_crf = False

            class train_args():
                learning_method = "adam_decay"
                start_learning_rate = 0.003
                decay_steps = 6000
                decay_rate = 0.75
                grad_clip = 5

            placeholders = [
                ("input_zh_x", tf.int64, [None, None]),
                ("input_zh_x_length", tf.int64, [None]),
                ("input_zh_y_ner", tf.int64, [None, None]),
                ("input_zh_y_pos", tf.int64, [None, None]),
                ("input_zh_y_seg", tf.int64, [None, None]),
                ("dropout", tf.float32, None)
            ]

            net = [  # 这里输入输出的name表示self.name,而不是计算图中的名字
                [("input_zh_x",), "Word2Vec", word2vec_args,
                 "word2vec", ("words_vec",)],
                [("words_vec", "input_zh_x_length", "dropout"),
                 "Rnn", rnn_args, "rnn", ("words_vec2",)],
                [("words_vec2", "input_zh_x_length", "dropout"),
                 "Rnn", rnn_args2, "rnn2", ("words_vec3",)],
                [("words_vec3", "input_zh_y_ner", "input_zh_x_length"), "Outputs",
                 outputs_args_ner, "output_ner", ("predictions_zh_ner", "loss_zh_ner")],
                [("words_vec2", "input_zh_y_pos", "input_zh_x_length"), "Outputs",
                 outputs_args_pos, "output_pos", ("predictions_zh_pos", "loss_zh_pos")],
                [("words_vec2", "input_zh_y_seg", "input_zh_x_length"), "Outputs",
                 outputs_args_seg, "output_seg", ("predictions_zh_seg", "loss_zh_seg")],
                [("loss_zh_ner",), "TrainOp", train_args,
                 "train1", ("train_op_zh_ner", "lr1")],
                [("loss_zh_pos",), "TrainOp", train_args,
                 "train2", ("train_op_zh_pos", "lr2")],
                [("loss_zh_seg",), "TrainOp", train_args,
                 "train3", ("train_op_zh_seg", "lr3")]
            ]

            train_task2io = {
                "ner": {
                    "inputs": [
                        "input_zh_x", "input_zh_x_length", "input_zh_y_ner", "dropout"], "outputs": [
                        "loss_zh_ner", "train_op_zh_ner"]}, "pos": {
                    "inputs": [
                        "input_zh_x", "input_zh_x_length", "input_zh_y_pos", "dropout"], "outputs": [
                            "loss_zh_pos", "train_op_zh_pos"]}, "seg": {
                                "inputs": [
                                    "input_zh_x", "input_zh_x_length", "input_zh_y_seg", "dropout"], "outputs": [
                                        "loss_zh_seg", "train_op_zh_seg"]}}
            predict_task2io = {
                "ner": {
                    "inputs": [
                        "input_zh_x", "input_zh_x_length", "dropout"], "outputs": [
                        "predictions_zh_ner", "input_zh_x_length"]}, "pos": {
                    "inputs": [
                        "input_zh_x", "input_zh_x_length", "dropout"], "outputs": [
                            "predictions_zh_pos", "input_zh_x_length"]}, "seg": {
                                "inputs": [
                                    "input_zh_x", "input_zh_x_length", "dropout"], "outputs": [
                                        "predictions_zh_seg", "input_zh_x_length"]}}

        class train_config():
            task_type = {"ner": "seq_tag", "pos": "seq_tag", "seg": "seq_tag"}
            epoch_num = 20
            summary_steps = 100
            session_conf = tf.ConfigProto(
                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9),
                device_count={'CPU': 20, 'GPU': 1},
                allow_soft_placement=True,
                log_device_placement=False)
            lrs = ["lr1", "lr2", "lr3"]
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
            # print "MODEL_PATH=", model_path
            # print "LOG_DIR=", summary_dir

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
