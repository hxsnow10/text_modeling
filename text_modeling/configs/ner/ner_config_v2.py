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
'''
two type of arguments control: argpase or config file; here select config file.
config about data,model,train,predict etc.
'''




import os
import time

import tensorflow as tf


class Vec(object):
    def __init__(
            self,
            vec_path,
            trainable,
            vocab_path,
            vocab_skip_head,
            max_vocab_size,
            vec_size=300,
            norm=False):
        self.vec_path = vec_path
        self.trainable = trainable
        self.vocab_path = vocab_path
        self.vocab_skip_head = vocab_skip_head
        self.max_vocab_size = max_vocab_size
        self.vec_size = int(open(vec_path).readline().strip().split()[-1])\
            if vec_path else vec_size
        self.norm = norm


def now():
    return time.strftime("%Y-%m-%d-%H", time.localtime(time.time()))


def get_config(ceph_path="/ceph_ai", mode="train", branch="test"):
    '''
    if Conifg should controled by some parameter, then get_config get this parameter.
    '''
    class Config():
        # ---------------- part1: data -----------------
        tags_paths = [ceph_path + "/xiahong/data/ner_data/tags.txt"]
        train_paths = [ceph_path + "/xiahong/data/ner_data/train.data"]
        dev_paths = [ceph_path + "/xiahong/data/ner_data/test.data"]
        test_paths = [ceph_path + "/xiahong/data/ner_data/test.data"]

        data_type = "ner"
        try:
            print os.listdir(ceph_path)
        except Exception as e:
            print e
        # if mode=="train":
        #    train_samples=sum(sum(1 for line in open(path) if line.strip()=="") for path in train_paths)
        #    print "TRAIN SAMPLES=", train_samples

        # ----------------- part2: model ---------------------
        # vec
        tok = 'char'  # word, char, word_char
        _words_vec = _chars_vec = None
        _chars_vec = Vec(
            vec_path=ceph_path + '/xiahong/data/ner_data/char_vec.txt',
            trainable=True,
            vocab_path=ceph_path + '/xiahong/data/ner_data/char_vec.txt',
            vocab_skip_head=True,
            max_vocab_size=200000,
            vec_size=None)
        char_len = 10
        if tok == 'word':  # word vec as word_vec
            split = ' '
            words_vec = _words_vec
            chars_vec = None
        elif tok == 'char':  # char vec as word_vec
            split = 'char'
            words_vec = _chars_vec
            chars_vec = None
            chars_vec = None
        elif tok == "word_char":  # word vec as word_vec, char vec as subword_vec
            split = ' '
            words_vec = _words_vec
            chars_vec = _chars_vec
            # define subword model
            char_len = 10  # define word has max len char
            subword_model = "cnn"
            char_filter_sizes = [1, 2, 3]
            char_filter_nums = [100, 100, 100]

        # text_model
        seq_len = 150
        text_model = 'cnn'
        # add, add-idf; cnn, rnn, birnn, rnn_attn, cnn_rnn, rnn_cnn
        # TODO: hs_rnn_attn, denpendency
        if 'add' in text_model or 'rnn' in text_model:
            use_seq_length = True
        else:
            use_seq_length = True
        if 'add' in text_model:
            seq_len = 1000
            use_idf = 'idf' in text_model
            default_idf = 7
            drop_idf = True
            drop_th = 3
        if 'cnn' in text_model:
            cnn_layer_num = 2
            gated = True  # gating of convolution
            seq_len = 200
            filter_sizes = [1, 2, 3, 4]
            filter_nums = [50, 20, 20, 20]
        if 'rnn' in text_model:
            rnn_cell = 'sru'
            cell_size = 300
            rnn_layer_num = 2
            grad_clip = 5
            bi = 'bi' in text_model
            attn_type = 'attn' in text_model

        # last layer
        objects = "seq_tag"
        tag_exclusive = True
        use_label_weights = False
        crf = True
        sampled = False
        loss_type = ''

        # regulization
        dropout_ratio = 0.5
        l2_lambda = 0.001

        # ----------------------- part3: train control -----------------
        learning_method = "adam_decay"
        learning_rate = 0.001
        start_learning_rate = 0.003
        decay_rate = 0.75
        batch_size = 500
        decay_steps = 6000
        grad_clip = 5
        epoch_num = 20
        summary_steps = 100
        session_conf = tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9),
            device_count={'CPU': 20, 'GPU': 1},
            allow_soft_placement=True,
            log_device_placement=False)
        # session_conf=None

        # ------------------- part4: output ----------------------
        ask_for_del = False
        super_params = ['text_model', 'cnn_layer_num']
        z = locals()
        suffix = '-'.join(["{}={}".format(name, z.get(name, None))
                          for name in super_params])
        tm = now()
        model_dir = ceph_path + \
            '/xiahong/RESULT/ner/{}/{}/model/{}'.format(branch, tm, suffix)
        summary_dir = ceph_path + \
            '/xiahong/RESULT/ner/{}/{}/log/{}'.format(branch, tm, suffix)
        model_path = os.path.join(model_dir, 'model')
        print "MODEL_PATH=", model_path
        print "LOG_DIR=", summary_dir
    return Config


if __name__ == "__main__":
    for name in dir(config):
        print name, '\t', getattr(config, name)
