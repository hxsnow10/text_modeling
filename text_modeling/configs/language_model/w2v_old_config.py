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
word2vec
'''
'''
two type of arguments control: argpase or config file; here select config file.
config about data,model,train,predict etc.
'''




import os

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


def get_config():
    '''
    if Conifg should controled by some parameter, then get_config get this parameter.
    '''
    class Config():
        # ---------------- part1: data -----------------
        # use for SA
        tags_path = "data/w2v/vocab.txt"
        train_paths = ["data/w2v/train.txt"]
        # dev_paths=["data/w2v/dev.txt"]

        data_type = "w2v"
        train_samples = sum(sum(1 for line in open(path))
                            for path in train_paths)
        # test_data_paths = []

        # ----------------- part2: model ---------------------
        # vec
        tok = 'word'  # word, char, word_char
        _words_vec = _chars_vec = None
        _words_vec = Vec(
            vec_path=None,
            trainable=True,
            vocab_path='data/w2v/vocab.txt',
            vocab_skip_head=False,
            max_vocab_size=200000,
            vec_size=300)
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
        seq_len = 1
        text_model = 'None'
        # add, add-idf; cnn, rnn, birnn, rnn_attn, cnn_rnn, rnn_cnn
        # TODO: hs_rnn_attn, denpendency
        if 'add' in text_model or 'rnn' in text_model:
            use_seq_length = True
        else:
            use_seq_length = False
        if 'add' in text_model:
            seq_len = 1000
            use_idf = 'idf' in text_model
            default_idf = 7
            drop_idf = True
            drop_th = 3
        if 'cnn' in text_model:
            cnn_layer_num = 1
            gated = True  # gating of convolution
            seq_len = 200
            filter_sizes = [2, 3, 4, 5, 6]
            filter_nums = [100, 100, 50, 20, 10]
        if 'rnn' in text_model:
            rnn_cell = 'gru'
            cell_size = 300
            rnn_layer_num = 1
            grad_clip = 5
            bi = 'bi' in text_model
            attn_type = 'attn' in text_model

        # last layer
        use_label_weights = False
        objects = "tag"
        sampled = True
        tag_exclusive = True
        num_true = 1
        num_sampled = 5

        # regulization
        dropout_ratio = 0.5
        l1_lambda = 0
        l2_lambda = 0.001

        # ----------------------- part3: train control -----------------
        learning_method = "adam"
        learning_rate = 0.001
        start_learning_rate = 0.003
        decay_rate = 0.75
        batch_size = 5
        decay_steps = train_samples / batch_size
        grad_clip = 5
        epoch_num = 20
        summary_steps = 500
        session_conf = tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.6),
            device_count={'CPU': 20, 'GPU': 0},
            allow_soft_placement=True,
            log_device_placement=False)
        # session_conf=None

        # ------------------- part4: output ----------------------
        exp_id = 'x'
        ask_for_del = False
        super_params = ['text_model', 'learning_method', 'rnn_cell']
        z = locals()
        suffix = str(
            exp_id) + '-'.join(["{}={}".format(name, z.get(name, None)) for name in super_params])
        del z
        model_dir = './RESULT/w2v/model' + suffix
        summary_dir = './RESULT/w2v/log' + suffix
        model_path = os.path.join(model_dir, 'model')
    return Config


config = get_config()
if __name__ == "__main__":
    for name in dir(config):
        print name, '\t', getattr(config, name)
