#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:
#   Author:         xiahong xiahahaha01@gmail.com
#   Create:

"""rnn module.
"""
import tensorflow as tf
from tensorflow.contrib import rnn

from .sru import SRUCell


def rnn_func(inputs, rnn_cell='lstm', cell_size=300, rnn_layer_num=1,
             attn_type=None, bi=False, sequence_length=None):
    """ rnn apply inputs to get outputs
    """
    if rnn_cell == 'rnn':
        cell_fn = rnn.BasicRNNCell
    elif rnn_cell == 'gru':
        cell_fn = rnn.GRUCell
    elif rnn_cell == 'lstm':
        cell_fn = rnn.BasicLSTMCell
    elif rnn_cell == 'sru':
        cell_fn = SRUCell
    else:
        raise Exception("rnn type not supported: {}".format(rnn_cell))

    cells = []
    for _ in range(rnn_layer_num):
        cell = cell_fn(cell_size)
        cells.append(cell)
    cell = rnn.MultiRNNCell(cells)
    if bi:
        cells = []
        for _ in range(rnn_layer_num):
            cell = cell_fn(cell_size)
            cells.append(cell)
        cell2 = rnn.MultiRNNCell(cells)
        (output_fw, output_bw), _ =\
            tf.nn.bidirectional_dynamic_rnn(
                cell, cell2, inputs,
                sequence_length=sequence_length, dtype=tf.float32)
        outputs = tf.concat([output_fw, output_bw], axis=-1)
    else:
        outputs, _ = tf.nn.dynamic_rnn(
            cell,
            inputs, sequence_length=sequence_length, dtype=tf.float32)
    return outputs


def attn_add(outputs):
    """TODO: not implemented finish
    """
    cell_size = None
    weight = tf.get_variable(
        "attn_W",
        shape=[
            cell_size,
            cell_size],
        dtype=tf.float32)
    batch_size = tf.shape(outputs)[0]
    seq_len = tf.shape(outputs)[1]

    outputs_reshape = tf.reshape(outputs, [-1, cell_size])
    u = tf.matmul(outputs_reshape, weight)
    # u=tf.reshape(u,[batch_size,seq_len,-1])
    alpha = tf.reshape(tf.multiply(u, u), [batch_size, seq_len, -1])
    alpha = tf.nn.softmax(tf.reduce_sum(alpha, axis=-1))
    outputs = tf.reduce_sum(
        tf.multiply(
            tf.expand_dims(
                alpha, -1), outputs), -2)
    return outputs
