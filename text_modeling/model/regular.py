#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       regular
#   Author:         xiahong xiahahaha01@gmail.com
#   Create:         19/04/2022
#   Description:    ---
"""Regularizer Loss.

Goodfellow, Ian, Y. Bengio, and A. Courville. 2016. ‘Regularization for Deep Learning’. Deep Learning, 216–61.

TF has many api to get it..
* low level: like tf.reduce_sum(tf.abs(v))
* middle level: like tf.nn.l2_loss
* high level: tf.keras.regularizers.Regularizer, https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/Regularizer

dropout: Inputs elements are randomly set to zero (and the other elements are rescaled).
More precisely: With probability rate elements of x are set to 0. The remaining elements are scaled up by 1.0 / (1 - rate), so that the expected value is preserved.
* tf.nn.dropout
* tf.keras.layers.Dropout
* Layer Embed
TODO: 在config中加入dropout,最早的重构后丢死了

where to apply dropout:
    * 原则上每层都可以加，但要控制好参数
    * 前面的层drop率应该低一些，避免噪声太大
    * after embeeding layer:  I don't think it's good, or we can set ratio is small
    * after cnn-pooling layer: good
    * bafore cnn-pooling layer: as pooling itself can reduce noise, this may not that infuenced.
"""

import argparse
import os
import sys

import tensorflow as tf


def l2_loss(l2_lambda):
    l2_loss = l2_lambda * tf.add_n([tf.cast(tf.nn.l2_loss(v), tf.float32)
                                    for v in tf.trainable_variables() if 'bias' not in v.name])
    return l2_loss


def l1_loss(l1_lambda):
    l1_loss = l1_lambda * tf.add_n([tf.cast(tf.reduce_sum(tf.abs(v)), tf.float32)
                                    for v in tf.trainable_variables() if 'bias' not in v.name])
    return l1_loss
