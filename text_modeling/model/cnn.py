#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:
#   Author:         xiahong xiahahaha01@gmail.com
#   Create:

"""cnn module

mainly Provide 2 type of cnn：
MultiFilterSizesCnn: normal cnn with multi size filter
MultiFilterSizesDebugCnn: explanative cnn

each class is also a function.
multi_filter_sizes_cnn is a function, which fist define a cnn layer,
    then return layer output.
"""
import tensorflow as tf


class MultiFilterSizesCnn(object):
    """cnn with Milti size filter.
    """

    def __init__(self,
                 sequence_length,
                 emb_length,
                 filter_sizes,
                 filter_nums,
                 activation,
                 name=None,
                 reuse=None,
                 # trainable=True,
                 # activity_regularizer=None,
                 padding='VALID',
                 pooling=True,
                 gated=False,
                 # de=False,
                 front_pad=0):
        self.sequence_length = sequence_length
        self.emb_length = emb_length
        self.filter_sizes = filter_sizes
        self.filter_nums = filter_nums
        self.activation = activation
        self.name = name
        self.reuse = reuse
        self.padding = padding
        self.pooling = pooling
        self.gated = gated
        self.front_pad = front_pad

    def apply(self, inputs):
        """layer apply, inputs->outputs.
        """
        pooled_outputs = []
        k = 0
        for filter_size, filter_num in zip(
                self.filter_sizes, self.filter_nums):
            if self.front_pad != 0:
                front_pad = filter_size - 1
            else:
                front_pad = 0
            if self.padding.lower() == "same" and filter_size >= 1:
                inputs_pad = tf.pad(
                    inputs, [[0, 0],
                             [front_pad, filter_size - 1 - front_pad],
                             [0, 0],
                             [0, 0]])
                # inputs_2=
                #   tf.pad(inputs, [[0,0],[0,filter_size-1],[0,0],[0,0]])
            else:
                inputs_pad = inputs
            print 'CNN on size', inputs_pad, filter_size, self.emb_length
            conv = tf.layers.conv2d(
                inputs_pad,
                filter_num,
                [filter_size, self.emb_length],
                name="{}-conv{}".format(self.name, k),
                reuse=self.reuse,
                padding="valid",
                activation=self.activation)
            k += 1
            if self.gated:
                conv_gated = tf.layers.conv2d(
                    inputs_pad,
                    filter_num,
                    [filter_size, self.emb_length],
                    name="{}-conv_gated_{}".format(self.name, k),
                    reuse=self.reuse,
                    padding="valid",
                    activation=self.activation)
                conv = conv * tf.tanh(conv_gated)
            print conv
            if not self.pooling:
                pooled_outputs.append(conv)
                continue
            if self.padding == 'same':
                size = self.sequence_length
            else:
                size = self.sequence_length - filter_size + 1
            print size
            pooled = tf.layers.max_pooling2d(
                conv,
                [size, 1],
                1)
            pooled_outputs.append(pooled)
        # Combine all the pooled features
        num_filters_total = sum(self.filter_nums)
        outputs = tf.concat(pooled_outputs, 3)
        print self.sequence_length, num_filters_total
        if not self.pooling:
            print 'xxxxxxxxx'
            outputs = tf.reshape(
                outputs, [-1, self.sequence_length, num_filters_total])
            print outputs
            return outputs

        outputs = tf.reshape(outputs, [-1, num_filters_total])
        # to get get W of [convs, filter_nums]
        print tf.shape(outputs)
        return outputs


def multi_filter_sizes_cnn(inputs,
                           sequence_length, emb_length,
                           filter_sizes, filter_nums,
                           activation=tf.nn.relu,
                           name=None,
                           reuse=None,
                           pooling=True,
                           padding="same",
                           gated=False,
                           front_pad=0):
    """multi filter size cnn layer apply.
    Returns:
        outputs of layer.apply(inputs).
    """
    layer = MultiFilterSizesCnn(
        sequence_length, emb_length,
        filter_sizes, filter_nums,
        activation=activation,
        name=name,
        reuse=reuse,
        pooling=pooling,
        padding=padding.lower(),
        gated=gated,
        front_pad=front_pad
    )
    return layer.apply(inputs)


class MultiFilterSizesDebugCnn(object):
    """multi size filer explabative cnn
    TODO: is it possible to merge into normal cnn
    """

    def __init__(self,
                 sequence_length,
                 emb_length,
                 filter_sizes,
                 filter_nums,
                 activation,
                 name=None,
                 # trainable=True,
                 # activity_regularizer=None,
                 # de=False
                 reuse=None):
        # super.(Dense,self).__init__(trainable=traiable, name=name,
        #        activity_regularizer=activity_regularizer)
        self.sequence_length = sequence_length
        self.emb_length = emb_length
        self.filter_sizes = filter_sizes
        self.filter_nums = filter_nums
        self.activation = activation
        self.name = name
        self.reuse = reuse

    def apply(self, inputs):
        """ layer apply, inputs -> outputs
        """
        pooled_outputs = []
        pooled_indexs = []
        k = 0
        for filter_size, filter_num in zip(
                self.filter_sizes, self.filter_nums):
            conv = tf.layers.conv2d(
                inputs,
                filter_num,
                [filter_size, self.emb_length],
                name="conv{}".format(k),
                reuse=self.reuse,
                activation=self.activation)
            pooled, pooled_index = tf.nn.max_pool_with_argmax(
                conv,
                [1, self.sequence_length - filter_size + 1, 1, 1],
                [1, 1, 1, 1],
                "VALID"
            )
            pooled_outputs.append(pooled)
            pooled_indexs.append(pooled_index)
            k += 1
        # Combine all the pooled features
        num_filters_total = sum(self.filter_nums)
        outputs = tf.concat(pooled_outputs, 3)
        pool = tf.reshape(outputs, [-1, num_filters_total])
        pooled_index = tf.concat(pooled_indexs, 3)
        pooled_index = tf.reshape(pooled_index, [-1, num_filters_total])
        return pool, pooled_index


def multi_filter_sizes_cnn_debug(inputs,
                                 sequence_length, emb_length,
                                 filter_sizes, filter_nums,
                                 activation=tf.nn.relu,
                                 name=None,
                                 reuse=None):
    """multi filter size debug/explanative cnn layer apply.
    Returns:
        outputs of layer.apply(inputs).
    """
    layer = MultiFilterSizesDebugCnn(
        sequence_length, emb_length,
        filter_sizes, filter_nums,
        activation=activation,
        name=name,
        reuse=reuse
    )
    return layer.apply(inputs)


def decode_index(z, a):
    a = a[::-1]
    y = []
    for x in a:
        y.append(z % x)
        z = z / x
    y = y[::-1]
    return y


def parse_text_cnn_index(
        filter_sizes,
        filter_nums,
        indexs,
        batch_size,
        seq_len):
    """get index from cnn
    """
    conv_shapes = [[batch_size, seq_len - size + 1, 1, num]
                   for size, num in zip(filter_sizes, filter_nums)]
    indexs = indexs.tolist()

    def get_loc(k, index, b):
        p = 0
        for p in range(len(filter_nums)):
            if sum(filter_nums[:p]) > k:
                break
        conv_shape = conv_shapes[p]
        loc = decode_index(index, conv_shape)
        loc = [b, [loc[1], loc[1] + filter_sizes[p]]]
        return loc
    locs = [[get_loc(k, index, b) for k, index in enumerate(a)]
            for b, a in enumerate(indexs)]
    return locs


class GraphCnn(object):
    # TODO: add graph-cnn
    pass
