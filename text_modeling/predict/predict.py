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
import tensorflow as tf


class TFModel():

    def __init__(self, sess, model_path, tasks):
        print tasks
        self.sess = sess
        saver = tf.train.import_meta_graph(
            "{}.meta".format(model_path), clear_devices=True)
        self.init = tf.global_variables_initializer()
        sess.run(self.init)
        saver.restore(self.sess, model_path)
        print 'MODEL LOADED SCCESSFULLY'

    def predict(self, fd, outputs):
        rval = self.sess.run(outputs, feed_dict=fd)
        return rval
