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
import argparse
import json
import os
from pprint import pformat

import tensorflow as tf

from subnet import subnets
from utils import load_config

config = None

'''
A general Model controled by parameter and config
'''


class GeneralModel(object):

    def __init__(
            self,
            sess,
            placeholders,
            net,
            train_task2io,
            predict_task2io):
        """
        Args:
            sess : tf sess
            placeholders: input for net
            net: net
            train_task2io
            predict_task2io
        """
        self.sess = sess
        self.placeholders = placeholders
        self.net = net
        self.train_task2io = train_task2io
        self.predict_task2io = predict_task2io

        self.build_placeholders(placeholders)
        self.build_net(net)
        self.build_others()

    def build_placeholders(self, placeholders):
        for name, dtype, shape in placeholders:
            setattr(self, name, tf.placeholder(tf.int64, shape, name=name))

    def build_net(self, net):
        for inputs, func_name, func_args, scope, outputs in net:
            with tf.variable_scope(scope):
                print '-' * 80
                print "INPUTS=\t", list(
                    zip(inputs, [getattr(self, name) for name in inputs]))
                print "FUNC_NAME=\t", func_name, subnets.get(func_name)
                print "FUNC_ARGS=\t", func_args
                print "OUTPUTS=\t", outputs
                args = {name: getattr(func_args, name)
                        for name in dir(func_args) if name[0] != '_'}
                print args.keys()
                sub_net = subnets.get(func_name)(**args)
                tmp = sub_net(*[getattr(self, name) for name in inputs])
                if not isinstance(tmp, list) and not isinstance(tmp, tuple):
                    tmp = (tmp,)
                print tmp
                for i, name in enumerate(outputs):
                    print i, name
                    setattr(self, name, tmp[i])
                    print "output---->", name, tmp[i]

    def build_others(self):
        self.step_summaries = tf.summary.merge_all()
        self.init = tf.global_variables_initializer()
        self.all_vars = list(set(
            (tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) +
             tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="share"))))
        self.train_vars = [
            x for x in self.all_vars if x in tf.trainable_variables()]
        self.all_saver = tf.train.Saver(self.all_vars)
        self.train_saver = tf.train.Saver(self.train_vars)
        # print 'ALL VAR:\n\t', '\n\t'.join(str(x) for x in self.all_saver._var_list)
        print 'TRAIN VAR:\n\t', '\n\t'.join(
            str(x) for x in self.train_saver._var_list)

        self.train_task2io = {
            task: {
                part: {
                    name: getattr(
                        self,
                        name) for name in name_list} for part,
                name_list in io.iteritems()} for task,
            io in self.train_task2io.iteritems()}
        self.predict_task2io = {
            task: {
                part: {
                    name: getattr(
                        self,
                        name) for name in name_list} for part,
                name_list in io.iteritems()} for task,
            io in self.predict_task2io.iteritems()}

        def indent(text, indent=8):
            fstring = ' ' * indent + '{}'
            return ''.join([fstring.format(ln)
                           for ln in text.splitlines(True)])
        print "TRAIN_TASK2IO:\n", indent(pformat(self.train_task2io))
        print "PREDICT_TASK2IO:\n", indent(pformat(self.predict_task2io))

    def save_info(self, model_dir):
        """save some import info"""
        info = {task: {part: {name: self.predict_task2io[task][part][name].name
                              for name in self.predict_task2io[task][part]}
                       for part in self.predict_task2io[task]}
                for task in self.predict_task2io}
        oo = open(os.path.join(model_dir, "info.txt"), "w")
        oo.write(json.dumps(info, ensure_ascii=False))
        oo.close()

    def inits(self, sess, restore):
        sess.run(self.init)
        if restore:
            try:
                self.train_saver.restore(sess, restore)
                print "reload model"
            except Exception as e:
                print e
                print "reload model fail"
                quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default=".")
    args = parser.parse_args()
    config = load_config(args.config_path)
    model = GeneralModel(None, config.model_config)
