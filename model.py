# encoding=utf-8
import sys
import os
from os import makedirs
from shutil import rmtree
import json
from pprint import pformat

import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report,f1_score,precision_score,recall_score

from utils.word2vec import load_w2v
from tf_utils.model import multi_filter_sizes_cnn_debug, multi_filter_sizes_cnn
from tf_utils.model.subnet import subnets
from tf_utils.model.rnn import  rnn_func
from tf_utils import load_config
from utils.base import get_vocab
from utils.base import get_func_args


import argparse
import json

config=None

'''
A general TextModel controled by parameter and config
'''
class TextModel(object):
    def __init__(self, sess, config):
        self.sess = sess
        self.config= config
        self.build_placeholders(config.placeholders)
        self.build_net(config.net)
        self.build_others()

    def build_placeholders(self, placeholders):
        for name,dtype,shape in placeholders:
            setattr(self, name, tf.placeholder(tf.int64, shape, name=name)) 
    
    def build_net(self, net):
        for inputs, func_name, func_args, scope, outputs in net:
            with tf.variable_scope(scope):
                print '-'*80
                print "INPUTS=\t", list(zip(inputs, [getattr(self, name) for name in inputs]))
                print "FUNC_NAME=\t",func_name, subnets.get(func_name)
                print "FUNC_ARGS=\t",func_args
                print "OUTPUTS=\t",outputs
                args= {name:getattr(func_args, name) for name in dir(func_args) if name[0]!='_'}
                print args.keys()
                sub_net = subnets.get(func_name)(**args)
                tmp = sub_net(*[getattr(self, name) for name in inputs])
                if type(tmp)!=list and type(tmp)!=tuple: tmp = (tmp,)
                print tmp
                for i,name in enumerate(outputs):
                    print i,name
                    setattr(self, name, tmp[i])
                    print "output---->", name, tmp[i]

    def build_others(self):
        self.step_summaries = tf.summary.merge_all()   
        self.init = tf.global_variables_initializer()
        self.all_vars=list(set(
            (tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)+
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="share"))))
        self.train_vars=[x for x in self.all_vars if x in tf.trainable_variables()]
        self.all_saver=tf.train.Saver(self.all_vars)
        self.train_saver = tf.train.Saver(self.train_vars)
        # print 'ALL VAR:\n\t', '\n\t'.join(str(x) for x in self.all_saver._var_list)
        print 'TRAIN VAR:\n\t', '\n\t'.join(str(x) for x in self.train_saver._var_list)
        
        self.train_task2io = \
            {task:{part:{name:getattr(self, name) for name in name_list} for part,name_list in io.iteritems()} 
                for task,io in self.config.train_task2io.iteritems()}
        self.predict_task2io = \
            {task:{part:{name:getattr(self, name) for name in name_list} for part,name_list in io.iteritems()} 
                for task,io in self.config.predict_task2io.iteritems()}
        def indent(text, indent=8):
            fstring = ' ' * indent + '{}' 
            return ''.join([fstring.format(l) for l in text.splitlines(True)])
        print "TRAIN_TASK2IO:\n", indent(pformat(self.train_task2io))
        print "PREDICT_TASK2IO:\n", indent(pformat(self.predict_task2io))

    def save_info(self, model_dir):
        """save some import info"""
        info = {task:{part:{name:self.predict_task2io[task][part][name].name for name in self.predict_task2io[task][part]} 
                    for part in self.predict_task2io[task]}
                        for task in self.predict_task2io}
        oo=open(os.path.join(model_dir, "info.txt"), "w")
        oo.write( json.dumps(info, ensure_ascii=False))
        oo.close()

    def inits(self, sess, restore):
        sess.run(self.init)
        if restore:
            try:
                self.train_saver.restore(sess, restore)
                print "reload model"
            except Exception,e:
                print e
                print "reload model fail"
                quit()

if __name__ == "__main__": 
    from tf_utils import load_config
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default=".")
    args = parser.parse_args()
    config = load_config(args.config_path)
    model = TextModel(None, config.model_config)
