#!/usr/bin/env python
# encoding=utf-8
import sys
import os
import json
import argparse
import time

import tensorflow as tf
import numpy as np

print "当前目录:", os.getcwd()
os.chdir(sys.path[0])
print "修改后当前目录:", os.getcwd()

from utils.word2vec import load_w2v
from utils.base import get_vocab
from utils.base import get_func_args

from tf_utils.model import multi_filter_sizes_cnn 
from tf_utils import load_config
from tf_utils import check_dir
from tf_utils.profile import *
from shutil import copy
from data_utils import load_data

from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder

from model import TextModel
from evaluate import *
config=None

'''
assumpation:
* load_d train_data, dev_data
* model has init, loss, train_op, step_summaries, saver
'''
def now():
    return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))

def test(sess, model, test_data, tags):
    if config.objects=="tag":
        if config.tag_exclusive:
            evaluate=evaluate_clf
        else:
            evaluate=evaluate_tag
    else:
        if config.tag_exclusive:
            evaluate=evaluate_seq_tag
    score,test_data_metrics = evaluate(sess,model,test_data,tags)
    print score, test_data_metrics
    
        
def train(sess, model, train_data, dev_datas=None, test_datas=None, summary_writers={}, tags=None, config = None):
    oo = open(config.top_log_path, "a+") 
    oo.write(now()+'\t'+config.model_dir+'\tstart_train\n')
    oo.close()
    best_dev,best_test,really_best_test = 0, 0, 0 
    best_dev_metrics, best_test_metrics, really_best_test_metrics = None, None, None

    profiler = model_analyzer.Profiler(graph=sess.graph)
    run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    
    show_params(profiler)

    step=0
    model.train_saver.save(sess, config.model_path, global_step=0)
    model.train_saver.save(sess, config.model_path)
    # score,dev_data_metrics = evaluate(sess,model,dev_data,tags)
    def eval(epoch,step, datas):
        for task_name, dev_data in datas.iteritems():
            oo = open(config.top_log_path, "a+") 
            oo.write('\t'+now()+'\t'+config.model_dir+'\tstart_eval'+str(epoch)+'\n')
            oo.close()
            if config.task_type[task_name]=="tag_exclusive":
                evaluate=evaluate_clf
            elif config.task_type[task_name]=="tag_noexclusive":
                evaluate=evaluate_tag
            elif config.task_type[task_name]=="seq_tag":
                evaluate=evaluate_seq_tag
            score,metrics = evaluate(sess,model,task_name, dev_data, target_names = tags[task_name])

            def add_summary(writer, metric, step):
                for name,value in metric.iteritems():
                    summary = tf.Summary(value=[                         
                        tf.Summary.Value(tag=name, simple_value=value),   
                        ])
                    writer.add_summary(summary, global_step=step)
            add_summary(summary_writers[(task_name,'dev')], metrics, step)
            model.train_saver.save(sess, config.model_path, global_step=step)
        return score, metrics
    for epoch in range(config.epoch_num):
        if epoch%config.decay_steps==0:
            import math
            for lr in config.lrs:
                sess.run(tf.assign(
                    getattr(model, lr), 
                    config.start_learning_rate*math.pow(config.decay_rate, epoch/config.decay_steps)
                    ))
                    
        for k,(task_name,inputs) in enumerate(train_data):
            inputs["dropout"]=1.0
            # print "get in {}, task_name = {}".format(step, task_name)
            fd = {getattr(model, name):inputs[name] for name in model.train_task2io[task_name]["inputs"].keys()}
            out = model.train_task2io[task_name]["outputs"].values()
            if step%config.summary_steps!=0:
                out_v=sess.run(out, feed_dict=fd)
            else:
                out_v=sess.run(\
                        #out, feed_dict=fd,\
                        out+[model.step_summaries], feed_dict=fd,\
                        options=run_options,
                        run_metadata=run_metadata)
                summary_writers[(task_name,'train')].add_summary(out_v[-1], step)
                summary_writers[(task_name,'train')].add_run_metadata(run_metadata, "train"+str(step))
            # print "epoch={}\ttask={}\tstep={}\tglobal_step={}\tout_v={}".format(epoch, task_name, k, step ,out_v)
            step+=1
            if step > 0 and step%10000==1:
                eval(epoch, step, dev_datas)
                
        if dev_datas:
            dev_f1, dev_metrics = eval(epoch, step, dev_datas)
            test_f1, test_metrics = eval(epoch, step, test_datas)
            if dev_f1>best_dev:
                best_dev = dev_f1
                best_test = test_f1
                best_dev_metrics = dev_metrics
                best_test_metrics = test_metrics
            if test_f1>really_best_test:
                really_best_test = test_f1
                really_best_test_metrics = test_metrics
            print '-'*40
            print 'NOW BEST DEV = ', best_dev_metrics
            print 'AND TEST = ', best_test_metrics
            print 'AND REALLY TEST = ', really_best_test_metrics
            print '-'*40

        else:
            model.train_saver.save(sess, config.model_path, global_step=step)
            
def main(mode):
    data_config, model_config, train_config = config.data_config, config.model_config, config.train_config
    check_dir(train_config.summary_dir, train_config.ask_for_del, train_config.restore)
    check_dir(train_config.model_dir, train_config.ask_for_del, train_config.restore)
    copy(args.config_path, train_config.summary_dir) 
    copy(args.config_path, train_config.model_dir)  
    
    data = load_data(data_config)
    
    with tf.Session(config=train_config.session_conf) as sess:
        # use tf.name_scope to manager variable_names
        model=TextModel(sess, model_config)
        model.inits(sess, train_config.restore)
        model.save_info(train_config.model_dir)
         
        summary_writers = {
            (task_name,sub_path):tf.summary.FileWriter(os.path.join(train_config.summary_dir,task_name,sub_path), sess.graph, flush_secs=5)
                for task_name in model_config.train_task2io for sub_path in ['train','dev']}
        
        if mode=="train":  
            train(sess, model,
                data.train_data, data.dev_data, data.test_data, 
                summary_writers=summary_writers,
                config = train_config, tags=data.tags)
        else:
            test(sess, model, data.dev_data, data.tags.vocab)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    parser.add_argument("-r", "--restore", type=str, default="")
    parser.add_argument("-c", "--ceph_path", type=str, default="/ceph_ai")
    parser.add_argument("-b", "--branch", type=str, default="test")
    parser.add_argument("-m", "--mode", type=str, default="train")
    args = parser.parse_args()
    global config
    config=load_config(args.config_path, branch=args.branch)
    config.ceph_path = args.ceph_path
    config.train_config.restore=args.restore
    main(args.mode)
