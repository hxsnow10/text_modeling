#!/usr/bin/env python
# encoding=utf-8
import numpy as np
import tensorflow as tf
from tf_utils.predict import TFModel
from tf_utils.data import sequence_line_processing 
from tf_utils import load_config
from utils.byteify import byteify
from utils.wraps import tryfunc
from data_utils import load_data
import os,json
import sys
import argparse
from evaluate import evaluate_3 as evaluate
import re

os.chdir(sys.path[0])

try:
    from nlp.tokenizer.tok import zh_tok
except:
    pass
bad_ner=re.compile(u'[\u4e00-\u9fff]+[a-z]+', re.UNICODE)
 
def rsplit(text):
    toks=[x for x in zh_tok(text) if x!='\t']
    new=' '.join(toks)
    return new

class Classifier(object):

    def __init__(self, model_path, use_gpu=False):
        self.pos_norm = {line.strip().split('\t')[0]:line.strip().split('\t')[1] for line in open("/ceph_ai/xiahong/data/segment_corpus/pos_tokens/pos_norm.txt")}
        print self.pos_norm
        model_dir = os.path.dirname(model_path)
        config=load_config(model_dir+'/pos_config.py', ceph_path="/ceph_ai", branch="develop")
        print os.path.basename(model_path)
        info = open( os.path.join(os.path.dirname(model_path), "info.txt") , "r").read()
        info = byteify( json.loads(info) )
        self.pre=lambda x:x.strip()
        session_conf = tf.ConfigProto(
              device_count = {'CPU': 1, 'GPU':1},
              intra_op_parallelism_threads=1,
              inter_op_parallelism_threads=1,
              allow_soft_placement=True,
              log_device_placement=False)
        sess=tf.Session(config=session_conf)
        data=load_data(config.data_config)
        # score,dev_data_metrics = evaluate(sess,model,data.dev_data,data.tags)
        self.tags = data.tags
        self.info = info
        config.seq_len=30
        self.seq_ps ={name: sequence_line_processing(config_.vocab, return_length=True, seq_len=config_.seq_len, split=' ', \
            sub_words=config_.sub_vocab, char_len=10, use_char_length=False) for name,config_ in config.data_config.task2configs.iteritems()} 
        print self.tags, self.seq_ps
        self.tf_model=TFModel(sess,model_path, info)

    def predict(self, line, task_name):
        tensors=[np.array(x) for x in self.seq_ps[task_name](self.pre(line.strip()))]+[1.0]
        names=['input_zh_x', 'input_zh_x_sub', 'input_zh_x_length', 'dropout']
        data={tf.get_default_graph().get_tensor_by_name(self.info[task_name]["inputs"][name]):tensor for name,tensor in zip(names, tensors)}
        outputs = self.info[task_name]["outputs"].values()
        lengths, tags=self.tf_model.predict(data, outputs)
        rval=[self.tags[task_name][i] for i in tags[0][:lengths[0]] ] 

        norm_rval = [self.pos_norm.get(p.lower(),p.lower()) for p in rval]
        return norm_rval

class Classifier_v2(object):

    def __init__(self, model_path, use_gpu=False):
        word_emb=load_vec(config.words_vec)
        char_emb=load_vec(config.chars_vec)
        with tf.Session(config=config.session_conf) as sess:
            # use tf.name_scope to manager variable_names
            model=TextModel(
                configp=config,
                vocab_size=len(data.words),
                num_classes=len(data.tags), 
                init_emb=word_emb,
                sub_init_emb=char_emb,
                reuse=False,# to use when several model share parameters
                debug=False,
            
                # debug model only work for cnn, tell how much score every ngram contribute to every label
                class_weights=data.class_weights,
                mode='train')
            model.restore(sess, model_path)
        
if __name__=="__main__":
    model_path = '/ceph_ai/xiahong/RESULT/ner/develop/2019-03-24-17v2/model/text_model=None-cnn_layer_num=None/model-78235'
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--model_path", type=str, default="")
    parser.add_argument("-m", "--method", type=str, default="demo")
    parser.add_argument("-fi", "--test_path", type=str, default="querys")
    parser.add_argument("-fo", "--output_path", type=str, default="result.txt")
    parser.add_argument("-fg", "--gold_path", type=str, default="")
    args = parser.parse_args()
    args.model_path = args.model_path or model_path
    tag_model=Classifier(args.model_path)
    if args.method=="demo":
        while True:
            try:
                line=raw_input("Input Inputs:")
                print 'RESULT=', json.dumps(tag_model.predict(line, "pos"), ensure_ascii=False)
            except Exception,e:
                print e
    elif args.method=="test":
        import time
        ii=open(args.test_path, "r")
        st=time.time()
        oo = open(args.output_path, "w")
        for k,line in enumerate(ii):
            line=line.strip()
            tokens = line.split()
            tags=['error',]*len(tokens)
            try:
                # line=line.decode('gbk', 'replace').encode('utf-8')
                # print json.dumps(tag_model.get_ner(line, 'seg'), ensure_ascii=False)
                # oo.write(json.dumps(tag_model.get_ner(line, 'str'), ensure_ascii=False)+'\n')
                tags = tag_model.predict(line, 'pos')
            except Exception,e: 
                import traceback
                traceback.print_exc()
                pass
            oo.write(' '.join(w+'/'+p for w,p in zip(tokens,tags))+'\n')
            if k%2000==0:
                print (time.time()-st)/2000
                st=time.time()
           
