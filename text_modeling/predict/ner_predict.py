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
import re
import sys

import numpy as np
import tensorflow as tf

from data import sequence_line_processing
from data_utils import load_data
from evaluate import evaluate_3 as evaluate
from predict import TFModel
from utils import load_config
from utils.byteify import byteify
from utils.wraps import tryfunc

os.chdir(sys.path[0])

try:
    from nlp.tokenizer.tok import zh_tok
except BaseException:
    pass
bad_ner = re.compile(u'[\u4e00-\u9fff]+[a-z]+', re.UNICODE)


def rsplit(text):
    toks = [x for x in zh_tok(text) if x != '\t']
    new = ' '.join(toks)
    return new


class Classifier(object):

    def __init__(self, model_path, use_gpu=False):
        model_dir = os.path.dirname(model_path)
        config = load_config(
            model_dir + '/ner_config.py',
            ceph_path="/ceph_ai",
            branch="develop")
        print os.path.basename(model_path)
        info = open(
            os.path.join(
                os.path.dirname(model_path),
                "info.txt"),
            "r").read()
        info = byteify(json.loads(info))
        self.pre = lambda x: x.strip()
        session_conf = tf.ConfigProto(
            device_count={'CPU': 1, 'GPU': 1},
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1,
            allow_soft_placement=True,
            log_device_placement=False)
        sess = tf.Session(config=session_conf)
        data = load_data(config.data_config)
        # score,dev_data_metrics = evaluate(sess,model,data.dev_data,data.tags)
        self.tags = data.tags
        self.task2config = config.data_config.task2configs
        self.info = info
        config.seq_len = 20
        self.seq_ps = {
            name: sequence_line_processing(
                config_.vocab,
                return_length=True,
                seq_len=config_.seq_len,
                split=config_.tok,
                sub_words=None,
                char_len=10) for name,
            config_ in config.data_config.task2configs.iteritems()}
        print self.tags, self.seq_ps
        self.tf_model = TFModel(sess, model_path, info)

    def predict(self, line, task_name):
        tensors = [
            np.array(x) for x in self.seq_ps[task_name](
                self.pre(line))] + [1.0]
        names = self.task2config[task_name].names[1:] + ["dropout"]
        data = {
            tf.get_default_graph().get_tensor_by_name(
                self.info[task_name]["inputs"][name]): tensor for name, tensor in zip(
                names, tensors)}
        outputs = self.info[task_name]["outputs"].values()
        v = self.tf_model.predict(data, outputs)[1][0]
        rval = [self.tags[task_name][i] for i in v]
        return rval

    def get_ner(self, text, task_name="str", type="str"):
        lines = text.split(' ')
        rvals = [self.get_ner_(line, task_name=task_name) for line in lines]
        if type == "str":
            r_lines = []
            for line, rval in zip(lines, rvals):
                if not rval:
                    r_lines.append(line)
                    continue
                for state, word, loc in rval[::-1]:
                    line = line[:loc[0]] + ' ' + word + \
                        '/' + state + ' ' + line[loc[1]:]
                r_lines.append(line.strip())
            rval = ' '.join(r_lines)
        else:
            rval = sum(rvals, [])
        return rval

    @tryfunc([])
    def get_ner_(self, text, task_name):
        text1 = text.decode('utf-8', 'replace')
        while text1 and (text1[-1] >= "a" and text1[-1]
                         <= "z" or text1[-1] == "'"):
            text1 = text1[:-1]
        if not text1.strip():
            return []
        labels = self.predict(text, task_name)[:len(text1)]
        assert len(text1) >= len(labels)
        state, word, start = None, '', 0
        i = 0
        rval = []
        for k, (ch, tag) in enumerate(zip(text1, labels)):
            # print ch, tag
            if tag == 'O' or tag == '</pad>' or tag == '</s>' or ch in ['\n']:
                if state and word:
                    rval.append(
                        (state, word, (start, start + len(word.encode('utf-8')))))
                state, word = None, ''
            else:
                l, state_ = tag.split('-')
                if state:
                    if state_ == state and l != 'E' and l != 'B':
                        word += ch
                    elif l == 'E':
                        word += ch
                        word = word.strip()
                        rval.append(
                            (state, word, (start, start + len(word.encode('utf-8')))))
                        state, word = None, ''
                    else:
                        rval.append(
                            (state, word, (start, start + len(word.encode('utf-8')))))
                        state, word, start = state_, ch, i
                else:
                    state, word, start = state_, ch, i
            i += len(ch.encode('utf-8'))
        if state and word:
            rval.append(
                (state, word, (start, start + len(word.encode('utf-8')))))
        rval = [
            (state,
             word.strip().encode('utf-8'),
             loc) for state,
            word,
            loc in rval if word.strip() and ' ' not in word and not bad_ner.match(word)]
        for i, (state, word, loc) in enumerate(rval):
            if '京东' in word and word != '京东':
                rval[i] = (
                    state,
                    '京东',
                    (loc[0] +
                     word.index("京东"),
                        loc[0] +
                        word.index("京东") +
                        6))
        return rval


class Classifier_v2(object):

    def __init__(self, model_path, use_gpu=False):
        word_emb = load_vec(config.words_vec)
        char_emb = load_vec(config.chars_vec)
        with tf.Session(config=config.session_conf) as sess:
            # use tf.name_scope to manager variable_names
            model = TextModel(
                configp=config,
                vocab_size=len(data.words),
                num_classes=len(data.tags),
                init_emb=word_emb,
                sub_init_emb=char_emb,
                reuse=False,  # to use when several model share parameters
                debug=False,

                # debug model only work for cnn, tell how much score every
                # ngram contribute to every label
                class_weights=data.class_weights,
                mode='train')
            model.restore(sess, model_path)


if __name__ == "__main__":
    model_path = "/ceph_ai/xiahong/RESULT/ner/develop/2018-06-27-17-modeltext_model=cnn-cnn_layer_num=2/model"
    model_path = "/ceph_ai/xiahong/RESULT/ner/develop/2018-07-01-21/model/text_model=cnn-cnn_layer_num=2/model"
    model_path = "/ceph_ai/xiahong/RESULT/ner/develop/2018-07-02-00/model/text_model=birnn-cnn_layer_num=None/model"
    model_path = "/ceph_ai/xiahong/RESULT/ner/develop/2018-07-14-17v2/model/text_model=None-cnn_layer_num=None/model-146509"
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--model_path", type=str, default="")
    parser.add_argument("-m", "--method", type=str, default="eval")
    parser.add_argument("-fi", "--test_path", type=str, default="querys")
    parser.add_argument("-fo", "--output_path", type=str, default="result.txt")
    parser.add_argument("-fg", "--gold_path", type=str, default="")
    args = parser.parse_args()
    args.model_path = args.model_path or model_path
    tag_model = Classifier(args.model_path)
    if args.method == "demo":
        while True:
            line = raw_input("Input Inputs:")
            print 'RESULT=', json.dumps(
                tag_model.get_ner(
                    line, "seg"), ensure_ascii=False)
            print 'RESULT=', json.dumps(
                tag_model.get_ner(
                    line, "pos"), ensure_ascii=False)
            print 'RESULT=', json.dumps(
                tag_model.get_ner(
                    line, "ner"), ensure_ascii=False)
    elif args.method == "test":
        import time
        ii = open(args.test_path, "r")
        st = time.time()
        oo = open(args.output_path, "w")
        for k, line in enumerate(ii):
            line = line.strip()
            try:
                # line=line.decode('gbk', 'replace').encode('utf-8')
                print json.dumps(
                    tag_model.get_ner(
                        line, 'seg'), ensure_ascii=False)
                # oo.write(json.dumps(tag_model.get_ner(line, 'str'), ensure_ascii=False)+'\n')
                oo.write(tag_model.get_ner(line, 'seg') + '\n')
                if k % 2000 == 0:
                    print (time.time() - st) / 2000
                    raw_input("")
                    st = time.time()
            except Exception as e:
                import traceback
                traceback.print_exc()
                raw_input('XXXXXXXXXXXX')
                pass
