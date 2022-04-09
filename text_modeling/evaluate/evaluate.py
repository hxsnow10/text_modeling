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
import numpy as np
from sklearn.metrics import (classification_report, f1_score, precision_score,
                             recall_score)

def evaluate_clf(
        sess,
        model,
        eval_data,
        target_names=None,
        # restore=False,
        dropout=True):
    """ evaluate score for classification"""
    total_y, total_predict_y = [], []
    print 'start evaluate...'
    print target_names
    for inputs in eval_data:
        if dropout:
            inputs = inputs + [1]
        fd = dict(zip(model.inputs, inputs))
        predict_y =\
            sess.run(model.predictions, feed_dict=fd)
        total_y = total_y + [np.argmax(inputs[0], -1)]
        total_predict_y = total_predict_y + [predict_y]
    total_y = np.concatenate(total_y, 0)
    total_predict_y = np.concatenate(total_predict_y, 0)
    # print total_y.shape, total_predict_y.shape
    print classification_report(total_y, total_predict_y)
    p, r, f = precision_score(total_y, total_predict_y, average='weighted'),\
        recall_score(total_y, total_predict_y, average='weighted'),\
        f1_score(total_y, total_predict_y, average='weighted')
    print p, r, f
    return f, {"precision": p, "recall": r, "f1": f}


def evaluate_tag(
        sess,
        model,
        eval_data,
        target_names=None,
        # restore=False,
        dropout=True):
    total_y, total_predict_y = [], []
    print 'start evaluate...'
    print target_names
    for inputs in eval_data:
        if dropout:
            inputs = inputs + [1.0]
        fd = dict(zip(model.inputs, inputs))
        predict_y =\
            sess.run(model.predictions, feed_dict=fd)
        total_y = total_y + [inputs[0]]
        total_predict_y = total_predict_y + [predict_y]
    total_y = np.concatenate(total_y, 0)
    total_predict_y = np.concatenate(total_predict_y, 0)
    # print total_y.shape, total_predict_y.shape
    # print classification_report(total_y, total_predict_y, target_names=target_names)
    p, r, f = precision_score(total_y, total_predict_y, average='weighted'),\
        recall_score(total_y, total_predict_y, average='weighted'),\
        f1_score(total_y, total_predict_y, average='weighted')
    print f, p, r
    return f, {"precision": p, "recall": r, "f1": f}


def evaluate_seq_tag(
        sess,
        model,
        task_name,
        eval_data,
        target_names=None,
        # restore=False,
        dropout=True):
    # 获得Y_hat labels=[batch_size, seq_len], length=[batch_size]
    # 获得Y y_true
    # 摊平连接，evaluate即与一般的分类等价
    total_y, total_predict_y = [], []

    def process(labels, lengths):
        tags = []
        for seq_tag, length in zip(labels.tolist(), lengths.tolist()):
            tags += seq_tag[:length]
        return tags

    def get_seq_for_segment(y, ignore=None):
        ignore = ignore or ['</s>', '</pad>', 'O']
        y = y.tolist()
        if not y:
            return []

        def get_tag(x):
            x = target_names[x]
            if 'B' in x:
                return 'B'
            if 'M' in x:
                return 'M'
            if 'E' in x:
                return 'E'
            if 'S' in x:
                return 'S'
            return 'O'
        y = [get_tag(x) for x in y]
        rval = []
        start = 0
        for i, tag in enumerate(y):
            if tag == 'B':
                rval.append((start, i))
                start = i
            elif tag == 'M':
                continue
            elif tag == 'S' or tag in ignore:
                rval.append((start, i))
                rval.append((i, i + 1))
                start = i + 1
                continue
            elif tag == 'E':
                rval.append((start, i + 1))
                start = i + 1
                continue
        if start < len(y):
            rval.append((start, len(y)))
        rval = [x for x in rval if x[1] - x[0] >= 1]
        return rval

    t_num, p_num, tp_num = 0, 0, 0
    total_time, num = 0, 0
    for _, inputs in enumerate(eval_data):
        if dropout:
            inputs["dropout"] = 1.0
        fd = {
            getattr(
                model,
                name): inputs[name] for name in model.predict_task2io[task_name]["inputs"].keys()}
        out = model.predict_task2io[task_name]["outputs"].values()
        import time
        st = time.time()
        lengths, predict_y =\
            sess.run(out, feed_dict=fd)
        et = time.time()
        total_time += (et - st)
        num += 1
        name = [name for name in inputs.keys() if '_y' in name][0]
        input_y = inputs[name]
        for i in range(len(lengths.tolist())):
            y = get_seq_for_segment(input_y[i][:int(lengths[i])])
            p_y = get_seq_for_segment(predict_y[i][:int(lengths[i])])
            t_num += len(y)
            p_num += len(p_y)
            tp_num += len(set(y) & set(p_y))
        total_y = total_y + process(input_y, lengths)
        total_predict_y = total_predict_y + process(predict_y, lengths)
    print "avg time = ", total_time/num
    seq_p, seq_r = tp_num * 1.0 / p_num, tp_num * 1.0 / t_num
    # seq_p,seq_r = 0, 0

    total_y = np.array(total_y)
    total_predict_y = np.array(total_predict_y)
    p, r, f = precision_score(total_y, total_predict_y, average='weighted'),\
        recall_score(total_y, total_predict_y, average='weighted'),\
        f1_score(total_y, total_predict_y, average='weighted')
    # TODO p/r/f for ner
    seq_f = (2 * seq_p * seq_r) / (seq_p + seq_r + 0.0001)
    print f, p, r, seq_p, seq_r, seq_f
    return seq_f, {"precision": p, "recall": r, "f1": f, "seq_p": seq_p,
                   "seq_r": seq_r, "seq_f1": (2 * seq_p * seq_r) / (seq_p + seq_r + 0.0001)}
