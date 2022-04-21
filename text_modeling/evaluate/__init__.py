#!/usr/bin/env python
# -*- encoding=utf8

"""one line of summary
"""
from .evaluate import evaluate_clf, evaluate_seq_tag, evaluate_tag


def evaluate(objects, tag_exclusive, sess, model, test_data, tags):
    """eval model on test_data.

    evaluater is designed by config.object.
    """
    evaluatef = evaluate_clf
    if objects == "tag":
        if config.tag_exclusive:
            evaluatef = evaluate_clf
        else:
            evaluatef = evaluate_tag
    else:
        if tag_exclusive:
            evaluatef = evaluate_seq_tag
        else:
            pass
            # TODO:raise error
    score, test_data_metrics = evaluatef(sess, model, test_data, tags)
    print score, test_data_metrics
