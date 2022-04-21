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
import inspect
import os
from collections import OrderedDict, defaultdict
from os import makedirs
from shutil import rmtree


def now():
    return time.strftime("%Y%m%d%H", time.localtime(time.time()))


def get_emb(path, words_num, vec_size=2):
    emb = []
    for line in open(path):
        vals = [float(x) for x in line.strip().split(' ')]
        emb.append(vals)
    rval = np.array(emb + [[0, ] * vec_size, ] *
                    (words_num - len(emb)), dtype=np.float32)
    return rval


def get_func_args(depth=1):
    """get variable of tempolary env,
        if depth=1, this is equal to func args.

    Args:
        depth: depth of variable to return
    Return:
        map of args and values
    """
    frame = inspect.currentframe(depth)
    args, _, _, values = inspect.getargvalues(frame)
    rval = {}
    for d in [args, {}, {}]:
        for arg in d:
            rval[arg] = values.get(arg, None)
    return rval


def dict_reverse(d):
    new_d = {}
    for key in d:
        value = d[key]
        new_d[value] = key
    return new_d


def dict_sub(d, s):
    return {k: d[k] for k in s if k in d}


def leave_values(d, keys):
    """leave values
    """
    if isinstance(d, type([1, 2, 3])):
        rval = []
        for k in d:
            rval += leave_values(k, keys)
    elif isinstance(d, type({1: 2})):
        rval = []
        for key in d:
            if key in keys:
                rval.append(d[key])
            else:
                rval += leave_values(d[key], keys)
    else:
        rval = []
    return rval


def get_vocab(vocab_path):
    ii = open(vocab_path, 'r')
    vocab, reverse_vocab = OrderedDict(), OrderedDict()
    for line in ii:
        word = line.strip()
        k = len(vocab)
        reverse_vocab[word] = k
        vocab[k] = word
    return vocab, reverse_vocab


class OrderedDefaultDict(OrderedDict, defaultdict):
    def __init__(self, default_factory=None, *args, **kwargs):
        # in python3 you can omit the args to super
        super(OrderedDefaultDict, self).__init__(*args, **kwargs)
        self.default_factory = default_factory


def get_file_paths(data_dir):
    data_paths = []
    for dir_path, _, files in os.walk(data_dir):
        for file_name in files:
            path = os.path.join(dir_path, file_name)
            data_paths.append(path)
    return data_paths


def check_dir(dir_path, ask_for_del=False, restore=True):
    """ check whether is a empty directoty
    """
    if os.path.exists(dir_path):
        if restore:
            return
        y = ''
        if ask_for_del:
            y = raw_input('new empty {}? y/n:'.format(dir_path))
        if y.strip() == 'y' or not ask_for_del:
            rmtree(dir_path)
        else:
            print 'use a clean summary_dir'
            quit()
    makedirs(dir_path)
    '''
    oo=open(os.path.join(dir_path,'config.txt'),'w')
    d={}
    for name in dir(config):
        if '__' in name:continue
        d[name]=getattr(config,name)
    try:
        oo.write(json.dumps(d,ensure_ascii=False))
    except:
        pass
    '''
