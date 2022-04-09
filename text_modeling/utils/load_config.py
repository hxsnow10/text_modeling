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
import importlib
import os
import sys

# TODO: 检测config的版本与框架的版本的兼容性，警告可能存在的接口变化


def load_config(path='.', *args, **kwargs):
    path = os.path.abspath(path)
    if os.path.isdir(path):
        dir_path = path
        config_name = "config"
    else:
        dir_path = os.path.dirname(path)
        config_name = os.path.basename(path).split('.')[0]

    sys.path.insert(0, dir_path)
    mo = importlib.import_module(config_name)
    config = mo.get_config(*args, **kwargs)
    config.path = os.path.join(dir_path, config_name)
    print "load config from {}/{}".format(dir_path, config_name)
    return config
