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


def byteify(inp):
    if isinstance(inp, dict):
        return {byteify(key): byteify(value)
                for key, value in inp.iteritems()}
    elif isinstance(inp, (list, tuple)):
        return [byteify(element) for element in inp]
    elif isinstance(inp, unicode):
        return inp.encode('utf-8')
    else:
        return inp
