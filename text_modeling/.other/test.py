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


class ngram_feature(object):

    def __init__(self, f_def):
        '''
        f_def defines ngram feature generate method
        f_def[k]="w[k-1],w[k];ch[k-1][-1],ch[k][0]"
        '''
        self.features = [self.parse(s) for s in f_def.split(';')]

    def parse(self, s):
        s = "lambda w,ch,k: [{}]".format(s)
        return eval(s)

    def __call__(self, w, ch=[]):
        for f in self.features:
            for k in range(len(w)):
                print f(w, ch, k)


nf = ngram_feature("w[k-1],w[k];ch[k-1][-1],ch[k][0]")
nf([u'我', u'爱', u'北京', u'天安门'], [u'我', u'爱', u'北京', u'天安门'])
