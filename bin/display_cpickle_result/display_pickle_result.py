# /usr/bin/env python
# -*- coding:utf-8-*-
import cPickle
import numpy as np

# View cPickle data.
if __name__ == '__main__':
    with open('../KsrPred_Result.pkl', 'rb') as fp:
        data = cPickle.load(fp)
        print data['auc']