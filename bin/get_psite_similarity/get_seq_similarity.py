# /usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
from utils import normalize, mpb


def similarity(S1, S2=None):
    '''calculate sequence similarity.'''
    if S2 == None: S2 = np.copy(S1)

    rst = np.zeros((len(S1), len(S2)), dtype=np.int)
    for i, s1 in enumerate(S1):
        for j, s2 in enumerate(S2):
            # t = 0
            # for s in range(len(s1)):
            #     t += mpb[s1[s]][s2[s]]
            # rst[i, j] = t
            # pick up speed, equal front code.
            rst[i, j] = np.sum(map(lambda x: mpb.get_value(*x), zip(s2, s1)))
        if i % 100 == 0: print i
    return rst


if __name__ == '__main__':
    lsq = np.loadtxt('sequence.txt', delimiter='\t', dtype=np.str)

    sim_lsq = similarity(lsq)

    clf = normalize()
    clf.fit(sim_lsq)

    np.savetxt('sim_lsq.txt', clf.predict(sim_lsq), delimiter='\t', fmt='%.3f')

    print 'Success!'
