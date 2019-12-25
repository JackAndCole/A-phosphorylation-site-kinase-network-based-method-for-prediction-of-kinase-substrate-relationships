# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd


# normalize matrix.
class normalize:
    def __init__(self):
        pass

    def fit(self, M):
        self.max = np.amax(M)
        self.min = np.amin(M)

    def predict(self, M):
        return (M - self.min) * 1. / (self.max - self.min)


# read BLOSUM62 matrix to memory.
def __blosum62__():
    df = pd.read_table('BLOSUM62.txt', index_col=0, sep='\s+', skiprows=6)
    return df


mpb = __blosum62__()
