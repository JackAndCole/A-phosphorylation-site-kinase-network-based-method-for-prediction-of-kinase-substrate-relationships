# /usr/bin/env python
# -*-coding:utf-8-*-
import cPickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import pairwise_kernels
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_validation import StratifiedKFold


def readpkl(fname):
    '''Read cPickle format data.'''
    with open(fname, 'rb') as fp:
        pkl = cPickle.load(fp)
    return pkl


def savepkl(fname, vars):
    '''Save data as cPickle format.'''
    with open(fname, 'wb') as fp:
        cPickle.dump(vars, fp, 1)


def get_net_similarity(X, Y=None, sigma=None):
    '''Computing network profile kernel similarity.'''
    if Y == None: Y = X
    gamma = len(Y) * sigma / np.sum(Y * Y)  # calculate gamma for gaussion function
    similarity = pairwise_kernels(X, Y, metric='rbf', gamma=gamma)
    return similarity


def k_fold(clf, idx, network, sim_of_seq, sim_of_ppi, weights, sigma, n_folds=10):
    '''K-folds cross validation.'''
    label = network[:, idx - 1]
    score = np.zeros(len(label), dtype=np.float)
    # k folds cross validation
    skf = StratifiedKFold(label, n_folds=n_folds)
    for tr, ts in skf:
        network_ = np.copy(network)
        network_[ts, idx - 1] = 0  # avoid introducing prior knowledge
        sim_of_net = get_net_similarity(network_, sigma=sigma)
        sim_of_total = np.average([sim_of_net, sim_of_seq, sim_of_ppi], weights=weights, axis=0)
        clf.fit(sim_of_total[tr, :][:, tr], label[tr])
        score[ts] = clf.predict(sim_of_total[ts, :][:, tr])
    # return forecast result
    return score


def cross_validation(network, sim_of_seq, sim_of_ppi, weights, index, alpha=5, sigma=0.25):
    '''Make prediction for negative data.'''
    scores = np.zeros((len(network), len(index)), dtype=np.float)
    for k, idx in enumerate(index):
        print 'Test Kinase %d (%d/%d)' % (idx, k + 1, len(index))
        clf = KernelRidge(kernel='precomputed', alpha=alpha)
        scores[:, k] = k_fold(clf, idx, network, sim_of_seq, sim_of_ppi, weights, sigma)
    labels = network[:, index - 1]
    return labels, scores


if __name__ == '__main__':
    # Load data
    data = readpkl('../data/kinase.pkl')
    index_of_kinase, name_of_kinase = data['index_of_kinase'], data['name_of_kinase']
    data = readpkl('../data/network.pkl')
    network = data['network']
    data = readpkl('../data/similarity.pkl')
    sim_of_seq, sim_of_ppi = data['sequence_similarity'], data['ppi_similarity']

    alpha, sigma = 5, 0.25
    weights = [0.333, 0.333, 0.333]
    labels, scores = cross_validation(network, sim_of_seq, sim_of_ppi, weights, index_of_kinase, alpha, sigma)

    area = list()
    sp, sn = list(), list()
    # Evaluating algorithm performance
    for k, (name, type) in enumerate(name_of_kinase):
        label, score = labels[:, k], scores[:, k]

        fpr, tpr, _ = roc_curve(label, score, pos_label=1)
        area.append(auc(fpr, tpr))
        sp.append(1 - fpr)
        sn.append(tpr)

        f, ax = plt.subplots(1, 1, figsize=(4, 3.5))
        ax.plot(fpr, tpr, label='KsrPred(%.2f%%)' % (area[k] * 100))
        ax.set(xlabel='false positive rate', ylabel='true positive rate')
        ax.set(title='%s' % name)
        ax.legend(loc=4)
        f.tight_layout()
        f.savefig('../images/%s.png' % name, dpi=600)
    # Save forecast result
    savepkl('KsrPred_Result.pkl', {'labels': labels, 'scores': scores, 'auc': area, 'sp': sp, 'sn': sn})
    raw_input('Please press the Enter key to exit!')
