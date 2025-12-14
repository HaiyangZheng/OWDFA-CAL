#!/usr/bin/env python
# coding: utf-8
import numpy as np
from loguru import logger
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def direct_acc(y_pred, y_true):
    """
    Calculate accuracy
    # Arguments
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        y_true: true labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    num_correct = np.sum(y_pred == y_true)
    res = num_correct / len(y_true)
    return res


def cluster_map(y_pred, y_true):
    """
    Calculate clustering mapping relation. Require scipy installed
    # Arguments
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        y_true: true labels, numpy.array with shape `(n_samples,)`
    # Return
        row_ind, col_ind
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    d = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((d, d), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    _, col_ind = linear_sum_assignment(w.max() - w)
    row_ind = {col_ind[x]: x for x in col_ind}
    return row_ind, col_ind


def cluster_acc(y_pred, y_true):
    """
    Calculate clustering accuracy. Require scipy installed
    # Arguments
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        y_true: true labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    d = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((d, d), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size


def nmi(y_pred, y_true):
    """
    Calculate normalized mutual information. Require scikit-learn installed
    # Arguments
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        y_true: true labels, numpy.array with shape `(n_samples,)`
    # Return
        NMI, in [0,1]
    """
    return normalized_mutual_info_score(y_true, y_pred)


def ari(y_pred, y_true):
    """
    Calculate adjusted Rand index. Require scikit-learn installed
    # Arguments
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        y_true: true labels, numpy.array with shape `(n_samples,)`
    # Return
        ARI, in [-1,1]
    """
    return adjusted_rand_score(y_true, y_pred)

def val_stat(y_pred, y_label, y_conf, num_known):
    logger.info(f'##### Eval Results #####')

    y_pred = np.asarray(y_pred)
    y_label = np.asarray(y_label)
    y_conf = np.asarray(y_conf)

    known_mask = y_label < num_known
    novel_mask = y_label >= num_known

    mean_uncert = 1 - np.mean(y_conf)

    over_acc = cluster_acc(y_pred, y_label)
    over_nmi = nmi(y_pred, y_label)
    over_ari = ari(y_pred, y_label)
    known_acc = direct_acc(y_pred[known_mask], y_label[known_mask])
    known_nmi = nmi(y_pred[known_mask], y_label[known_mask])
    known_ari = ari(y_pred[known_mask], y_label[known_mask])

    if np.any(novel_mask):
        novel_acc = cluster_acc(y_pred[novel_mask], y_label[novel_mask])
        novel_nmi = nmi(y_pred[novel_mask], y_label[novel_mask])
        novel_ari = ari(y_pred[novel_mask], y_label[novel_mask])

    else:
        novel_acc = novel_nmi = novel_ari = 0.0

    logger.info(f'Uncertainty: {mean_uncert:.4f}')
    logger.info(f'[All]   ACC: {over_acc:.4f}, NMI: {over_nmi:.4f}, ARI: {over_ari:.4f}')
    logger.info(f'[Known] ACC: {known_acc:.4f}, NMI: {known_nmi:.4f}, ARI: {known_ari:.4f}')
    logger.info(f'[Novel] ACC: {novel_acc:.4f}, NMI: {novel_nmi:.4f}, ARI: {novel_ari:.4f}')

    results = {
        'all_acc': over_acc, 'all_nmi': over_nmi, 'all_ari': over_ari,
        'known_acc': known_acc, 'novel_acc': novel_acc,
        'known_nmi': known_nmi, 'novel_nmi': novel_nmi,
        'known_ari': known_ari, 'novel_ari': novel_ari,
        'val_uncert': mean_uncert,
    }

    return results
