# -*- encoding: utf-8 -*-

import numpy as np


def make_amount(proba_vector_r, thresholds, segment_bins):
    '''predictive amount for proba_arr samples'''
    proba_vector_r = np.array(proba_vector_r)
    thresholds = np.array(thresholds)
    segment_bins = np.array(segment_bins)

    # amount above threshold
    above_th_amount = segment_bins[proba_vector_r >= thresholds]
    if above_th_amount.shape[0] < 1:  # there is no amount above threshold,return default amount: 0
        amount = 0
    else:
        amount = above_th_amount.max()
    return int(amount)
