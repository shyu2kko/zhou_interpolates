'''
metrics
'''

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ks_2samp
from scipy.spatial.distance import jensenshannon
from skimage.metrics import structural_similarity
from sklearn.metrics import root_mean_squared_error, r2_score

def metric(actual, interpolated, method):

    methods_to_consider = ['pearson', 'spearman', 'ssim', \
                           'r2', 'rmse', 'kolmogorov_smirnov', \
                            'jensen_shannon', 'aggregate', 'residual']
    
    if method not in methods_to_consider:
        print('invalid method choice: please try again with one of the following')
        print(methods_to_consider)

    elif method == 'pearson':
        return np.corrcoef(actual, interpolated)[1,0]
    
    elif method == 'spearman':
        return spearmanr(actual, interpolated).correlation
    
    elif method == 'ssim':
        return structural_similarity(actual, interpolated, data_range = 666)
    
    elif method == 'r2':
        return r2_score(actual, interpolated)
    
    elif method == 'rmse':
        return root_mean_squared_error(actual, interpolated)
    
    elif method == 'kolmogorov_smirnov':
        return ks_2samp(actual, interpolated).statistic

    elif method == 'jensen_shannon':
        # convert to probability vector
        prob_actual = abs(actual/np.sum(actual))
        prob_interpolated = abs(interpolated/np.sum(interpolated))

        return jensenshannon(prob_actual, prob_interpolated, base=2)**2
        
    elif method == 'residual':
        return abs(actual - interpolated)


def accuracy_rank(metrics):

    metrics = metrics.dropna(axis=1)

    rank_scores = []

    metric_names = list(metrics.index)
    score_col = []
    list_up = list(range(1, metrics.shape[1]+1))
    list_down = list(range(metrics.shape[1], 0, -1))

    for metric_name in metric_names:
        if metric_name in ['pearson', 'spearman', 'ssim', 'r2']:
            rank_scores.append(pd.Series(list_down, index=metrics.loc[metric_name,:].sort_values(ascending=False).index))
            
        if metric_name in ['jensen_shannon', 'rmse', 'kolmogorov_smirnov', 'run_time']:
            rank_scores.append(pd.Series(list_up, index=metrics.loc[metric_name,:].sort_values(ascending=False).index))
        score_col.append(metric_name)

    score=pd.concat([m for m in rank_scores],axis=1)
    score.columns = score_col
    score = score/metrics.shape[1]
    return score