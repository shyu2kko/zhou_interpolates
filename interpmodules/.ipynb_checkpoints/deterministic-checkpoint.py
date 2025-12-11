'''
spatial interpolation methods
that apply a deterministic rule
on points and neighbours
'''


from . import helpers, metrics

import numpy as np
import pandas as pd
import random
from scipy.interpolate import RBFInterpolator
from scipy.spatial import KDTree
import time






def interpolate_random(X0, Y0, x1, timeit=True):
    '''
    Simply fill the map's missing point with points within
    min-max range of the known values

    x1: (n, 3) points to interpolate, with x-y-z coordinates
    '''
    if timeit: t1=time.time()
        
    vmin = np.nanmin(Y0)
    vmax = np.nanmax(Y0)

    y1 = [ random.uniform(vmin, vmax) for i in range(x1.shape[0]) ]

    if timeit: t2 = time.time(); return np.array(y1), t2 - t1
        
    return np.array(y1)




def interpolate_IDW(X0, Y0, x1, power=2, timeit=False):
    '''
    X0: (N, 3) N-many sampled points, with x-y-z coordinates
    Y0: (N,) N-many sampled values at points X0
    
    x1: (n, 3) point to interpolate, with x-y-z coordinates
    '''

    if timeit: t1=time.time()

    Dist = helpers._get_distance_between(X0, x1)

    weights = 1 / (Dist**power)
    weights /= weights.sum(axis=0)
    y1 = np.dot(weights.T, Y0)

    if timeit: t2=time.time(); return y1,t2-t1
        
    return y1



def interpolate_KNN(X0, Y0, x1, n_neighbours=5, timeit=False):
    '''
    Find nearest n-many neighbours, average their values
    and weigh inversely by distance abagen-style
    this is euclidean distance if i'm using vertex coordinates
    '''
    
    if timeit:
        t1=time.time()
        
    tree= KDTree(np.c_[X0[:,0], X0[:,1], X0[:,2]])

        
    dist, idx = tree.query(x1, k=n_neighbours)

    w = helpers._get_inverse_distance_weights(dist)

    # get average of nearest neighbors
    y1 = np.sum(Y0[idx]* w, axis=1) / np.sum(w, axis = 1)

    if timeit: t2=time.time(); return y1,t2-t1
        
    return y1



def interpolate_RBF(X0, Y0, x1, n_neighbours=10, kernel='cubic', timeit=False):
    '''
    wrapper for scipy interpolator

    kernels to focus on: `linear`, `cubic`, `gaussian`
    others are available through scipy
    '''
    
    if timeit: t1=time.time()
    
    y1 = np.full(x1.shape[0], np.nan)

    for eps in range(1,6):

        try:
            interpolator = RBFInterpolator(y=X0, d=Y0, neighbors=n_neighbours, \
                            kernel=kernel, epsilon=eps, smoothing=1e-4) # small scaling of smoothing parameter for very sparse samples
            y1 = interpolator(x1);

        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err): continue

    if timeit: t2=time.time(); return y1,t2-t1
    
    return y1





#----OPTIMIZATION (aka overfitting)----#

def how_long(fn, expected, args):
    t1 = time.time()

    result = fn(*args)

    t2 = time.time()

    if np.array_equal(result, expected):
        return t2 - t1
    print('wrong function or input')




def interpolate_and_optimize(method, X0, Y0, x1, ground_truth, bounds, timeit=True):


    metrics_to_consider =  ['pearson', 'spearman', 'ssim', \
                           'r2', 'rmse', 'kolmogorov_smirnov', \
                            'jensen_shannon']
    
    list_of_kernels = None
    not_above_k10 = None

    if method == 'idw': interpfunc = interpolate_IDW
    if method == 'knn': interpfunc = interpolate_KNN
    if method == 'rbf': 
        interpfunc = interpolate_RBF
        list_of_kernels = ['cubic', 'quintic', 'thin_plate_spline', 'gaussian', 'linear', 'multiquadric', 'inverse_quadratic', 'inverse_multiquadric']
        not_above_k10 = ['quintic', 'cubic' ,'thin_plate_spline']

        param_vals_tags = [ str(n_nbs) + ' ' + ki for n_nbs in range(*bounds) for ki in list_of_kernels ]

        marrays = dict.fromkeys(param_vals_tags )
        res_optimize = dict.fromkeys(param_vals_tags)

        for n_nbs in range(*bounds): 
            for ki in list_of_kernels:

                marrays[str(n_nbs) + ' ' + ki] = np.zeros(len(metrics_to_consider))
                res_optimize[str(n_nbs) + ' ' + ki] = np.zeros(x1.shape[0])

        for n_nbs in range(*bounds):

            for ki in list_of_kernels:

                if n_nbs < 10 and ki in not_above_k10: 
                    marrays[str(n_nbs) + ' ' + ki] = np.nan
                    res_optimize[str(n_nbs) + ' ' + ki] = np.nan
                    continue

                res = interpfunc(X0, Y0, x1, n_nbs, ki); res_optimize[str(n_nbs) + ' ' + ki] = res


                marray = np.array([ metrics.metric(res, ground_truth, mi) for mi in metrics_to_consider ], dtype = 'float32')
                
                marrays[str(n_nbs) + ' ' + ki] = marray

        mdf = pd.DataFrame(marrays, index=metrics_to_consider)
        param_val_optim = metrics.accuracy_rank(mdf).index[0]; print(param_val_optim)
        res_optimized = res_optimize[param_val_optim]

        param_val_optim = int(param_val_optim.split(' ')[0]), param_val_optim.split(' ')[1]
        rt = how_long(interpfunc, res_optimized, [X0, Y0, x1, param_val_optim[0], param_val_optim[1]])

        return res_optimized, param_val_optim, rt


    marrays = dict.fromkeys(range(*bounds))
    res_optimize = dict.fromkeys(range(*bounds))

    for param_val in range(*bounds): 
        marrays[param_val] = np.zeros(len(metrics_to_consider))
        res_optimize[param_val] = np.zeros(x1.shape[0])


    # inverse distance weighting
    for param_val in range(*bounds):

        res = interpfunc(X0, Y0, x1, param_val); res_optimize[param_val] = res


        marray = np.array([ metrics.metric(res, ground_truth, mi) for mi in metrics_to_consider ], dtype = 'float32')
        
        marrays[param_val] = marray

    mdf = pd.DataFrame(marrays, index=metrics_to_consider)
    param_val_optim = metrics.accuracy_rank(mdf).index[0]
    res_optimized = res_optimize[param_val_optim]

    rt = how_long(interpfunc, res_optimized, [X0, Y0, x1, param_val_optim])

    return res_optimized, param_val_optim, rt
