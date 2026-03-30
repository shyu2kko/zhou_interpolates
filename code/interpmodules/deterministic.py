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

METRICS_TO_CONSIDER =  ['pearson', 'spearman', 'ssim', \
                           'r2', 'rmse', 'kolmogorov_smirnov', \
                            'jensen_shannon']
LIST_OF_KERNELS = ['cubic', 'quintic', 'thin_plate_spline', \
                   'gaussian', 'linear', \
                   'multiquadric', 'inverse_quadratic', 'inverse_multiquadric']
ONLY_ABOVE_10 = ['quintic', 'cubic' ,'thin_plate_spline']


def interpolate_random(X0, Y0, x1, timeit=True):
    '''
    Simply fill the map's missing point by random values within the
    min-max range of known values

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
    y1[ np.where(np.isnan(y1))[0] ] = 0
    
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



def how_long(fn, expected, args):
    t1 = time.time()

    result = fn(*args)

    t2 = time.time()

    if np.array_equal(result, expected):
        return t2 - t1
    print('wrong function or input')


def _find_param_best_dist_chunk(method, center, known_coords, full_map, n_dist_folds=5):
    
    p_params = []
    k_params = []
    kern_params = []

    # this data frame is as long as the number of distance folds
    mdf_concat = pd.DataFrame()
    
    center = center.reshape(1,-1)
    d_j = helpers._get_distance_between(known_coords, center)
    chunk_coords, chunk_verts, chunk_bounds = helpers._chunking_express(d=d_j, coords=known_coords, n_chunks=n_dist_chunks)

    for test_chunk_ind in range(n_dist_chunks):
        
        test = chunk_coords[test_chunk_ind]
        train = [chunk for i,chunk in enumerate(chunk_coords) if i!=test_chunk_ind]
        train = np.concatenate( chunk_coords[:test_chunk_ind] + chunk_coords[test_chunk_ind+1:] )
        
        test_verts = chunk_verts[test_chunk_ind]
        train_verts = [chunk for i,chunk in enumerate(chunk_verts) if i!=test_chunk_ind]
        train_verts = np.concatenate( chunk_verts[:test_chunk_ind] + chunk_verts[test_chunk_ind+1:] )
    
        _, param_best, mdf, _ = _interpolate_and_optimize(method = method, X0=train, Y0=full_map[ train_verts ], \
                                                 x1=test, ground_truth=full_map[ test_verts ], bounds = (2,20))
        print(param_best)
        if method == 'idw': 
            p_params.append(param_best)
        if method == 'knn': 
            k_params.append(param_best)
        if method == 'rbf': 
            k_params.append(param_best[0])
            kern_params.append(param_best[1])
            
        mdf_concat = pd.concat([mdf_concat, mdf], axis = 1)

    return p_params, k_params, kern_params, mdf_concat



def _find_param_best_cv(method, pvdata, known_coords, full_map, distdep, center=None, nfolds=5, seed=321):
    
    p_params = []
    k_params = []
    kern_params = []

    mdf_concat = pd.DataFrame()


    if distdep: # this is for distance weighting with a gaussian kernel rather than just chunking the data
        cv_train_coords, cv_train_verts, cv_test_coords, cv_test_verts = helpers.cv_distdep_noreplace(pvdata=pvdata, center=center, nfolds=nfolds, rseed=seed)

    if not distdep:
        cv_train_coords, cv_train_verts, cv_test_coords, cv_test_verts = helpers.cv_random_noreplace(pvdata=pvdata, nfolds=nfolds, rseed=seed)
    

    interps = []

    for test_fold_ind in range(nfolds):
    
        train = cv_train_coords[test_fold_ind]
        train_verts = cv_train_verts[test_fold_ind]
        test = cv_test_coords[test_fold_ind]
        test_verts = cv_test_verts[test_fold_ind]
    
        _, param_best, mdf, _ = _interpolate_and_optimize(method = method, X0=train, Y0=full_map[ train_verts ], \
                                                 x1=test, ground_truth=full_map[ test_verts ], bounds = (2,20))
        print(param_best)
        if method == 'idw': 
            p_params.append(param_best)
        if method == 'knn': 
            k_params.append(param_best)
        if method == 'rbf': 
            k_params.append(param_best[0])
            kern_params.append(param_best[1])
            
        mdf_concat = pd.concat([mdf_concat, mdf], axis = 1)

    return p_params, k_params, kern_params, mdf_concat
    


def cross_validate_det_interp(method, centers, full_map, all_sampling, standard = 'top', timeit=True):

    '''
    distance-dependent cross-validation for deterministic interpolation parameters
    
    '''
    from statistics import mode, mean
    from joblib import Parallel, delayed

    p_params = [] # power
    k_params = [] # n neighbours
    kern_params = [] # kernel function
    mdf_concat = pd.DataFrame()
    
    known_coords, known_verts, unknown_coords, _ = all_sampling # unpack

    if distdep:
        results = Parallel(n_jobs=-1)(
        delayed(_find_param_best_dist_folds)(method=method, center=centers[j], known_coords=known_coords, full_map = full_map, n_dist_folds=5) 
        for j in range(len(centers))
        )
    
        for j in range(len(centers)):
            
            if method == 'idw':
                p_params += results[j][0]
                    
            elif method == 'knn':
                k_params += results[j][1]
                    
            elif method == 'rbf':
                k_params += results[j][1]
                kern_params += results[j][2]
            mdf_concat = pd.concat([mdf_concat, results[j][-1]], axis = 1)

    else:
        results = _find_param_best_random_folds(method, known_coords, full_map, n_folds=5)
        mdf_concat = results[j][-1]
    
    chosen_params = {}

    if standard == "consensus":
        if method == 'rbf':
            df = pd.DataFrame( [k_params, kern_params] ).T; df.columns = ['n_neighbours', 'kernel']
            chosen_kernel = mode(kern_params); print(chosen_kernel)
    
            neighbours_for_kernel = df[ df['kernel'] == chosen_kernel ]['n_neighbours']
            chosen_nneighbours = int(neighbours_for_kernel.mean()); print(chosen_nneighbours)
            chosen_params['n_neighbours'] = chosen_nneighbours
            chosen_params['kernel'] = chosen_kernel
        elif method == 'knn':
            chosen_nneighbours = int(mean(k_params)); print(chosen_nneighbours)
            chosen_params['n_neighbours'] = chosen_nneighbours
        elif method == 'idw':
            chosen_power = int(mean(p_params)); print(chosen_power)
            chosen_params['power'] = chosen_power

    elif standard == "top":
        nruns = len(mdf_concat.columns) # total how many times we're cross-validating
        mdf_concat.columns = np.arange(nruns) # relabel the columns by iteration
        ranks_cv = metrics.accuracy_rank(mdf_concat) # rank all the metrics
        
        top10pct_end = int(len(mdf_concat.columns) * .1) # 
        mdf_top10pct = ranks_cv[:top10pct_end]

        top10pct = mdf_top10pct.index

        print(mdf_top10pct)
        
        if method == 'idw':
            p_params_tmp = np.array(p_params)
            optimized_param = p_params_tmp[top10pct].mean()

            chosen_params['power'] = optimized_param
            
        if method == 'knn':
            k_params_tmp = np.array(k_params)
            optimized_param = int(k_params_tmp[top10pct].mean())

            chosen_params['n_neighbours'] = optimized_param

        elif method == 'rbf':
            df = pd.DataFrame( [k_params, kern_params] ).T; df.columns = ['n_neighbours', 'kernel']
            chosen_kernel = mode(kern_params)
            chosen_params['kernel'] = chosen_kernel

            k_params_tmp = np.array(k_params)
            optimized_param = int(k_params_tmp[top10pct].mean())
            chosen_params['n_neighbours'] = optimized_param

        print(chosen_params)
    res, param_best, rt = _interpolate_best(method, X0 = known_coords, Y0 = full_map[ known_verts.astype(int) ], x1 = unknown_coords,
                                   chosen_params=chosen_params, timeit=True)

    if timeit: return res, param_best, rt
    
    return res, param_best


def _interpolate_best(method, X0, Y0, x1, chosen_params, timeit=True):

    interpfuncs = {'idw': interpolate_IDW, \
                   'knn': interpolate_KNN, \
                   'rbf': interpolate_RBF}
    res, rt = interpfuncs[method] (X0=X0, Y0=Y0, x1=x1, timeit=timeit, **chosen_params)
    
    if method == 'idw': param_best = chosen_params['power']
    elif method == 'knn': param_best = chosen_params['n_neighbours']
    elif method == 'rbf': param_best = (chosen_params['n_neighbours'], chosen_params['kernel'])

    return res, param_best, rt



def _interpolate_and_optimize(method, X0, Y0, x1, ground_truth, bounds, timeit=True):
    
    interpfuncs = {'idw': interpolate_IDW, \
                   'knn': interpolate_KNN, \
                   'rbf': interpolate_RBF}
    
    metrics_to_consider = METRICS_TO_CONSIDER
    
    if x1.shape[0] < 2: metrics_to_consider = ['residual']
    
    interpfunc = interpfuncs[method]
    
    if method == 'rbf' and x1.shape[0] >= 2:
        
        
        list_of_kernels = ['gaussian']
        #list_of_kernels = LIST_OF_KERNELS
        param_vals_tags = [ str(n_nbs) + ' ' + ki for n_nbs in range(*bounds) for ki in list_of_kernels ]
        
        # instantiate dictionaries to compile results
        # with arrays of zeros
        marrays, res_optimize = _gen_param_dict(method, param_vals_tags, metrics_to_consider, x1.shape[0])
        
        for n_nbs in range(*bounds):
            for ki in list_of_kernels:
                if n_nbs < 10 and ki in ONLY_ABOVE_10: 
                    marrays[str(n_nbs) + ' ' + ki] = np.nan
                    res_optimize[str(n_nbs) + ' ' + ki] = np.nan
                    continue

                res = interpfunc(X0, Y0, x1, n_nbs, ki)
                res_optimize[str(n_nbs) + ' ' + ki] = res

                marray = np.array([ metrics.metric(res, ground_truth, mi) for mi in metrics_to_consider ], dtype = 'float32')
                marrays[str(n_nbs) + ' ' + ki] = marray

        mdf = pd.DataFrame(marrays, index=metrics_to_consider)
        param_val_optim = metrics.accuracy_rank(mdf).index[0]; print(param_val_optim)
        res_optimized = res_optimize[param_val_optim]

        param_val_optim = int(param_val_optim.split(' ')[0]), param_val_optim.split(' ')[1]
        rt = how_long(interpfunc, res_optimized, [X0, Y0, x1, param_val_optim[0], param_val_optim[1]])
        print(mdf)
        return res_optimized, param_val_optim, mdf[f"{param_val_optim[0]} {param_val_optim[1]}"], rt

    param_vals_tags=list(range(*bounds))
    
    marrays, res_optimize = _gen_param_dict(method, param_vals_tags, metrics_to_consider, x1.shape[0])

    for param_vals_tag in param_vals_tags:

        res = interpfunc(X0, Y0, x1, param_vals_tag)
        res_optimize[param_vals_tag] = res

        marray = np.array([ metrics.metric(res, ground_truth, mi) for mi in metrics_to_consider ], dtype = 'float32')
        marrays[param_vals_tag] = marray.flatten()

    mdf = pd.DataFrame(marrays, index=metrics_to_consider)
    ranked = None
    if len(metrics_to_consider) == 1: 
        ranked = mdf.T.sort_values(by = metrics_to_consider[0], ascending=True)
    else:
        ranked = metrics.accuracy_rank(mdf)
    
    param_val_optim = ranked.index[0]
    res_optimized = res_optimize[param_val_optim]

    rt = how_long(interpfunc, res_optimized, [X0, Y0, x1, param_val_optim])
        
    return res_optimized, param_val_optim, mdf[param_val_optim], rt



def _gen_param_dict(method, param_vals_tags, metrics_to_consider, unknown_size):
    
    marrays = dict.fromkeys(param_vals_tags)
    res_optimize = dict.fromkeys(param_vals_tags)

    for param_vals_tag in param_vals_tags:
        marrays[param_vals_tag] = np.zeros(len(metrics_to_consider))
        res_optimize[param_vals_tag] = np.zeros(unknown_size)

    return marrays, res_optimize