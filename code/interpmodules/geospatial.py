'''
spatial interpolation methods
that apply a spatially-informed rule
based on the full mesh
'''


import copy
from gstools.covmodel import JBessel, Gaussian, Exponential
from gstools.krige import Ordinary, Universal
from . import helpers, metrics
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
import numpy as np
import pandas as pd
from pykrige.ok3d import OrdinaryKriging3D
from pykrige.rk import RegressionKriging
from pykrige.uk3d import UniversalKriging3D
import skgstat as skg
import time


METRICS_TO_CONSIDER =  ['pearson', 'spearman', 'ssim', \
                           'r2', 'rmse', 'kolmogorov_smirnov', \
                            'jensen_shannon']



# Geographically-weighted regression
# takes covariates
def _learn_gw_regressor(X0, Y0, Yk0, bandwidth, spherical=True):
    '''
    Yk0: covariates measured at sampled location
    yk1: covariates measured at target location

    LEARN GEOGRAPHICALLY WEIGHTER REGRESSOR ONE SHOT
    REUSE REGRESSOR FOR INTERPOLATION AT EVERY POINT - MUCH faster

    OLS INITIATED AT EVERY LOCATION i ACROSS n points
    LEARN COEFFICIENTS FOR EVERY COVARIATE k ACROSS p predictors

    KERNEL BEING BISQUARE WITH ADAPTIVE NEIGHBOURHOOD FOR MAKING WEIGHTS (binary)
    
    '''

    Lon, Lat, Rho = helpers._cartesian_to_spherical(X0, unit='degrees')
    Sph = np.vstack([Lon, Lat])

    # Reformat coordinates array to make the libpysal function happy
    coords = list(zip(Sph[0,:], Sph[1,:]))
    y = Y0.reshape(-1,1)
    X = Yk0.T

    # get bandwith (aka adaptive neighbourhood size) for regressions
    if bandwidth == None:
        selector = Sel_BW(coords, y, X_loc=X, fixed=False, spherical=spherical)
        
        try:
            bandwidth = selector.search(criterion='AICc', search_method = 'golden_section') # minimize estimation error using spatial information from sampled locations
            # as opposed to CV minimizing prediction error in subsets of sampled location - not the most useful

        except np.linalg.LinAlgError:
            bandwidth = 100.0
    print(bandwidth)
    regressor = GWR(coords, y, X, bw=bandwidth, kernel='bisquare', spherical=spherical, fixed=False) # Wheeler and Paez, 2010
    
    return regressor, selector



def interpolate_SWR(X0, Y0, Yk0, x1, yk1, bandwidth=None, spherical=True, timeit=False):
    '''
    iteratively train a regressor across query points
    '''
    
    if timeit: t1 = time.time()
        
    reg, sel = _learn_gw_regressor(X0, Y0, Yk0, bandwidth, spherical=True)

    mdl = reg.fit()

    scale = mdl.scale
    residuals = mdl.resid_response

    pred_coords = x1
    pred_X = yk1
    pred_results = reg.predict(pred_coords, pred_X.T, scale, residuals)

    y1_interp = pred_results.predictions.flatten()
    
    if timeit: t2 = time.time(); return y1_interp, t2 - t1

    return y1_interp



def _make_a_kriger(X0, Y0, x1, yk1=None, Yk0=None, model = 'exponential', method='Ordinary', timeit=True, plotit=False, filename=None):
    '''
    exporting variogram fitted with gkstat to gstools

    for Ordinary and Regression kriging
    '''

    #if timeit: t1=time.time()

    binned_vario = skg.Variogram(coordinates = X0, \
             values = Y0, \
             model = model, \
             n_lags = 25, bin_func = 'even', use_nugget = True)
    
    x, y = binned_vario.get_empirical(bin_center=True)

    bounds = (0.00001, [np.nanmax(x), np.nanmax(y), np.nanmax(y)/10])
    binned_vario.fit(force=True, method = 'trf', p0=bounds[1], maxfev=1000)

    if plotit:
        import matplotlib.pyplot as plt
    
        plt.scatter(x, y, color = 'k')
        
        plt.plot(binned_vario.data()[0], binned_vario.data()[1], color = 'r')
        plt.axhline(binned_vario.cof[1], color = 'r', linestyle = 'dotted')
        plt.axvline(binned_vario.cof[0], color = 'r', linestyle = 'dotted')
        plt.ylabel('semivariance')
        plt.xlabel('distance')
        plt.savefig(filename)
        
    fitted_vario = binned_vario.to_gstools()
    
    if method == 'Ordinary':
        kriger = OrdinaryKriging3D(x=X0[:,0], y=X0[:,1], z=X0[:,2], val = Y0,\
                                   variogram_model = fitted_vario, \
                                exact_values = True)

    elif method == 'Universal':
        kriger = UniversalKriging3D(x=X0[:,0], y=X0[:,1], z=X0[:,2], val = Y0,\
                                   variogram_model = fitted_vario, \
                                exact_values = True)

    if method == 'Regression':
        
        kriger = RegressionKriging(variogram_model = fitted_vario, \
                                   coordinates_type = 'euclidean', \
                                   method = 'ordinary3d', exact_values=True)
        kriger.fit(p=Yk0.T, x=X0, y=Y0)
        
    return kriger
    

def interpolate_Krige(X0, Y0, x1, yk1=None, Yk0=None, model = 'exponential', method='Ordinary', timeit=False, plotit=False, filename=None):
    '''
    wrapper for kriging modalities
    '''
    

    if method == 'Simple':
        
        try:
            y1, error, run_time = interpolate_Krige_wrapper(X0, Y0, x1, model=model, timeit=True, plotit=plotit,filename=filename)
        except (ValueError, RuntimeError) as e:
            y1, error = np.nan, np.nan
            print(f"       >:(  00PS: could not fit a positive-variance {model} model !!       ")
            return y1, error
        if timeit: return y1, error, run_time
        else: return y1, error

        
    t1 = time.time()
    try:
        kriger = _make_a_kriger(X0, Y0, x1, yk1=yk1, Yk0=Yk0, model = model, method=method, timeit=timeit, plotit=plotit, filename=filename)
    except (ValueError, RuntimeError) as e:
        y1, error = np.nan, np.nan
        print(f"       >:(  00PS: could not fit a positive-variance {model} model !!       ")
        return y1, error
        
    if method == 'Regression':

        y1 = kriger.predict(p=yk1.T, x=x1)
        resid = kriger.krige_residual(X0)

        if timeit: t2 = time.time(); return y1, resid, t2 - t1
        return y1, resid

        
    elif method in ['Ordinary', 'Universal']:

        y1, error = kriger.execute(style = 'points', xpoints = x1[:,0], ypoints = x1[:,1], zpoints = x1[:, 2])

        if timeit: 
            t2 = time.time()
            return y1, error, t2-t1
        return y1, error
    
    t2 = time.time()
    print('invalid method, time eloped =' + str(t2 - t1))

    
    

    
def _iterate_chunks(method, pvdata, covariates, full_map, center = None, distdep = True, distchunks=False, nfolds=5, seed=321):

    known_coords = pvdata.points
    
    # this data frame is as long as the number of distance chunks
    mdf_concat = pd.DataFrame()

    if distchunks:
        center = center.reshape(1,-1)
        d_j = helpers._get_distance_between(known_coords, center)
        chunk_coords, chunk_verts, chunk_bounds = helpers._chunking_express(d=d_j, coords=known_coords, n_chunks=nfolds)

        for test_chunk_ind in range(n_dist_chunks):
            
            test = chunk_coords[test_chunk_ind]
            train = [chunk for i,chunk in enumerate(chunk_coords) if i!=test_chunk_ind]
            train = np.concatenate( chunk_coords[:test_chunk_ind] + chunk_coords[test_chunk_ind+1:] )
            
            test_verts = chunk_verts[test_chunk_ind]
            train_verts = [chunk for i,chunk in enumerate(chunk_verts) if i!=test_chunk_ind]
            train_verts = np.concatenate( chunk_verts[:test_chunk_ind] + chunk_verts[test_chunk_ind+1:] )

            if method == 'krige':
                res, mdf = _interpolate_and_optimize(method = method, X0=train, Y0=full_map[ train_verts ], \
                x1=test, ground_truth=full_map[ test_verts ])
            elif method in ['swr', 'regkrige']:
                res, mdf = _interpolate_and_optimize(method = method, X0=train, Y0=full_map[ train_verts ], \
                x1=test, Yk0=covariates[:,train_verts], yk1=covariates[:,test_verts],ground_truth=full_map[ test_verts ])
                
            mdf_concat = pd.concat([mdf_concat, mdf], axis = 1)

    if distdep:
        cv_train_coords, cv_train_verts, cv_test_coords, cv_test_verts = helpers.cv_distdep_noreplace(pvdata=pvdata, nfolds=nfolds, center=center, rseed=seed)
    else:
        # do random samples
        cv_train_coords, cv_train_verts, cv_test_coords, cv_test_verts = helpers.cv_random_noreplace(pvdata=pvdata, nfolds=nfolds, rseed=seed)
        
    interps = []

    for test_fold_ind in range(nfolds):
    
        train = cv_train_coords[test_fold_ind]
        train_verts = cv_train_verts[test_fold_ind]
        test = cv_test_coords[test_fold_ind]
        test_verts = cv_test_verts[test_fold_ind]
        
        if method == 'krige':
            res, mdf = _interpolate_and_optimize(method = method, X0=train, Y0=full_map[ train_verts ], \
            x1=test, ground_truth=full_map[ test_verts ])
        elif method in ['swr', 'regkrige']:
            res, mdf = _interpolate_and_optimize(method = method, X0=train, Y0=full_map[ train_verts ], \
            x1=test, Yk0=covariates[:,train_verts], yk1=covariates[:,test_verts],ground_truth=full_map[ test_verts ])
        interps.append(res)
        mdf_concat = pd.concat([mdf_concat, mdf], axis = 1)

    return mdf_concat


def _interpolate_and_optimize(method, X0, Y0, x1, ground_truth, Yk0=None, yk1=None):
    
    if method in ['swr', 'regkrige'] and not isinstance(Yk0, np.ndarray): 
        raise ValueError("require covariates")
    marrays = {}
    interpfuncs = {"swr": interpolate_SWR, \
                   "regkrige": interpolate_Krige, \
                   "krige": interpolate_Krige}

    if method == 'swr':
        res = interpfuncs[method](X0=X0, Y0=Y0, Yk0=Yk0, x1=x1, yk1=yk1, timeit=False)
    elif method == 'krige':
        res,_ = interpfuncs[method](X0=X0, Y0=Y0, Yk0=Yk0, x1=x1, yk1=yk1, method = "Ordinary", timeit=False)
    elif method == 'regkrige':
        res,_ = interpfuncs[method](X0=X0, Y0=Y0, Yk0=Yk0, x1=x1, yk1=yk1, method = "Regression", timeit=False)
    
    metrics_to_consider = METRICS_TO_CONSIDER
    if not isinstance(res, np.ndarray) and np.isnan(res):
        mdf = pd.DataFrame(np.array([np.nan] * len(metrics_to_consider)), index=metrics_to_consider)
        return res, mdf

    marray = np.array([ metrics.metric(res.flatten(), ground_truth.flatten(), mi) for mi in metrics_to_consider ], dtype = 'float32')
    marrays[method] = marray
    mdf = pd.DataFrame(marrays, index=metrics_to_consider)

    print(mdf)

    return res, mdf
    


def cross_validate_gsp_interp(method, pvdata, full_map, all_sampling, covariates, distdep=False, centers = None, timeit=True):

    from statistics import mode, mean
    from joblib import Parallel, delayed

    mdf_concat = pd.DataFrame()
    
    known_coords, known_verts, unknown_coords, _ = all_sampling # unpack

    if distdep:
        results = Parallel(n_jobs=-1)(
        delayed(_iterate_chunks)(method, pvdata, covariates, full_map, \
                                 center = centers[j], distdep = distdep, n_chunks=5) 
        for j in range(len(centers))
        )
    else:
        results =_iterate_chunks(method, pvdata, covariates, full_map, \
                                 center = None, distdep = distdep, n_chunks=5, seed=321)




#-------------------------------OBSOLETE--------------------------------#
def interpolate_Krige_wrapper(X0, Y0, x1, model='gaussian', timeit=True, plotit=False, filename=None):

    '''
    wrapper for both gkstat and gstolls
    '''

    if timeit: t1=time.time()

    binned_vario = skg.Variogram(coordinates = X0, \
             values = Y0, \
             model = model, \
             n_lags = 25, bin_func = 'even', use_nugget = True)
    
    x, y = binned_vario.get_empirical(bin_center=True)

    bounds = (0.00001, [np.nanmax(x), np.nanmax(y), np.nanmax(y)/10])
    binned_vario.fit(force=True, method = 'trf', p0=bounds[1], maxfev=1000)

    if plotit:
        import matplotlib.pyplot as plt
        plt.figure(figsize = (4,2))
        plt.scatter(x, y, color = 'k')
        
        plt.plot(binned_vario.data()[0], binned_vario.data()[1], color = 'green')
        plt.axhline(binned_vario.cof[1], color = 'green', linestyle = 'dotted')
        plt.axvline(binned_vario.cof[0], color = 'green', linestyle = 'dotted')
        plt.ylabel('semivariance')
        plt.xlabel('distance')
        plt.savefig(filename)
        
    kriger = binned_vario.to_gs_krige()
    y1 = np.zeros(x1.shape[0])
    error = np.zeros(x1.shape[0])
    
    for pi in range(x1.shape[0]):
        
        y1[pi], error[pi] = kriger(x1[pi,:], 'structured')

    if timeit: t2=time.time(); return y1, error, t2-t1
    return y1, error



def interpolate_OKrig(X0, Y0, x1, model='best', timeit=False, plotit=False, filename=None):
    '''
    wrapper for gstool krige class `Ordinary`

    model: gaussian, exponential, jbessel hole model, or best - 
    fit all of them and find the one with least sse
    '''

    if timeit: t1=time.time()
        
    Dist = helpers.internal_distance_matrix(X0)
    summary = helpers._fit_empirical_variogram(x=Y0,D=Dist,X=X0, fast = True, plotit=plotit, filename=filename)

    best_idx = summary['models'].index(summary['vario_model'])
    ran = summary['vario_range']; l = helpers._convert_to_length(ran)
    nugget = summary['vario_nugget']

    if model=='best': model = summary['vario_model']
        
    if model=='gaussian': mdl = Gaussian(dim=3, var=1, len_scale=l,nugget=nugget)
    elif model=='exponential': mdl = Exponential(dim=3, var=1, len_scale=l, nugget=nugget)
    elif model=='hole': mdl = JBessel(dim=3, var=1, len_scale=l, nugget=nugget)

    
    kriger = Ordinary(model=mdl, cond_pos=X0, cond_val=Y0, \
                      exact=True, cond_err='nugget')

    y1, error = kriger(x1)

    if timeit: t2=time.time(); return y1,error, t2-t1

    return y1, error

def interpolate_UKrig(X0, Y0, x1, model='best', timeit=True, plotit=False, filename=None):
    '''
    wrapper for gstool krige class `Ordinary`

    model: gaussian, exponential, jbessel hole model, or best - 
    fit all of them and find the one with least sse
    '''

    if timeit: t1=time.time()
        
    Dist = helpers.internal_distance_matrix(X0)
    summary = helpers._fit_empirical_variogram(x=Y0,D=Dist,X=X0,fast=True, plotit=plotit,filename=filename)
    best_idx = summary['models'].index(summary['vario_model'])
    ran = summary['vario_range']; l = helpers._convert_to_length(ran)
    nugget = summary['vario_nugget']

    if model=='best': model = summary['vario_model']
        
    if model=='gaussian': mdl = Gaussian(dim=3, var=1, len_scale=l,nugget=nugget)
    elif model=='exponential': mdl = Exponential(dim=3, var=1, len_scale=l, nugget=nugget)
    elif model=='hole': mdl = JBessel(dim=3, var=1, len_scale=l, nugget=nugget)

    
    kriger = Universal(model=mdl, cond_pos=X0, cond_val=Y0, drift_functions = 1,\
                      exact=True, cond_err='nugget')

    #krig_mesh = kriger.mesh(mesh = x_mesh, points = 'points', name = 'UKrig_field')

    y1, error = kriger(x1)

    if timeit: t2=time.time(); return y1,error, t2-t1

    return y1, error

def interpolate_RKrig(X0, Y0, Yk0, x1, yk1, model='best', timeit=True, plotit=False, filename=None):
    '''
    wrapper for PyKrige krige class `RegressionKriging`

    model: gaussian, exponential, jbessel hole model, or best - 
    fit all of them and find the one with least sse
    '''

    if timeit: t1=time.time()
        
    Dist = helpers._internal_distance_matrix(X0)
    summary = helpers._fit_empirical_variogram(x=Y0,D=Dist,X=X0,fast=True, plotit=plotit, filename=filename)
    best_idx = summary['models'].index(summary['vario_model'])
    ran = summary['vario_range']; l = helpers._convert_to_length(ran)
    nugget = summary['vario_nugget']

    if model=='best': model = summary['vario_model']
        
    if model=='hole': model = 'hole-effect'

    
    Lon, Lat, Rho = helpers._cartesian_to_spherical(X0, unit = 'degrees')
    Sph = np.vstack([Lon, Lat])

    kriger = RegressionKriging(variogram_model = model, coordinates_type = 'geographic', exact_values = True)
    kriger.fit(p=Yk0.T, x=Sph.T, y=Y0)

    lon1, lat1, rho1 = helpers._cartesian_to_spherical(x1, unit='degrees')
    sph1 = np.vstack([lon1, lat1])
    y1 = kriger.predict(p=yk1.T, x=sph1.T)
    error = kriger.krige_residual(sph1.T)

    if timeit: t2=time.time(); return y1,error, t2-t1
        
    return y1, error


    
    
#-------------------obsolete-----------------#
def _smoothing_over_GWR(X0, Y0, Yk0, bandwidth, spherical=True, timeit=False):

    '''
    X0: (N, 3) coordinates at sample location
    Y0: (N,) values
    Yk0: covariates measured at sampled location
    x1: (k, 3) coordinates at unknown location
    yk1: (k,) covariates measured at unknown locations

    bandwidth: to be determined

    spherical: pertaining to coordinates type
    
    '''
    if timeit: t1 = time.time()

        
    regressor, selector = _learn_gw_regressor(X0, Y0, Yk0, bandwidth, spherical=spherical)
    regressor = regressor.fit()

    y1_interp = regressor.predy.flatten()

    print('Note: not learning new coefficient, using coefficients at KNOW LOCATIONS to smooth over ZEROED UNKNOWN VALUES')

    if timeit: t2=time.time(); return y1_interp,t2-t1

    return y1_interp



def _propagate_and_predict(regressor, data):
    '''
    intake data in spherical coordinates at single query point

    data: (,3), (,k) tuple of spherical coordinate and covariate
    '''
    
    sph1, yk1 = data # unpack at data point
    reg = copy.deepcopy(regressor) # propagate the regressor
    y1 = reg.predict(sph1, yk1) # get one prediction
    
    return y1.predy