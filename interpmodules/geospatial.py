'''
spatial interpolation methods
that apply a spatially-informed rule
based on the full mesh
'''


import copy
from gstools.covmodel import JBessel, Gaussian, Exponential
from gstools.krige import Ordinary, Universal
from . import helpers
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
import numpy as np
from pykrige.ok3d import OrdinaryKriging3D
from pykrige.rk import RegressionKriging
from pykrige.uk3d import UniversalKriging3D
import skgstat as skg
import time




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
        bandwidth = selector.search(criterion='AICc') # minimize estimation error using spatial information from sampled locations
        # as opposed to CV minimizing prediction error in subsets of sampled location - not the most useful

    regressor = GWR(coords, y, X, bw=bandwidth, kernel='bisquare', spherical=spherical, fixed=False) # Wheeler and Paez, 2010
        
    return regressor, selector



def smoothing_over_GWR(X0, Y0, Yk0, bandwidth, spherical=True, timeit=False):

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
        
    regressor, selector = _learn_gw_regressor(X0, Y0, Yk0, bandwidth, spherical=True)
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



def interpolate_GWR(X0, Y0, Yk0, x1, yk1, bandwidth, spherical=True, timeit=False, n_jobs=10):
    '''
    iteratively train a regressor across query points
    '''
    
    if timeit: t1 = time.time()
        
    regressor, selector = _learn_gw_regressor(X0, Y0, Yk0, bandwidth, spherical=True)
    regressor = regressor.fit()
    
    lon1, lat1, rho1 = helpers._cartesian_to_spherical(x1, 'degrees')
    sph1 = np.vstack([lon1, lat1]).T
    
    y1_interp = np.zeros(sph1.shape[0])
    
    y1_interp = Parallel(n_jobs = 10)(delayed(_propagate_and_predict)(regressor, (sph1[i:i+1, :], yk1[:,i:i+1])) for i in range(sph1.shape[0]))
    y1_interp = np.array(y1_interp).flatten()
    
    if timeit: t2 = time.time(); return y1_interp, t2 - t1

    return y1_interp



def _make_a_kriger(X0, Y0, x1, x_mesh, yk=None, Yk=None, model = 'gaussian', method='Ordinary', timeit=True, plotit=False, filename=None):
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
        kriger.fit(p=Yk.T, x=X0, y=Y0)
        
    return kriger
    

def interpolate_Krige(X0, Y0, x1, x_mesh, yk=None, Yk=None, model = 'gaussian', method='Ordinary', timeit=False, plotit=False, filename=None):
    '''
    wrapper for kriging modalities
    '''
    

    if method == 'Simple':
        
        y1, error, run_time = interpolate_Krige_wrapper(X0, Y0, x1, x_mesh, model=model, timeit=True, plotit=plotit,filename=filename)
        if timeit: return y1, error, run_time
        else: return y1, error

        
    t1 = time.time()
    kriger = _make_a_kriger(X0, Y0, x1, x_mesh, yk=yk, Yk=Yk, model = model, method=method, timeit=timeit, plotit=plotit, filename=filename)

    
    if method == 'Regression':

        y1 = kriger.predict(p=yk.T, x=x1)
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
    

    






#-------------------------------OBSOLETE--------------------------------#
def interpolate_Krige_wrapper(X0, Y0, x1, x_mesh, model='gaussian', timeit=True, plotit=False, filename=None):

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



def interpolate_OKrig(X0, Y0, x1, x_mesh, model='best', timeit=False, plotit=False, filename=None):
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

def interpolate_UKrig(X0, Y0, x1, x_mesh, model='best', timeit=True, plotit=False, filename=None):
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

    krig_mesh = kriger.mesh(mesh = x_mesh, points = 'points', name = 'UKrig_field')

    y1, error = kriger(x1)

    if timeit: t2=time.time(); return y1,error, t2-t1

    return y1, error

def interpolate_RKrig(X0, Y0, Yk, x1, yk1, x_mesh, model='best', timeit=True, plotit=False, filename=None):
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
    kriger.fit(p=Yk.T, x=Sph.T, y=Y0)

    lon1, lat1, rho1 = helpers._cartesian_to_spherical(x1, unit='degrees')
    sph1 = np.vstack([lon1, lat1])
    y1 = kriger.predict(p=yk1.T, x=sph1.T)
    error = kriger.krige_residual(sph1.T)

    if timeit: t2=time.time(); return y1,error, t2-t1
        
    return y1, error