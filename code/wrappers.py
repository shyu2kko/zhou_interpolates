'''
function definition for a single run of interpolation assessment
'''
import sys
sys.path.append('../')

from config import *

def generate_covariates(curr_map, mesh, nspins=1000, datasource='grf'):

    
    from interpmodules import helpers
    import copy
    from neuromaps import nulls
    import numpy as np
    import random
    from scipy.spatial import KDTree
    from scipy.ndimage import gaussian_filter1d
    

    ground_truth = curr_map
    Yk0 = None

    if datasource == 'grf':
        y = copy.deepcopy(ground_truth)
        x0 = copy.copy(y)

        # strategy=most correlated spun map
        spun = nulls.spins.gen_spinsamples(coords = mesh.points, hemiid = np.zeros(mesh.n_points), n_rotate = nspins, seed = 321)
        correlations = np.zeros(nspins)
        for this_iter in range(spun.shape[1]):
            correlations[this_iter] = np.corrcoef(x0[spun[:,this_iter]], y)[1,0]
        print(correlations.max())
        retain_closest = np.where(correlations == correlations.max())[0]

        # strategy=apply gaussian filter to add variance over three foci
        X = mesh.points
        D = helpers.internal_distance_matrix(X)
        tree = KDTree(X)
        max_lag = np.nanmax(D)
        spdist = tree.sparse_distance_matrix(tree, max_distance=max_lag, output_type='coo_matrix')

        n_foci = 3

        random.seed(321)
        opoint_rd_indices = [ random.randint(0, mesh.n_points) for i in range(n_foci) ]
        opoints = [ mesh.points[opoint_rd_idx] for opoint_rd_idx in opoint_rd_indices ]

        weights = [ helpers.gaussian_sphere(X, opoint, 1) for opoint in opoints]

        neighbours = [tree.query(x=opoint, k=mesh.n_points - 1)[0] for opoint in opoints ]
        probabilities = [ weights[i] / np.sum(weights[i]) for i in range(len(neighbours)) ]

        # MAKE COVARIATES

        # strategy=add noise
        xk1 = x0 + np.random.uniform(low = -2, high = 2, size = x0.shape)
        # strategy=add filter
        xk2 = gaussian_filter1d(xk1,1, 0)
        # strategy=spun map
        xk3 = x0[spun[:,retain_closest]].flatten()
        # strategy=apply gaussian weights
        xk4 = x0*(np.sum(weights, axis = 0))
        #x5 = (x0**3 - x0.min()) / (x0.max() - x0.min())
        # strategy=thresholding
        xk6 = copy.copy(x0)
        xk6[np.where(y <= 0)[0]] = 0

        X_covariates = np.vstack([xk1, xk2, xk3, xk4, xk6])

        X_covariates = np.array(X_covariates, dtype = np.float64)
        Yk0 = X_covariates

    return Yk0


def run_one_iteration(datasource, curr_iter, \
                      all_sampling, \
                      Yk0, \
                      reduced_mesh, \
                      maptag, \
                      outdir,\
                      tags,\
                      verbose=1):
    
    
    r"""
    ARGUMENT TYPES
    ---------------
    samp_i: int
    sampling: 4-element tuple of ndarrays
    mesh: pvdata
    ground_truth: ndarray
    verbose: bool

    """
    print(curr_iter)
    strategy = tags.get('strategy')
    percentage = tags.get('percentage')
    geometry = tags.get('geometry')
    
    import copy
    import sys
    sys.path.append('../')
    from interpmodules import deterministic, geospatial, helpers, metrics
    import joblib
    import numpy as np
    import pandas as pd

    ground_truth = reduced_mesh[maptag]

    # unpack variables
    all_known_coords, all_known_verts, all_unknown_coords, all_unknown_verts = all_sampling
    known_coords = all_known_coords[:,:,curr_iter]
    known_verts = all_known_verts[:,curr_iter].astype(int)
    unknown_coords = all_unknown_coords[:,:,curr_iter]
    unknown_verts = all_unknown_verts[:,curr_iter].astype(int)

    RETURN_INFO = dict.fromkeys(steps)
    RETURN_INFO['determ_params'] = np.empty(4, dtype = 'object')
    RETURN_INFO['interpolated'] = dict.fromkeys(approaches)
    RETURN_INFO['interpolant_labels'] = []



    #=======assess sampling strategy======#
    va, eva, vase = helpers._get_voronoi_area_standard_error(X0=known_coords)
    if verbose: print(f'{strategy} SAMPLING \n VORONOI AREA S.E.= {str(vase)}')

    RETURN_INFO['voronoi'] = va, eva, vase
    #=====================================#



    #=======variography=========#
    Dist = helpers.internal_distance_matrix(v=known_coords, method = 'euclidean')
    input_vals = ground_truth[known_verts].flatten()
    input_coords = known_coords

    emp_h, emp_ev, emp_N = helpers.get_empirical_variogram_fast(x=input_vals,\
                                                                X=input_coords,\
                                                                D=Dist, nh=25, h_bounds=0)
    
    import skgstat as skg

    binned_vgm = skg.Variogram(coordinates=input_coords,\
                               values=input_vals,\
                               model='gaussian',\
                               n_lag=25, bin_func='even',use_nugget=True)
    
    x,y=binned_vgm.get_empirical(bin_center=True)
    bounds=(0.00001, [np.nanmax(x), np.nanmax(y), np.nanmax(y) / 10])
    binned_vgm.fit(force=True,method='trf',p0=bounds[1],maxfev=1000)
    fit_effective_range, fit_sill, fit_nugget = binned_vgm.parameters

    if verbose: print(f'c0={str(fit_nugget)}, sill={str(fit_sill)}, range={str(fit_effective_range)}')
    RETURN_INFO['variogram'] = (emp_h, emp_ev, emp_N, fit_effective_range, fit_sill, fit_nugget)
    #=========================#



    #=======interpolating=====#

    #=========================DETERMINISTIC=====================#
    RUN_TIMES = dict.fromkeys(approaches)

    X0 = input_coords
    x1 = unknown_coords
    Y0 = input_vals
    y0 = ground_truth[unknown_verts].flatten() # actual values

    if verbose:
        print('known coordinates array :' + str(X0.shape))
        print('UNKNOWN coordinates array :' + str(x1.shape))
        print('known values array :' + str(Y0.shape))
        print('UNKNOWN (actual) value array :' + str(y0.shape))

    # approach=inverse distance weighting, optimized for powers 1 through 10
    res_best, param_best, rt = deterministic.interpolate_and_optimize(method='idw', X0=X0, Y0=Y0, x1=x1, ground_truth=y0, bounds=[1,10])
    RETURN_INFO['determ_params'][0] = param_best
    RETURN_INFO['interpolated']['idw'] = res_best
    RETURN_INFO['interpolant_labels'].append(f'idw{int(param_best)}')
    RUN_TIMES['idw'] = rt

    # approach=k nearest neighbours, optimized for neighbours 2 through 20
    res_best, param_best, rt = deterministic.interpolate_and_optimize(method='knn', X0=X0, Y0=Y0, x1=x1, ground_truth=y0, bounds=[2,20])
    RETURN_INFO['determ_params'][1] = param_best
    RETURN_INFO['interpolated']['knn'] = res_best
    RETURN_INFO['interpolant_labels'].append(f'knn{int(param_best)}')
    RUN_TIMES['knn'] = rt

    # approach=radial basis function, optimized for neighbours 2 through 20 across kernel functions. see deterministic module for list of kernels
    # some kernels don't work for too few neighbours so adjustments were made in deterministic module
    res_best, param_best,rt = deterministic.interpolate_and_optimize(method='rbf', X0=X0, Y0=Y0, x1=x1, ground_truth=y0, bounds=[2,20])
    RETURN_INFO['determ_params'][2:] = param_best
    RETURN_INFO['interpolated']['rbf'] = res_best
    RETURN_INFO['interpolant_labels'].append(f'rbf{str(param_best[0])}{param_best[1]}')
    RUN_TIMES['rbf'] = rt



    #========================GEOSPATIAL=======================#
    # approach=kriging and regression kriging, ignore if variogram fit is unsatisfactory; it is expected that not all random samples at this sparsity works
    try:
        res_best, _, rt = geospatial.interpolate_Krige(X0=X0, Y0=Y0, x1=x1, x_mesh = reduced_mesh, method='Simple',timeit=True)
        RETURN_INFO['interpolated']['krige'] = res_best
        RUN_TIMES['krige'] = rt

        # subsampling covariates, no need to do this if cannot krige in the first place
        Yk = Yk0[:, known_verts]
        x1 = np.array(unknown_coords, dtype = np.float64)
        yk1 = Yk0[:,unknown_verts]

        res_best, _,rt = geospatial.interpolate_Krige(X0=X0, Y0=Y0, x1=x1, x_mesh = reduced_mesh, Yk=Yk, yk=yk1, method = 'Regression',timeit=True)
        RETURN_INFO['interpolated']['regkrige'] = res_best
        RUN_TIMES['regkrige'] = rt

    except ValueError:
        RETURN_INFO['interpolated']['krige'] = np.nan
        RETURN_INFO['interpolated']['regkrige'] = np.nan
        RUN_TIMES['krige'] = np.nan
        RUN_TIMES['regkrige'] = np.nan
        if verbose: print("could not fit a positive-variance gaussian model")
        
    # approach=spatially weighted regression. zero out unknown locations and use the learned coefficients at known locations to predict values
    X0_extended = reduced_mesh.points
    Y0_extended = copy.deepcopy(ground_truth)
    Y0_extended[unknown_verts] = 0
    Yk_extended = Yk0

    res_best, rt = geospatial.smoothing_over_GWR(X0=X0_extended, Y0=Y0_extended, Yk0=Yk_extended, bandwidth=None, timeit=True)
    res_best = res_best[unknown_verts]
    RETURN_INFO['interpolated']['swr'] = res_best
    RUN_TIMES['swr'] = rt



    #===============performance metrics=================#
    metrics_to_consider = ['pearson', 'spearman', 'ssim', \
                            'r2', 'rmse', 'kolmogorov_smirnov', \
                                'jensen_shannon', 'run_time']

    marrays = dict.fromkeys(approaches)

    for approach in marrays.keys():

        if verbose: print(ground_truth[unknown_verts])
        if np.isnan(RETURN_INFO['interpolated'][approach]).any(): marrays[approach] = np.full(7, np.nan); continue
        if verbose: print(RETURN_INFO['interpolated'][approach])

        marrays[approach] = np.array([ metrics.metric(ground_truth[unknown_verts], \
                            RETURN_INFO['interpolated'][approach], mi) for mi in metrics_to_consider[:-1] ], dtype = 'float32')


    df = pd.DataFrame(marrays, index=metrics_to_consider[:-1])
    
    
    scores = metrics.accuracy_rank(df)
    if verbose: print(scores)

    scores = np.mean(scores, axis = 1)
    
    print(scores)

    df.loc['run_time'] = RUN_TIMES

    if verbose: print(df)

    RETURN_INFO['performance'] = (df, scores)

    filename = outdir + f'/{template}_{maptag}_{geometry}_{datasource}_samp-{strategy}_pct-{str(percentage)}_{curr_iter}_all_info.pkl'

    if verbose: print(filename)

    joblib.dump(RETURN_INFO, filename)



def recombine_iterations(niters, curr_map, samp_i, percent_label, results_dir):
    
    import joblib


    #SAMPLING=complete_sampling

    #ALL_SAMPLING = {'train': SAMPLING[samp_i][1], 'test': SAMPLING[samp_i][-1]}
    ALL_VORONOI = dict.fromkeys(range(niters))
    ALL_VARIOGRAMS = dict.fromkeys(range(niters))
    ALL_DETERM_PARAMS = dict.fromkeys(range(niters))
    ALL_RESULTS = dict.fromkeys(range(niters))
    ALL_PERFORMANCE = dict.fromkeys(range(niters))
    ALL_LABELS = dict.fromkeys(range(niters))

    for curr_iter in range(niters):

        RETURN_INFO = joblib.load(results_dir + f'neuromaps_{curr_map}_samp-{samp_i}_pct-{percent_label}_{curr_iter}_all_info.pkl')
        ALL_VORONOI[curr_iter] = RETURN_INFO['voronoi']
        ALL_VARIOGRAMS[curr_iter] = RETURN_INFO['variogram']
        ALL_DETERM_PARAMS[curr_iter] = RETURN_INFO['determ_params']
        ALL_RESULTS[curr_iter] = RETURN_INFO['interpolated']
        ALL_PERFORMANCE[curr_iter] = RETURN_INFO['performance']
        ALL_LABELS[curr_iter] = RETURN_INFO['interpolant_labels']


    ALL_INFO = ALL_VORONOI, ALL_VARIOGRAMS, ALL_DETERM_PARAMS, ALL_RESULTS, ALL_PERFORMANCE, ALL_LABELS
    return ALL_INFO