'''
function definition for a single run of interpolation assessment
'''
import sys
sys.path.append('../')

from config import *


def _get_dist_centers(geometry='sphere', strategy='fibonacci', \
                      percentage=10, niterations=100, datasource=None, maptag=None):
    
    import joblib
    import numpy as np
    if geometry == 'sphere':
        sphere_centroids = np.array([[100, 0, 0], \
                                     [0, 100, 0], \
                                     [0, 0, 100], \
                                     [-100, 0, 0], \
                                     [0, -100, 0], \
                                     [0, 0, -100]], dtype = 'float32')
        return sphere_centroids

    elif geometry in ['reduced-midthickness', 'left-volume', 'volume']:
    
        if strategy == 'lobes':
            _, _, centroids_lobes = joblib.load(projdir + '/data/fsLR-4k_midthickness_parc-lobes.pkl')
            return centroids_lobes
        
        _, _, centroids_dsk33 = joblib.load(projdir + '/data/fsLR-4k_midthickness_parc-dsk33.pkl')
        return centroids_dsk33
    
    else:
        print('not available, check your inputs')



def _get_compare_maptag(datasource, maptag):
    if datasource == 'neuromaps':
        return maptag
    
    elif datasource == 'microarray-pet':
        import pandas as pd
        import joblib
        mappings = pd.read_csv(projdir + '/data/beliveau2017_serotonergic_names_mappings.csv')
        tracer_tag = mappings[ mappings['gene'] == maptag.split('_')[-1] ]['tracer'].values[0]
        return f'pet_beliveau2017_{tracer_tag}'
        
    elif datasource in [ f"ieeg-{fband}-pet" for fband in fbands ]:
        
        return f"hcps1200_meg{maptag.split('_')[0]}"

    

    
def generate_covariates(curr_map, src_polydata, trg_polydata, nspins=1000, datasource='grf'):

    '''
    currmap: `str` that indexes the polydata scalar array
    polydata: 
    '''
    
    from interpmodules import helpers
    import copy
    from neuromaps import nulls
    import numpy as np
    import random
    from scipy.spatial import KDTree
    from scipy.ndimage import gaussian_filter1d
    

    ground_truth = src_polydata[curr_map]
    Yk0, yk1 = None, None

    if datasource == 'grf':
        y = copy.deepcopy(ground_truth)
        x0 = copy.copy(y)

        # strategy=most correlated spun map
        spun = nulls.spins.gen_spinsamples(coords = src_polydata.points, hemiid = np.zeros(src_polydata.n_points), n_rotate = nspins, seed = 321)
        correlations = np.zeros(nspins)
        for this_iter in range(spun.shape[1]):
            correlations[this_iter] = np.corrcoef(x0[spun[:,this_iter]], y)[1,0]
        print(correlations.max())
        retain_closest = np.where(correlations == correlations.max())[0]

        # strategy=apply gaussian filter to add variance over three foci
        X = src_polydata.points
        D = helpers.internal_distance_matrix(X)
        tree = KDTree(X)
        max_lag = np.nanmax(D)
        spdist = tree.sparse_distance_matrix(tree, max_distance=max_lag, output_type='coo_matrix')

        n_foci = 3

        random.seed(321)
        opoint_rd_indices = [ random.randint(0, src_polydata.n_points) for i in range(n_foci) ]
        opoints = [ src_polydata.points[opoint_rd_idx] for opoint_rd_idx in opoint_rd_indices ]

        weights = [ helpers.gaussian_sphere(X, opoint, 1) for opoint in opoints]

        neighbours = [tree.query(x=opoint, k=src_polydata.n_points - 1)[0] for opoint in opoints ]

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

        
    
    elif datasource in ['neuromaps', 'neuromaps_civet', 'microarray-pet'] + [ f"ieeg-{fband}-pet" for fband in fbands ]:

        # find top 5 correlates
        #----------STEP 2.1. Build correlation matrix between maps-------------#
        
        other_maps = src_polydata.array_names
        if datasource == 'microarray-pet': other_maps = src_polydata.array_names[5:]
        if datasource in [ f"ieeg-{fband}-pet" for fband in fbands ]: other_maps = src_polydata.array_names[1:]  

        CMAT = np.ones(len(other_maps))

        for j, other_map in enumerate(other_maps):
            
            if other_map == curr_map: continue

            CMAT[j] = np.ma.corrcoef(np.ma.masked_invalid(src_polydata[curr_map]), \
                                np.ma.masked_invalid(src_polydata[other_map]))[1,0]

        
        chosen_map_idx = abs(CMAT).argsort()
        if datasource in ['neuromaps', 'neuromaps_civet']:
            chosen_map_idx = chosen_map_idx[:-1]
        top5_cov = np.array(other_maps)[chosen_map_idx][-5:]
        print(curr_map)
        print(top5_cov)
        print(CMAT[chosen_map_idx[-5:]])

        X_covariates = np.vstack([src_polydata[top5_cov_i] for top5_cov_i in top5_cov])
        Yk0 = X_covariates

        if trg_polydata == None: return Yk0
        
        yk1 = np.vstack([trg_polydata[top5_cov_i] for top5_cov_i in top5_cov])

        return Yk0, yk1
    return Yk0, yk1




def run_one_iteration(datasource, curr_iter, \
                      all_sampling, \
                      Yk0, \
                      yk1, \
                      src_polydata, \
                      trg_polydata, \
                      benchmark_polydata, \
                      maptag, \
                      outdir,\
                      tags,\
                      usecase=False,\
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
    


    #import copy
    import sys
    sys.path.append('../')
    from interpmodules import deterministic, stochastic, helpers, metrics
    import joblib
    import numpy as np
    import pandas as pd
    

    RETURN_INFO = dict.fromkeys(steps)

    RETURN_INFO['determ_params'] = np.empty(4, dtype = 'object')
    RETURN_INFO['interpolated'] = dict.fromkeys(approaches)
    RETURN_INFO['interpolant_labels'] = []

    # check that trg_polydata is in the same space as benchmark_polydata
    same_output_space = np.array_equal( trg_polydata.points, benchmark_polydata.points)
    if not same_output_space: raise ValueError('Target space not identical to benchmark space. Please check your inputs.')

    
    if usecase:
        compare_maptag = _get_compare_maptag(datasource, maptag)
        if verbose: print(':::::::::::::::::::comparing use case against: ' + compare_maptag)
        benchmark = benchmark_polydata[compare_maptag]

    #ground_truth = src_polydata[maptag]

    # unpack variables
    if all_sampling != None:
        all_known_coords, all_known_verts, all_unknown_coords, all_unknown_verts = all_sampling
        known_coords = all_known_coords[:,:,curr_iter]
        known_verts = all_known_verts[:,curr_iter].astype(int)
        unknown_coords = all_unknown_coords[:,:,curr_iter]
        unknown_verts = all_unknown_verts[:,curr_iter].astype(int)

        #=======assess sampling strategy======#
        va, eva, vase = helpers._get_voronoi_area_standard_error(X0=known_coords)
        if verbose: print(f'{strategy} SAMPLING \n VORONOI AREA S.E.= {str(vase)}')

        RETURN_INFO['voronoi'] = va, eva, vase
        #=====================================#

    else:
        known_coords = src_polydata.points
        known_verts = np.arange(src_polydata.n_points)
        unknown_coords = trg_polydata.points
        unknown_verts = np.arange(trg_polydata.n_points)


    if not usecase: benchmark = src_polydata[maptag][unknown_verts]
    
    #=======variography=========#
    Dist = helpers.internal_distance_matrix(v=known_coords, method = 'euclidean')
    input_vals = src_polydata[maptag][known_verts].flatten()
    input_coords = known_coords

    emp_h, emp_ev, emp_N = helpers.get_empirical_variogram_fast(x=input_vals,\
                                                                X=input_coords,\
                                                                D=Dist, nh=25, h_bounds=0)
    
    import skgstat as skg

    if datasource == 'grf': cov_model = 'gaussian'
    elif datasource in ['neuromaps', 'microarray-pet'] + [ f"ieeg-{fband}-pet" for fband in fbands ]: cov_model = 'exponential'
    
    try:

        binned_vgm = skg.Variogram(coordinates=input_coords,\
                                values=input_vals,\
                                model=cov_model,\
                                n_lags=25, bin_func='even',use_nugget=True)


        x,y=binned_vgm.get_empirical(bin_center=True)
        bounds=(0.00001, [np.nanmax(x), np.nanmax(y), np.nanmax(y) / 10])
        binned_vgm.fit(force=True,method='trf',p0=bounds[1],maxfev=1000)
        fit_effective_range, fit_sill, fit_nugget = binned_vgm.parameters


        if verbose: print(f'c0={str(fit_nugget)}, sill={str(fit_sill)}, range={str(fit_effective_range)}')
    
    except RuntimeError:

        binned_vgm = helpers.get_empirical_variogram_fast(X=input_coords,\
                                                          x=input_vals, D=Dist,\
                                                          nh=25, method='euclidean')
        emp_h, emp_ev, emp_N = binned_vgm
        fit_effective_range, fit_sill, fit_nugget = np.nan, np.nan, np.nan

    RETURN_INFO['variogram'] = (emp_h, emp_ev, emp_N, fit_effective_range, fit_sill, fit_nugget)
    #=========================#



    #=======interpolating=====#

    #=========================DETERMINISTIC=====================#
    RUN_TIMES = dict.fromkeys(approaches)

    X0 = known_coords
    x1 = unknown_coords
    Y0 = input_vals
    y0 = src_polydata[maptag]
    if usecase:
        x1 = fslr_to_mni_coords_transform(unknown_coords)
    #else: y0 = src_polydata[maptag][unknown_verts].flatten() # actual values

    if verbose:
        print('known coordinates array :' + str(X0.shape))
        print('UNKNOWN coordinates array :' + str(x1.shape))
        print('known values array :' + str(Y0.shape))
        if not usecase: print('UNKNOWN (actual) value array :' + str(y0.shape))

    #centers = _get_dist_centers(**tags)
    
    # approach=inverse distance weighting, optimized for powers 2 through 20
    if verbose: print(f"----------IDW ongoing at iteration {curr_iter}")
    res_best, param_best,_, rt = deterministic._interpolate_and_optimize(method='idw', X0=X0, Y0=Y0, x1=x1, ground_truth=benchmark, bounds=[2,20], timeit=True)
    RETURN_INFO['determ_params'][0] = param_best
    RETURN_INFO['interpolated']['idw'] = res_best
    RETURN_INFO['interpolant_labels'].append(f'idw{int(param_best)}')
    RUN_TIMES['idw'] = rt

    # approach=k nearest neighbours, optimized for neighbours 2 through 20
    if verbose: print(f"----------KNN ongoing at iteration {curr_iter}")
    res_best, param_best,_, rt = deterministic._interpolate_and_optimize(method='knn', X0=X0, Y0=Y0, x1=x1, ground_truth=benchmark, bounds = [2,20], timeit=True)
    RETURN_INFO['determ_params'][1] = param_best
    RETURN_INFO['interpolated']['knn'] = res_best
    RETURN_INFO['interpolant_labels'].append(f'knn{int(param_best)}')
    RUN_TIMES['knn'] = rt

    # approach=radial basis function, optimized for neighbours 2 through 20 across kernel functions. see deterministic module for list of kernels
    # some kernels don't work for too few neighbours so adjustments were made in deterministic module
    if verbose: print(f"----------RBF ongoing at iteration {curr_iter}")
    res_best, param_best,_, rt = deterministic._interpolate_and_optimize(method='rbf', X0=X0, Y0=Y0, x1=x1, ground_truth=benchmark, bounds = [2,20], timeit=True)
    RETURN_INFO['determ_params'][2:] = param_best
    RETURN_INFO['interpolated']['rbf'] = res_best
    RETURN_INFO['interpolant_labels'].append(f'rbf{str(param_best[0])}{param_best[1]}')
    RUN_TIMES['rbf'] = rt


    
    #========================STOCHASTIC=======================#
    # approach=kriging and regression kriging, ignore if variogram fit is unsatisfactory; it is expected that not all random samples at this sparsity works

    # subsampling covariates
    Yk = Yk0[:, known_verts]
    if not usecase: yk1 = Yk0[:,unknown_verts]

    
    try:
        if verbose: print(f"----------Kriging ongoing at iteration {curr_iter}")
        res_best, _, rt = stochastic.interpolate_Krige(X0=known_coords, Y0=input_vals, x1=unknown_coords, method='Simple',timeit=True)
        RETURN_INFO['interpolated']['krige'] = res_best
        RUN_TIMES['krige'] = rt

        if verbose: print(f"----------Regression kriging ongoing at iteration {curr_iter}")
        res_best, _,rt = stochastic.interpolate_Krige(X0=known_coords, Y0=input_vals, x1=unknown_coords, Yk0=Yk, yk1=yk1, method = 'Regression',timeit=True)
        RETURN_INFO['interpolated']['regkrige'] = res_best
        RUN_TIMES['regkrige'] = rt

    except ValueError:
    
        RETURN_INFO['interpolated']['krige'] = np.nan
        RETURN_INFO['interpolated']['regkrige'] = np.nan
        RUN_TIMES['krige'] = np.nan
        RUN_TIMES['regkrige'] = np.nan
        if verbose: print(f"       >:(  00PS: could not fit a positive-variance ${cov_model} model !!       ")

    # approach=spatially weighted regression. zero out unknown locations and use the learned coefficients at known locations to predict values
    #X0_extended = reduced_mesh.points
    #Y0_extended = copy.deepcopy(ground_truth)
    #Y0_extended[unknown_verts] = 0
    #Yk_extended = Yk0

    if verbose: print(f"----------GWR ongoing at iteration {curr_iter}")
    #res_best, rt = stochastic.smoothing_over_GWR(X0=X0_extended, Y0=Y0_extended, Yk0=Yk_extended, bandwidth=None, timeit=True)
    #res_best = res_best[unknown_verts]
    #RETURN_INFO['interpolated']['swr'] = res_best
    #RUN_TIMES['swr'] = rt
    # fixed prediction at spatially-weighted regression

    try:
        res_best, rt = stochastic.interpolate_SWR(X0=known_coords, Y0=input_vals, x1=unknown_coords, Yk0=Yk, yk1=yk1, timeit=True)
        RETURN_INFO['interpolated']['swr'] = res_best
        RUN_TIMES['swr'] = rt
    except np.linalg.LinAlgError:
        RETURN_INFO['interpolated']['swr'] = np.nan
        RUN_TIMES['swr'] = np.nan
        if verbose: print("         >>:( ARGHHH: covariates are not linearly independent at this sample !!      ")

    #===============performance metrics=================#
    metrics_to_consider = ['pearson', 'spearman', 'ssim', \
                            'r2', 'rmse', 'kolmogorov_smirnov', \
                                'jensen_shannon', 'run_time']

    marrays = dict.fromkeys(approaches)

    for approach in marrays.keys():

        #if verbose: print(ground_truth[unknown_verts])
        if np.isnan(RETURN_INFO['interpolated'][approach]).any(): marrays[approach] = np.full(7, np.nan); continue
        if verbose: print(RETURN_INFO['interpolated'][approach])

        marrays[approach] = np.array([ metrics.metric(benchmark, \
                            RETURN_INFO['interpolated'][approach], mi) for mi in metrics_to_consider[:-1] ], dtype = 'float32')


    df = pd.DataFrame(marrays, index=metrics_to_consider[:-1])
    
    
    scores = metrics.accuracy_rank(df)
    if verbose: print(scores)

    scores = np.mean(scores, axis = 1)
    
    print(scores)

    df.loc['run_time'] = RUN_TIMES

    if verbose: print(df)

    RETURN_INFO['performance'] = (df, scores)

    import pathlib
    pathlib.Path(outdir + f'/{datasource}/{strategy}/').mkdir(parents=True, exist_ok=True) 

    filename = outdir + f'/{datasource}/{strategy}/{template}_{maptag}_{geometry}_{datasource}_samp-{strategy}_pct-{str(percentage)}_{curr_iter}_all_info.pkl'

    if verbose: print(filename)

    joblib.dump(RETURN_INFO, filename)




def recombine_iterations(datasource, niters, \
                      maptag, \
                      outdir,\
                      tags,\
                      verbose=1):
    
    import joblib

    strategy = tags.get('strategy')
    percentage = tags.get('percentage')
    geometry = tags.get('geometry')

    #SAMPLING=complete_sampling

    #ALL_SAMPLING = {'train': SAMPLING[samp_i][1], 'test': SAMPLING[samp_i][-1]}
    ALL_VORONOI = dict.fromkeys(range(niters))
    ALL_VARIOGRAMS = dict.fromkeys(range(niters))
    ALL_DETERM_PARAMS = dict.fromkeys(range(niters))
    ALL_RESULTS = dict.fromkeys(range(niters))
    ALL_PERFORMANCE = dict.fromkeys(range(niters))
    ALL_LABELS = dict.fromkeys(range(niters))

    for curr_iter in range(niters):

        RETURN_INFO = joblib.load(outdir + f'{template}_{maptag}_{geometry}_{datasource}_samp-{strategy}_pct-{percentage}_{curr_iter}_all_info.pkl')
        ALL_VORONOI[curr_iter] = RETURN_INFO['voronoi']
        ALL_VARIOGRAMS[curr_iter] = RETURN_INFO['variogram']
        ALL_DETERM_PARAMS[curr_iter] = RETURN_INFO['determ_params']
        ALL_RESULTS[curr_iter] = RETURN_INFO['interpolated']
        ALL_PERFORMANCE[curr_iter] = RETURN_INFO['performance']
        ALL_LABELS[curr_iter] = RETURN_INFO['interpolant_labels']


    ALL_INFO = ALL_VORONOI, ALL_VARIOGRAMS, ALL_DETERM_PARAMS, ALL_RESULTS, ALL_PERFORMANCE, ALL_LABELS
    header = ['voronoi', 'variogram', 'determ_params', 'interpolated', 'performance']
    
    return header, ALL_INFO
