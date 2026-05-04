'''
helpers
'''

import copy
import gstools as gs
from joblib import Parallel, delayed
from math import pi, sqrt
import matplotlib.pyplot as plt
import neuromaps.points
import nibabel as nb
import numpy as np
import random
import pyvista as pv
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.spatial import SphericalVoronoi, KDTree
import time


def cv_distdep_noreplace(pvdata, center, nfolds, rseed=321):

    coords = pvdata.points

    pct = 1/nfolds

    npts = coords.shape[0]
    available = np.arange(npts)
    
    cv_test_samples, cv_test_dist = [], []
    cv_train_samples = []
    
    # take samples without replacement
    for fi in range(nfolds):

        if fi == nfolds - 1: 
            idx = available
            cv_test_samples.append(idx)
            di = np.linalg.norm( coords[idx] - center, axis = 1)
            cv_test_dist.append(di)
            break
            
        rng = np.random.default_rng(321)
    
        n_train = int(round(pct * npts))
    
        dist = np.linalg.norm( coords[available] - center, axis = 1)
    
        sigma = .5

        # weigh w/ a gaussian kernel
        
        scores = np.exp( -(dist / sigma )**2 )
        w = scores / scores.sum()
    
        idx = rng.choice( available, size = n_train, replace = False, p=w)
        cv_test_samples.append(idx)

        di = np.linalg.norm( coords[idx] - center, axis = 1)
        cv_test_dist.append(di)
        available = np.setdiff1d(available, idx)
    
    cv_test_verts = tuple(cv_test_samples)
    cv_test_coords = tuple( [ pvdata.points[test_vert_set,:] for test_vert_set in cv_test_verts ] )
    
    cv_train_verts = tuple( [ np.setdiff1d(np.arange(pvdata.n_points), test_vert_set) for test_vert_set in cv_test_verts] )
    cv_train_coords = tuple( [ pvdata.points[train_vert_set, :] for train_vert_set in cv_train_verts] )

    return cv_train_coords, cv_train_verts, cv_test_coords, cv_test_verts



def cv_random_noreplace(pvdata, nfolds, rseed=321):

    np.random.seed(rseed)
    
    n_test = int( 1/nfolds * pvdata.n_points )
    print(n_test)
    cv_test_samples, cv_train_samples = [], []

    available = np.arange(pvdata.n_points)

    for chi in range(nfolds):
        if chi == nfolds - 1:
            cv_test_samples.append(available)
            break
        random_sample = np.random.choice(available, size = n_test, replace=False)
        cv_test_samples.append(random_sample)
        available = np.setdiff1d(available, random_sample)

    cv_test_verts = tuple(cv_test_samples)
    cv_test_coords = tuple( [ pvdata.points[test_vert_set,:] for test_vert_set in cv_test_verts ] )
    
    cv_train_verts = tuple( [ np.setdiff1d(np.arange(pvdata.n_points), test_vert_set) for test_vert_set in cv_test_verts] )
    cv_train_coords = tuple( [ pvdata.points[train_vert_set, :] for train_vert_set in cv_train_verts] )

    return cv_train_coords, cv_train_verts, cv_test_coords, cv_test_verts
    

def gaussian_sphere(sphere_verts, center_vert, sigma=0.5):
    '''
    sphere_verts: in cartesian coordinates
    center_vert: in cartesian coordinates
    sigma: scalar
    '''
    # normalize
    center_vert = center_vert / np.linalg.norm(center_vert)
    sphere_vertss = sphere_verts / np.linalg.norm(sphere_verts, axis=1, keepdims=True)

    # compute angular dist in rad
    products = np.clip(np.dot(sphere_verts, center_vert), -1.0, 1.0)
    rads = np.arccos(products)

    # gaussian weights
    w = np.exp(-(rads**2) / (2 * sigma**2))
    
    return w


def _find_duplicates_across_arrays(list_of_arrays):
    '''
    list -> list
    '''
    all_dups = []
    
    for i in range(len(list_of_arrays)):
        for j in range(len(list_of_arrays)):
            if i == j: continue
            all_dups += (np.intersect1d(list_of_arrays[i], list_of_arrays[j]).tolist())
    
    return all_dups



def _recover_full_map(train_idx, test_idx, interpolated, sampled):
    nvtx = len(train_idx) + len(test_idx)
    full_map = np.zeros(nvtx)
    
    full_map[train_idx.flatten()] = sampled
    full_map[test_idx.flatten()] = interpolated
    return full_map




def _chunking_express(d, coords, n_chunks=5):
    '''
    d: distance vector
    coords: 3d coords for training points
    n_chunks: number of even chunks

    1d-array, 2d-array, int -> 2d-array, 1d-array
    '''

    # sort coordinates by distance (ascending)
    sort_ind = d.argsort(axis = 0).flatten()
    dist_sorted_coords = coords[ sort_ind ]
    chunk_size = d.shape[0] // n_chunks

    chunk_coords = []
    chunk_verts = []
    chunk_bounds = [0]

    for n in range(n_chunks):
        start = chunk_size * n
        end = chunk_size * (n + 1)
        
        if n == n_chunks - 1: end = d.shape[0]

        chunk_coords.append(dist_sorted_coords[start:end])
        chunk_verts.append(sort_ind[start:end])
        chunk_bounds.append(d.flatten()[sort_ind][end - 1])
        
    return tuple(chunk_coords), tuple(chunk_verts), tuple(chunk_bounds)



def get_training_set(mesh, sphere, pct_split, n_iters, parcellation_name=None, method='random', sphere_foci=None, seed=321):

    '''
    random sampling
    ---
    mesh: pyvista polydata with N points
    pct_split: 0 to 1

    return: four tuples
    (N * [1 - pct_split], 3, n_iters), (N * [1 - pct_split], n_iters),
    (N * pct_split, 3, n_iters), (N * pct_split, n_iters)
    
    '''
    
    nvtx = mesh.n_points
    
    n_train = int(nvtx * pct_split / 100)
    n_test = int(nvtx - n_train)

    known_coords = np.zeros((n_train, 3,n_iters))
    known_verts = np.zeros((n_train,n_iters))
    unknown_coords = np.zeros((n_test, 3, n_iters))
    unknown_verts = np.zeros((n_test, n_iters))

    random.seed(seed)
    np.random.seed(seed)

    if method == 'fibonacci':

        curr_iter = 0
        
        print(f'computing nearest solution to Fibonacci spherical sampling with {str(n_train)} vertices')

        # find nearest existing vertex to fibonacci sphere
        radius = abs(sphere.bounds[0])
        ideal_samp = _generate_fibonacci_sphere(n_train, radius)

        best_samp = []
        retain_points = []
        retain_verts = []
        consider_points = copy.deepcopy(sphere.points)
        consider_verts = list(range(nvtx))

        for i in range(n_train):
            best_samp = _find_nearest_point(ideal_samp[i], consider_points); print(best_samp)
            #print(str(best_samp) + ' | ' + str(consider_points.shape))
            retain_points.append(consider_points[best_samp])
            retain_verts.append(np.where(np.all(sphere.points == consider_points[best_samp], axis=1))[0])

            tmp = consider_verts.pop(best_samp)
            consider_points = np.delete(consider_points, best_samp,0)

        #print(n_train)

        known_verts[:,curr_iter] = np.array(retain_verts).flatten().astype(int)
        #known_coords[:,:,curr_iter] = np.array(retain_points)
        known_coords[:, :,curr_iter] = mesh.points[known_verts[:, curr_iter].astype(int)]
        unknown_verts[:,curr_iter] = np.array(consider_verts).flatten().astype(int)
        unknown_coords[:,:,curr_iter] = mesh.points[unknown_verts[:, curr_iter].astype(int)]


    if method == 'random':
        
        random_samples_coords = np.zeros((n_train, 3, n_iters)) # x,y,z coordinates
        random_samples_idx = np.zeros((n_train, n_iters))
        test_point_coords = np.zeros((nvtx-n_train, 3, n_iters))
        test_point_idx = np.zeros((nvtx-n_train, n_iters))
        
        for n_iter in range(n_iters):
            train_points = [None]*n_train
            
            c=0
            while None in train_points:
                new_point = random.randint(0,nvtx - 1)
                # avoid duplicates
                if new_point in train_points: continue
                else: train_points[c] = new_point; c+=1
            
            test_points = []
            for i in range(nvtx):
                if i not in train_points: test_points.append(i)
            
            X0_random = np.array(mesh.points[train_points], dtype = np.float64)
            x1 = np.array(mesh.points[test_points], dtype = np.float64)
            
            random_samples_coords[:,:,n_iter] = X0_random
            random_samples_idx[:,n_iter] = train_points
            test_point_coords[:,:, n_iter] = x1
            test_point_idx[:,n_iter] = test_points

        random_samples_idx = np.array(random_samples_idx, dtype = int)
        test_point_idx = np.array(test_point_idx, dtype = int)
    
        known_coords, known_verts, unknown_coords, unknown_verts = random_samples_coords, random_samples_idx, test_point_coords, test_point_idx    


    if method == 'focaldiffuse':

        tree = KDTree(sphere.points)

        parcellated = mesh[parcellation_name]

        rel_size = np.unique(parcellated, return_counts=True)[1] / nvtx # relative size for each parcel
        sample_counts = np.ceil(rel_size * n_train) # rounds to nearest integer

        

        for curr_iter in range(n_iters):

            sampled_verts = []

            for fi,focus in enumerate(sphere_foci):

                parcel_verts = np.where(parcellated == fi)[0]
                
                neighbours = tree.query(x = focus, k = parcel_verts.shape[0]) 

                whole_brain_weights = gaussian_sphere(sphere.points, focus, .5)
                in_parcel_weights = whole_brain_weights[parcel_verts]

                in_parcel_probabilities = in_parcel_weights / in_parcel_weights.sum()


                sampled_verts.append( np.random.choice(parcel_verts, size=round(sample_counts[fi]), \
                                                        replace=False, p=in_parcel_probabilities) )

            combined = np.unique(np.concatenate(sampled_verts, 0))[:int(n_train)]
            known_verts[:,curr_iter] = combined
            known_coords[:,:,curr_iter] = mesh.points[combined]
            unknown_verts[:,curr_iter] = np.array(np.setdiff1d(list(range(nvtx)), combined), dtype = int)
            unknown_coords[:,:,curr_iter] = mesh.points[np.array(unknown_verts[:,curr_iter], dtype=int)]


    return known_coords, known_verts, unknown_coords, unknown_verts



def _cartesian_to_spherical(X, unit):
    '''
    X: (*, 3) coordinates for however many samples
    unit: "degrees" or "radians"
    
    returns
    theta: longitude
    phi: latitude
    rho: radius
    '''

    rho = np.max(X)
    X = np.asarray(X, dtype=np.double).reshape(3,-1)
    theta = np.arctan2(X[1], X[0])
    phi = np.arcsin(np.maximum(np.minimum(X[2] / rho, 1.0), -1.0))
    
    if unit == 'radians': 
        return theta, phi, rho
    else:
        return np.rad2deg(theta), np.rad2deg(phi), rho



def _convert_to_length(r):
    '''
    r : range
    s : scaling factor
    '''
    
    s = sqrt(pi) / (sqrt(3) * 2)
    return s*r


def convert_to_range(l):
    s = sqrt(pi) / (sqrt(3) * 2)
    return l / s

def _get_distance_between(X0=None, x1=None, method = 'euclidean', polydata=None, x1_vert =None):
    '''
    X0: (N, 3) N-many sampled points, with x-y-z coordinates
    x1: (1,3) point to interpolate, with x-y-z coordinates

    also implement for geodesic distance
    '''

    if method == 'euclidean':
        from scipy.spatial.distance import cdist
    
        D = cdist(X0, x1, metric=method)
    if method == 'geodesic':
        if polydata == None: raise(ValueError, "please provide a mesh for geodesic distance")
        dist = internal_distance_matrix(polydata.points, polydata.faces, method ='geodesic')
        D = dist[x1_vert,:]
    return D



def internal_distance_matrix(v, f=None, method = 'euclidean', n_proc = -1):

    if method == 'euclidean':
        # plug the scipy method
        dist = _get_distance_between(v,v)
        
    elif method == 'geodesic':
        graph = neuromaps.points.make_surf_graph(v, f)

        n_vert = v.shape[0]
        labels, mask = None, np.zeros(n_vert, dtype=bool)
        dist = np.vstack(Parallel(n_jobs=n_proc, max_nbytes=None)(
            delayed(neuromaps.points._get_graph_distance)(n, graph, labels) for n in range(n_vert)
        ))

    return dist

def morans_i(dist, y, normalize=True, local=False, invert_dist=True):
    """
    Calculates Moran's I from distance matrix `dist` and brain map `y`
    Parameters
    ----------
    dist : (N, N) array_like
        Distance matrix between `N` regions / vertices / voxels / whatever
    y : (N,) array_like
        Brain map variable of interest
    normalize : bool, optional
        Whether to normalize rows of distance matrix prior to calculation.
        Default: False
    local : bool, optional
        Whether to calculate local Moran's I instead of global. Default: False
    invert_dist : bool, optional
        Whether to invert the distance matrix to generate a weight matrix.
        Default: True
    Returns
    -------
    i : float
        Moran's I, measure of spatial autocorrelation
    """

    # convert distance matrix to weights
    if invert_dist:
        with np.errstate(divide='ignore'):
            dist = 1 / dist
    np.fill_diagonal(dist, 0)

    # normalize rows, if desired
    if normalize:
        dist /= dist.sum(axis=-1, keepdims=True)

    # calculate Moran's I
    z = y - y.mean()
    if local:
        with np.errstate(all='ignore'):
            z /= y.std()

    zl = np.squeeze(dist @ z[:, None])
    den = (z * z).sum()

    if local:
        return (len(y) - 1) * z * zl / den

    return len(y) / dist.sum() * (z * zl).sum() / den
    
    

def _get_empirical_variogram(x,D,nh,h_bounds=0):
    '''
    calculate the empirical variogram from
    data `x` with internal distance matrix `D`

    prespecify number of bins with nh, usually 25 works okay (according to Vince)
    h_bounds are cutoffs for values in distance lag

    returns `h` distance lag (x-axis of semivariogram)
    and `ev` emppirical semivariance (y-axis of semivariogram)

    from Rob Leech Nature Neuro 2024
    '''
    
    n = x.size
    triu = np.triu_indices(n, k=1)  # upper triangular inds
    b=3

    diff_ij = np.subtract.outer(x, x)
    v = 0.5 * np.square(diff_ij)[triu]
    u=D[triu]

    if h_bounds==0:
        h = np.linspace(u.min(), u.max(), nh)
    else: 
        h = np.linspace(h_bounds[0], h_bounds[1], nh)

    # Subtract each h from each pairwise distance u
    # Each row corresponds to a unique h

    du = np.abs(u - h[:, None])
    w = np.exp(-np.square(2.68 * du / b) / 2)
    denom = np.nansum(w, axis=1)
    wv = w * v[None, :]
    num = np.nansum(wv, axis=1)
    ev = num / denom #empirical variogram
    
    return h,ev



def get_empirical_variogram_fast(x,X,D,nh,h_bounds='auto', method='euclidean'):
    '''
    calculate the empirical variogram from
    data `x` with coordinates `X`

    replace large adjacency matrix with KDTree

    returns `h` distance lag (x-axis of semivariogram)
    and `ev` empirical semivariance (y-axis of semivariogram)
    '''

    from scipy.sparse import coo_matrix

    max_lag = np.nanmax(D)
    
    if method == 'euclidean':
        tree = KDTree(X)
        spdist = tree.sparse_distance_matrix(tree, max_distance=max_lag, output_type='coo_matrix')
    elif method == 'geodesic':
        spdist = coo_matrix(D)
    
    i_idx = spdist.row
    j_idx = spdist.col
    dist_ij = spdist.data
    
    var = .5 * (x[i_idx] - x[j_idx])**2

    import pandas as pd
    
    # Create dataframe
    df = pd.DataFrame({
        'h': dist_ij,
        'ev': var
    })
    
    # Define bins

    bins = np.linspace(0.000001, max_lag, nh + 1)
    
    if isinstance(h_bounds, np.ndarray):
        bins = h_bounds

    df['bins'] = pd.cut(df['h'], bins)
    
    # count values per bin
    N = df.groupby('bins')['ev'].count().values
    # Average semivariance per bin
    empirical_variogram = df.groupby('bins')['ev'].mean().values
    #bin_centers = 0.5 * (bins[:-1] + bins[1:])

    h = (bins[:-1] + bins[1:]) / 2
    ev = empirical_variogram
    
    return h, ev, N



def hole(h, r, c0,nugget):
    '''
    Theoretical variogram functions from Rob Leech
    
    J. Bessel's hole-effect model

    h: distance bounds
    r: range
    c0: partial sill i.e. sill - nugget
    '''
    a = r/3
    theoVar=nugget + c0*(1-np.sin(h/a)/(h/a))

    return theoVar

def gaussian(h, r, c0, nugget):
    '''
    as above
    '''
    a = r / 2
    return nugget + c0 * (1. - np.exp(- (h ** 2 / a ** 2)))

    import math

def exponential(h,r,c0,nugget):
    '''
    as above
    '''
    theoVar=nugget+c0*(1-np.exp(-h/(r/3)))
    return theoVar
    


def fit_empirical_variogram(x,D,X,smoothing=False, method = 'best', fast=True, nh=25,h_bounds=0, summary=True, plotit=False, filename=None):

    models = [exponential, gaussian, hole]
    mdl_names = ['exponential', 'gaussian', 'hole']

    if method != 'best': models = [models[mdl_names.index(method)]]
    nparams=3; nmodels = len(models)
    
    if fast: u, ev, counts = get_empirical_variogram_fast(x=x,X=X,D=D,nh=nh,h_bounds=h_bounds)
    else: u, ev = _get_empirical_variogram(x=x,D=D,nh=nh,h_bounds=h_bounds)
    
    # smoothing the empirical variogram
    if smoothing: 
        smoothing_scale = 1; 
        u = gaussian_filter1d(u, sigma = smoothing_scale)
        ev = gaussian_filter1d(ev, sigma=smoothing_scale)
    
    # instantiate fit summary
    global_spatial_info = {}
    
    # concatenate model fit diagnostics
    model_cof_map_ev = np.zeros((nmodels, nparams))
    model_pred_map_ev = np.zeros((nmodels, nh))
    model_sse_map_ev = np.zeros((nmodels,))
    
    x_long=np.linspace(0,np.max(D), num=nh)

    bounds = ([0.00001]*nparams, [np.nanmax(u), np.nanmax(ev), np.nanmax(ev)/nh])
    
    cof = None; cov = None;
    
    for ind, model in enumerate(models):
        cof, cov = curve_fit(model, u, ev, method = 'trf', p0=bounds[1], bounds = bounds, maxfev=1000)
        v_pred = model(x_long, *cof)
        v_pred2=model(u,*cof)
        model_cof_map_ev[ind,:] = cof
        model_pred_map_ev[ind,:] = v_pred
        model_sse_map_ev[ind] = np.sum((v_pred2 - ev)**2)
        gof = model_sse_map_ev
        
    best = np.where(gof == np.min(gof))[0][0]
    global_spatial_info['vario_model'] = mdl_names[best]
    
    est_range = model_cof_map_ev[best,0]
    est_sill = model_cof_map_ev[best,1] + model_cof_map_ev[best,2]
    global_spatial_info['vario_range'] = est_range
    global_spatial_info['vario_sill'] = est_sill
    global_spatial_info['vario_nugget'] = model_cof_map_ev[best,2]
    
    global_spatial_info['sse'] = model_sse_map_ev
    global_spatial_info['pred'] = model_pred_map_ev
    global_spatial_info['cof'] = model_cof_map_ev
    global_spatial_info['models'] = mdl_names

    saveit = filename != None
    
    if plotit:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        ax.scatter(u, ev, color = 'k', label = 'empirical')
        ax.plot(x_long, global_spatial_info['pred'][best], color = 'magenta', label = global_spatial_info['vario_model'])

        ax.axhline(global_spatial_info['vario_sill'], color = 'magenta', linestyle = 'dotted')
        ax.axvline(global_spatial_info['vario_range'], color = 'magenta', linestyle = 'dotted')
        ax.legend(loc = 'lower right')
        ax.set_ylabel('semivariance')
        ax.set_xlabel('distance')

        if saveit:
            fig.savefig(filename)
        else:
            plt.show()
    
    return global_spatial_info



def _get_inverse_distance_weights(dist):
    
    with np.errstate(divide='ignore'):
        weights = 1. / dist
    isinf = np.isinf(weights)
    infrow = np.where(isinf)
    weights[infrow] = isinf[infrow]

    return weights



def _local_cdist(coords_i, coords, rho):
    """
    Compute Haversine distance matrix
    """
    dLat = np.radians(coords[:, 1] - coords_i[1])
    dLon = np.radians(coords[:, 0] - coords_i[0])
    lat1 = np.radians(coords[:, 1])
    lat2 = np.radians(coords_i[1])
    a = np.sin(
        dLat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dLon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    R = rho
    return R * c



def gii_to_polydata(gii_data):
    '''
    load a gifti to a polydata for pyvista
    '''
    verts, faces = gii_data
    return pv.PolyData(verts, np.c_[np.ones((faces.shape[0],),dtype = int)*3, faces])



def make_random_maps(ran, mesh, seed):
    
    l = _convert_to_length(ran)

    model = gs.Gaussian(dim = 3, len_scale=l, var=1.0)
    srf = gs.SRF(model)
    random_map = srf.mesh(mesh, points='points', seed=seed)

    return random_map


def _coords_icosahedron_to_perfect_sphere(vertices, radius=1.0):
    '''
    project (x,y,z) coordinates at vertices sampled from a icosahedron
    to a perfect sphere with pre-specified radius
    '''
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    return radius * vertices / norms
    

def _generate_fibonacci_sphere(n_points, rho):
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle
    points = []

    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2
        rho_xy = np.sqrt(1 - y * y) # start with unit sphere
        theta = phi * i
        x = np.cos(theta) * rho_xy
        z = np.sin(theta) * rho_xy

        points.append((x * rho, y * rho, z * rho)) # scale

    return np.array(points)



def _get_expected_voronoi_area(n_points, rho=100):
    '''
    for a given number of sampling points,
    calculate what the expected voronoi area would be if
    the sphere was uniformly sampled
    based on a Fibonacci sphere

    the radius of an icosahedral fsaverage5 spherical mesh is 100
    '''

    points = _generate_fibonacci_sphere(n_points, rho)
    
    sv = SphericalVoronoi(points, radius = rho)
    A = sv.calculate_areas()
    E = np.mean(A)
    
    return E



def _get_voronoi_area_standard_error(X0):
    '''
    small value means uniformly sampled
    large value means non-uniform
    '''

    n_points = X0.shape[0]
    samp_coords = _coords_icosahedron_to_perfect_sphere(X0, radius = 100)
    samp_sv = SphericalVoronoi(samp_coords, radius = 100)
    samp_A = samp_sv.calculate_areas()
    
    E = _get_expected_voronoi_area(n_points=len(X0))

    se = sqrt(np.sum(abs(samp_A - E)) / n_points)

    return samp_A, E, se



def _find_nearest_point(point, points):
    '''
    given a point find the nearest point from a set of points
    '''

    points = np.asarray(points)
    point = np.asarray(point)
    
    distances = np.sqrt(np.sum((points - point)**2, axis=1))
    return np.argmin(distances)