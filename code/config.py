'''
configuration
'''


import numpy as np

projdir = "../"
outdir = projdir + '/results/'

template = 'fsLR-4k'

approaches = ['idw', 'knn', 'rbf', 'swr', 'krige', 'regkrige']

test_strategies = ['fibonacci', 'random', 'focaldiffuse']
test_percentages = [5, 10, 25, 50]
test_ranges = [25, 50, 75, 100]

metrics_to_consider = ['pearson', 'spearman', 'ssim', \
                        'r2', 'rmse', 'kolmogorov_smirnov', \
                        'jensen_shannon', 'run_time']

steps = ['voronoi', 'variogram', 'determ_params', 'interpolated', 'performance', 'interpolant_labels']

fbands = ['delta', 'theta', 'alpha', 'beta', 'gamma1', 'gamma2']

nlin6asym_to_2009csym = np.array([[0.9932, 0.0003, -0.0025, -0.0083],
                                  [-0.0005, 0.9935, -0.0029, 0.0499],
                                  [0.0017, 0.0060, 0.9764, 0.5897], 
                                  [0.000, 0.000, 0.000, 1.000]], dtype = 'float32')
nlin2009csym_to_6asym = np.linalg.inv(nlin6asym_to_2009csym)


# for plotting
mesh_aes = {"show_scalar_bar": False,\
            "smooth_shading": True,\
            "specular": 0, "diffuse":0, "ambient":1}
point_aes = {"render_points_as_spheres": True, "diffuse":0, "ambient":1,"specular":0}

def fslr_to_mni_coords_transform(coords, inverse=False):
    from nibabel.affines import apply_affine
    if inverse: return apply_affine(nlin2009csym_to_6asym, coords)
    return apply_affine(nlin6asym_to_2009csym, coords)


def fetch_polydata(geometry='sphere', space = 'fsLR-4k', datasource = 'grf'):
    '''
    geometry: 'sphere' 'reduced-midthickness' or 'volume'
    space: 'fsLR-4k' or 'icbm09c'
    datasource: 'grf' 'neurompas' or 'microarray'
    '''
    
    import joblib
    
    pathstring = projdir + f'/data/{space}_{geometry}_{datasource}.pkl'
    var = joblib.load( pathstring )
    
    return var



def fetch_sample(geometry='sphere', strategy='fibonacci', percentage=10, niterations=100, datasource=None, maptag=None):
    '''
    this only applies for surface interpolations
    '''
    import joblib
    
    sampdir = projdir + '/data/sampling/'
    pathstring = sampdir + f'/{template}_{geometry}_samp-{strategy}_pct-{str(percentage)}_iters-{niterations}.pkl'
    if strategy == 'fibonacci': pathstring = sampdir + f'/{template}_{geometry}_samp-{strategy}_pct-{str(percentage)}.pkl'

    var = joblib.load( pathstring )

    return var
