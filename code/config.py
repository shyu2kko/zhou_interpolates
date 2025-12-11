'''
configuration file
'''

projdir = "/home/bic/yiguz/projects/interpolate/"
outdir = projdir + '/results/'

template = 'fsLR-4k'

approaches = ['idw', 'knn', 'rbf', 'swr', 'krige', 'regkrige']

test_percentages = [2, 5, 10, 25, 50]
test_ranges = [25, 50, 75, 100]
MAP_NAMES = ['abagen_genepc1',
 'hcps1200_megalpha',
 'hcps1200_megbeta',
 'hcps1200_megdelta',
 'hcps1200_meggamma1',
 'hcps1200_meggamma2',
 'hcps1200_megtheta',
 'hcps1200_megtimescale',
 'hcps1200_myelinmap',
 'hcps1200_thickness',
 'hill2010_devexp',
 'hill2010_evoexp',
 'margulies2016_fcgradient01',
 'mueller2013_intersubjvar',
 'raichle_cbf',
 'raichle_cbv',
 'raichle_cmr02',
 'raichle_cmrglc',
 'sydnor2021_SAaxis',
 'xu2020_FChomology',
 'xu2020_evoexp']

test_strategies = ['fibonacci', 'random', 'focaldiffuse']

metrics_to_consider = ['pearson', 'spearman', 'ssim', \
                        'r2', 'rmse', 'kolmogorov_smirnov', \
                        'jensen_shannon', 'run_time']

steps = ['voronoi', 'variogram', 'determ_params', 'interpolated', 'performance', 'interpolant_labels']



def fetch_polydata(geometry='sphere', space = 'fsLR-4k', datasource = 'grf'):
    
    import joblib
    
    pathstring = projdir + f'/data/{space}_{geometry}_{datasource}.pkl'
    var = joblib.load( pathstring )
    
    return var



def fetch_sample(geometry='sphere', strategy='fibonacci', percentage=10, niterations=100, datasource=None, maptag=None):
    
    import joblib
    
    sampdir = projdir + '/data/sampling/'
    pathstring = sampdir + f'/{template}_{geometry}_samp-{strategy}_pct-{str(percentage)}_iters-{niterations}.pkl'
    if strategy == 'fibonacci': pathstring = sampdir + f'/{template}_{geometry}_samp-{strategy}_pct-{str(percentage)}.pkl'

    var = joblib.load( pathstring )

    return var
