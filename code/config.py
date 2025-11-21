'''
configuration file
'''

projdir = "/home/yzhou/gitrepo/interpolate/"
outdir = projdir + '/results/'

template = 'fsLR-4k'

approaches = ['idw', 'knn', 'rbf', 'swr', 'krige', 'regkrige']

test_percentages = [2, 5, 10, 25, 50]
test_ranges = [25, 50, 75, 100]
test_strategies = ['fibonacci', 'random', 'focaldiffuse']

metrics_to_consider = ['pearson', 'spearman', 'ssim', \
                        'r2', 'rmse', 'kolmogorov_smirnov', \
                        'jensen_shannon', 'run_time']

steps = ['voronoi', 'variogram', 'determ_params', 'interpolated', 'performance']



def fetch_polydata(geometry='sphere', space = 'fsLR-4k', datasource = 'grf'):
    
    import joblib
    
    pathstring = projdir + f'/data/{space}_{geometry}_{datasource}.pkl'
    var = joblib.load( pathstring )
    
    return var



def fetch_sample(geometry='sphere', strategy='fibonacci', percentage=10, niterations=100, maptag=None):
    
    import joblib
    
    sampdir = projdir + '/data/sampling/'
    pathstring = sampdir + f'/{template}_{geometry}_samp-{strategy}_pct-{str(percentage)}_iters-{niterations}.pkl'
    if strategy == 'fibonacci': pathstring = sampdir + f'/{template}_{geometry}_samp-{strategy}_pct-{str(percentage)}.pkl'

    var = joblib.load( pathstring )

    return var