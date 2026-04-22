'''
spatial nulls
'''

import sys
sys.path.append('../')

from brainspace.null_models import MoranRandomization
from config import *
from figs_config import *
from interpmodules import helpers
import joblib
import matplotlib.pyplot as plt
import pyvista as pv
import seaborn as sns
import pandas as pd



mappings = pd.read_csv(projdir + '/data/beliveau2017_serotonergic_names_mappings.csv')
space = "fsLR-4k"
strategy = "all"
percentage = "100"
datasource = 'microarray-pet'

# load colour palette
blue = plt.cm.winter(0.7)
green = plt.cm.summer(0.5)
yellow = plt.cm.spring(1.0)

bgy,_ = three_colours_gradient(lo=blue, mid=green, hi=yellow)
gene_colours = bgy(np.linspace(0.0, 1.0, len(mappings['gene'])))

reduced_mesh = joblib.load(projdir + "/data/fsLR-4k_reduced-midthickness_beliveau.pkl")

# initiate Moran spectral randomization
D = helpers.internal_distance_matrix(v=reduced_mesh.points, f=None, method = 'euclidean', n_proc = -1)
msr = MoranRandomization(n_rep=1000, procedure='singleton', tol=1e-6, random_state=321)
msr.fit(D)

print(D.shape)
# collect spatial nulls
plot = False

fig, axs = plt.subplots(2,3, figsize = (12, 8), sharey=True, sharex=True)

for gi, gene in enumerate(mappings['gene']):

    row = gi // 3
    col = gi % 3
    
    emp_r = dict.fromkeys(approaches)
    pvals = dict.fromkeys(approaches)
    all_nulldistr = dict.fromkeys(approaches)

    compare_maptag = f"pet_beliveau2017_{mappings['tracer'][gi]}"
    maptag = f"mrna_{gene}"
    results = joblib.load(projdir + \
                          f'/results/{datasource}/{strategy}/{space}_mrna_{gene}_cic-volume_{datasource}_samp-{strategy}_pct-{percentage}_0_all_info.pkl')    

    
    if plot:
        # null distribution histograms - at gene level
        fig1, axs1 = plt.subplots(2,3, figsize = (8,3))
    for ai, approach in enumerate(approaches):

        if plot: row, col = ai // 3, ai % 3

        dt = results['interpolated'][approach]
        if not isinstance(results['interpolated'][approach], np.ndarray): emp_r[approach] = 0; pvals[approach] = 1; all_nulldistr[approach] = np.zeros(1000); continue
            
        else: emp_r[approach] = np.corrcoef( results['interpolated'][approach].flatten(), reduced_mesh[compare_maptag] )[1,0]

        
        mrnulls = msr.randomize(dt)
        nulldistr = [ abs(np.corrcoef( mrnulls[i,:], reduced_mesh[compare_maptag]) )[1,0] for i in range(1000)]
        all_nulldistr[approach] = np.array(nulldistr)

        
        pvals[approach] = np.mean(np.abs(nulldistr) >= np.abs(emp_r[approach]))

        if plot:
            axs1[row, col].set_title(approach)
            axs1[row, col].hist(nulldistr, facecolor = 'lightgray', bins = 25)
            axs1[row, col].axvline(emp_r[approach], color = gene_colours[gi])
        
    if plot:
        plt.tight_layout()
        plt.savefig(projdir + f'/figs/main/use_case_microarray_compare_pet_{gene}_histograms.svg')
        plt.close(fig1)
    
    nulldistr_plotdata = pd.DataFrame(all_nulldistr)
    df = pd.DataFrame([emp_r, pvals], index = ['r', 'p'])
    df = df.T
    is_significant = df['p'] < .05

    sns.boxplot(nulldistr_plotdata, linecolor = 'k', color = 'lightgray', ax=axs[row, col])
    axs[row, col].scatter(
        df.index[~is_significant],
        df['r'][~is_significant],
        color="w",
        marker = 'd',
        edgecolor = 'k',
        s=100,
        zorder=5,
        label="Not significant"
    )
    axs[row, col].scatter(
        df.index[is_significant],
        df['r'][is_significant],
        color=gene_colours[gi],
        marker = 'd',
        edgecolor = 'k',
        s=100,
        zorder=5,
        label="Significant"
    )
    axs[row, 0].set_ylabel("Pearson's R")
    joblib.dump((df, nulldistr_plotdata), projdir + f'/results/use_case_microarray_{gene}_nulldistr.pkl')

plt.tight_layout()
plt.savefig(projdir + f'/figs/main/use_case_microarray_compare_pet_moran_nulls_boxplot.svg')
