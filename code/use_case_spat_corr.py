'''
CRITERION II
correlation w/ spatial nulls
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



space = "fsLR-4k"
strategy = "all"
percentage = "100"


#-------------INTRACRANIAL EEG-------------

datasource = 'ieeg-alpha_no_outlier'
maptag = 'alpha_no_outlier'
compare_maptag = 'hcps1200_megalpha'

reduced_mesh = joblib.load(projdir + '/data/fsLR-4k_reduced-midthickness_neuromaps.pkl')
results = joblib.load(projdir + f'/results/{datasource}/{strategy}/{space}_{maptag}_volume_{datasource}_samp-{strategy}_pct-{percentage}_0_all_info.pkl')



# initiate Moran spectral randomization instance with the interpolating mesh
D = helpers.internal_distance_matrix(v=reduced_mesh.points, f=None, method = 'euclidean', n_proc = -1)
msr = MoranRandomization(n_rep=1000, procedure='singleton', tol=1e-6, random_state=321)
msr.fit(D)



emp_r = dict.fromkeys(approaches)
pvals = dict.fromkeys(approaches)
all_nulldistr = dict.fromkeys(approaches)

for approach in approaches:

    dt = results['interpolated'][approach]

    emp_r[approach] = np.corrcoef( results['interpolated'][approach].flatten(), reduced_mesh[compare_maptag] )[1,0]

    mrnulls = msr.randomize(dt)
    nulldistr = [ abs(np.corrcoef( mrnulls[i,:], reduced_mesh[compare_maptag]) )[1,0] for i in range(1000)]
    all_nulldistr[approach] = np.array(nulldistr)
    

    pvals[approach] = np.mean(np.abs(nulldistr) >= np.abs(emp_r[approach]))

nulldistr_plotdata = pd.DataFrame(all_nulldistr)
df = pd.DataFrame([emp_r, pvals], index = ['r', 'p'])
df = df.T
is_significant = df['p'] < .05

joblib.dump((df, nulldistr_plotdata), projdir + f'/results/use_case_ieeg_alpha_nulldistr.pkl')


#-------------MICROARRAY-------------#

mappings = pd.read_csv(projdir + '/data/beliveau2017_serotonergic_names_mappings.csv')

datasource = 'microarray-pet'
maptag = f"mrna_HTR1A"
compare_maptag = f"pet_beliveau2017_cumi101"

reduced_mesh = joblib.load(projdir + "/data/fsLR-4k_reduced-midthickness_beliveau.pkl")
results = joblib.load(projdir + \
                  f'/results/{datasource}/{strategy}/{space}_mrna_{gene}_cic-volume_{datasource}_samp-{strategy}_pct-{percentage}_0_all_info.pkl')



D = helpers.internal_distance_matrix(v=reduced_mesh.points, f=None, method = 'euclidean', n_proc = -1)
msr = MoranRandomization(n_rep=1000, procedure='singleton', tol=1e-6, random_state=321)
msr.fit(D)



emp_r = dict.fromkeys(approaches)
pvals = dict.fromkeys(approaches)
all_nulldistr = dict.fromkeys(approaches)



for approach in approaches:

    dt = results['interpolated'][approach]
    emp_r[approach] = np.corrcoef( results['interpolated'][approach].flatten(), reduced_mesh[compare_maptag] )[1,0]

    mrnulls = msr.randomize(dt)
    nulldistr = [ abs(np.corrcoef( mrnulls[i,:], reduced_mesh[compare_maptag]) )[1,0] for i in range(1000)]
    all_nulldistr[approach] = np.array(nulldistr)

    pvals[approach] = np.mean(np.abs(nulldistr) >= np.abs(emp_r[approach]))

nulldistr_plotdata = pd.DataFrame(all_nulldistr)
df = pd.DataFrame([emp_r, pvals], index = ['r', 'p'])
df = df.T
is_significant = df['p'] < .05

joblib.dump((df, nulldistr_plotdata), projdir + f'/results/stats/use_case_microarray_HTR1A_nulldistr.pkl')